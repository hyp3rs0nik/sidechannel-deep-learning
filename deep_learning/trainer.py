# trainer.py
import os
import json
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from scipy.fft import rfft, irfft
import gc
import pickle
from torch.cuda.amp import autocast, GradScaler
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_calculations import feature_functions

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def load_config(model_type):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, f'{model_type}_config.json')
    with open(absolute_config_path, 'r') as file:
        return json.load(file)

def load_data(dataset_dir):
    accel_path = os.path.join(dataset_dir, 'accel.csv')
    keys_path = os.path.join(dataset_dir, 'keys.csv')
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)
    accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'])
    keys_data['timestamp'] = pd.to_datetime(keys_data['timestamp'])
    return accel_data.sort_values(by='timestamp'), keys_data.sort_values(by='timestamp')

def apply_fft_denoise(data, cutoff=0.1, sampling_rate=100):
    data_fft = rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1./sampling_rate)
    data_fft[np.abs(frequencies) > cutoff] = 0
    return irfft(data_fft, n=len(data))

def merge_data(accel_data, keys_data, tolerance='100ms', sampling_rate=100, selected_features=None):
    if selected_features is None:
        selected_features = ["x", "y", "z"]
    merged_data = pd.merge_asof(
        accel_data.sort_values("timestamp"),
        keys_data.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance)
    ).dropna(subset=["key"])
    for axis in ['x', 'y', 'z']:
        denoised_col = f'{axis}_denoised'
        merged_data[denoised_col] = apply_fft_denoise(merged_data[axis].values, cutoff=0.1, sampling_rate=sampling_rate)
    for feature in selected_features:
        if feature in ['x', 'y', 'z']:
            merged_data[feature] = merged_data[f'{feature}_denoised']
        else:
            if feature in feature_functions:
                merged_data[feature] = feature_functions[feature](merged_data)
            else:
                print(f"Feature '{feature}' not found in feature_functions.")
                sys.exit(1)
    required_features = selected_features + ["key"]
    missing_features = [feat for feat in required_features if feat not in merged_data.columns]
    if missing_features:
        print(f"Missing features after computation: {missing_features}")
        sys.exit(1)
    return merged_data

def extract_features_dynamically(data, selected_features):
    if not selected_features:
        print("No features selected for extraction.")
        sys.exit(1)
    feature_data = {feature: data[feature] for feature in selected_features}
    features_df = pd.DataFrame(feature_data)
    features_df["key"] = data["key"].values
    return features_df.dropna()

def prepare_data(merged_data):
    X = merged_data[selected_features].values
    y = merged_data["key"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    return X_tensor, y_tensor, label_encoder, scaler

def create_sequences(X, y, sequence_length):
    sequences = []
    labels = []
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i + sequence_length])
        labels.append(y[i + sequence_length])
    return torch.stack(sequences), torch.tensor(labels)

class RNNModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, dropout_rate, num_classes):
        super(RNNModel, self).__init__()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return
        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def objective(trial, config, X_train, X_val, y_train, y_val, num_classes, model_type):
    hidden_size = trial.suggest_int('hidden_size', 50, 150, step=25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = RNNModel(model_type, X_train.shape[2], hidden_size, dropout_rate, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=0)
    early_stopping = EarlyStopping(patience=config['patience'])
    scaler = GradScaler()
    
    # Print trial start information
    print(f"Trial {trial.number}: Starting training with hidden_size={hidden_size}, dropout_rate={dropout_rate:.2f}, learning_rate={learning_rate:.6f}")
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)
        
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct.double() / len(train_loader.dataset)
        
        model.eval()
        total_val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                total_val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == y_batch)
        
        val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = correct_val.double() / len(val_loader.dataset)
        
        # Print epoch progress on the same line
        print(f"\rTrial {trial.number}, Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end='', flush=True)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nTrial {trial.number}: Early stopping at epoch {epoch+1}")
            break
    
    # Ensure the final epoch metrics are printed on a new line
    if not early_stopping.early_stop:
        print()
    
    # Clean up to free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_acc.item()

def evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler, model_save_path):
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)
    test_loss = total_loss / len(test_loader.dataset)
    test_acc = correct.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save scaler and label encoder for future use
    scaler_path = os.path.splitext(model_save_path)[0] + '_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    label_encoder_path = os.path.splitext(model_save_path)[0] + '_label_encoder.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label Encoder saved to: {label_encoder_path}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['lstm', 'gru'], required=True, help='Type of model to use: lstm or gru')
    parser.add_argument('--dataset', default='./data/training/', help='Path to dataset folder containing accel.csv and keys.csv')
    args = parser.parse_args()
    
    config = load_config(args.model)
    accel_data, keys_data = load_data(args.dataset)
    merged_data = merge_data(
        accel_data,
        keys_data,
        tolerance=config.get('merge_tolerance', '100ms'),
        sampling_rate=config.get('sampling_rate', 100),
        selected_features=selected_features
    )
    features_df = extract_features_dynamically(merged_data, selected_features)
    X, y, label_encoder, scaler = prepare_data(features_df)
    X_seq, y_seq = create_sequences(X, y, config['sequence_length'])
    
    # Ensure y_seq is a NumPy array for stratification
    X_seq_np = X_seq.numpy()
    y_seq_np = y_seq.numpy()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq_np, y_seq_np, test_size=0.3, random_state=42, stratify=y_seq_np
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Convert back to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    num_classes = len(label_encoder.classes_)
    
    # Create an Optuna study with MedianPruner for efficient hyperparameter optimization
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, config, X_train, X_val, y_train, y_val, num_classes, args.model), config['max_trials'])
    
    print(f"\nBest hyperparameters: {study.best_params}")
    
    best_trial = study.best_trial
    
    # Combine training and validation sets for final training
    X_final_train = torch.cat((X_train, X_val), dim=0).to(device, non_blocking=True)
    y_final_train = torch.cat((y_train, y_val), dim=0).to(device, non_blocking=True)
    
    model = RNNModel(
        args.model,
        X_final_train.shape[1],  # sequence_length
        best_trial.params['hidden_size'],
        best_trial.params['dropout_rate'],
        num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_trial.params['learning_rate'])
    train_dataset = TensorDataset(X_final_train, y_final_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=0)
    scaler_final = GradScaler()
    early_stopping_final = EarlyStopping(patience=config['patience'])
    
    print("Starting final training on combined training and validation sets...")
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler_final.scale(loss).backward()
            scaler_final.step(optimizer)
            scaler_final.update()
            total_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct.double() / len(train_loader.dataset)
        # Print epoch progress on the same line
        print(f"\rFinal Training Epoch {epoch+1}/{config['epochs']} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}", end='', flush=True)
        
        # Optional: Implement early stopping if desired
        # early_stopping_final(train_loss)
        # if early_stopping_final.early_stop:
        #     print(f"\nFinal Training: Early stopping triggered at epoch {epoch+1}")
        #     break
    
    # Ensure the final epoch metrics are printed on a new line
    print()
    
    # Evaluate on test set
    evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler, model_save_path='./data/model/deep_learning/' + args.model + '.pt')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = [
        "rms", "rmse", "x_min", "x_max", "y_min", "y_max",
        "z_min", "z_max", "magnitude_min", "magnitude_max", "sma",
        "displacement_2d"
    ]
    main()
