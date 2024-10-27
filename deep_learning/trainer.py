# trainer.py
import os
import json
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold, train_test_split
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
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
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
                logging.error(f"Feature '{feature}' not found in feature_functions.")
                sys.exit(1)
    required_features = selected_features + ["key"]
    missing_features = [feat for feat in required_features if feat not in merged_data.columns]
    if missing_features:
        logging.error(f"Missing features after computation: {missing_features}")
        sys.exit(1)
    return merged_data

def extract_features_dynamically(data, selected_features):
    if not selected_features:
        logging.error("No features selected for extraction.")
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
    if np.isnan(X_scaled).any():
        logging.error("NaN values found in features.")
        sys.exit(1)
    if np.isinf(X_scaled).any():
        logging.error("Infinite values found in features.")
        sys.exit(1)
    # Keep tensors on CPU
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
    def __init__(self, model_type, input_size, hidden_size, dropout_rate, num_classes, num_layers=1):
        super(RNNModel, self).__init__()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            logging.error(f"Unsupported model type: {model_type}")
            sys.exit(1)
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

def print_epoch_progress(fold, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
    message = (f"Fold {fold}, Epoch {epoch}/{total_epochs} - "
               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"\r{message}", end='', flush=True)

def print_partial_progress(fold, trial_number, epoch, batch_idx, total_batches, partial_train_loss, partial_train_acc):
    message = (f"Trial {trial_number}, Fold {fold}, Epoch {epoch} - "
               f"Batch {batch_idx}/{total_batches} | "
               f"Train Loss: {partial_train_loss:.4f}, Train Acc: {partial_train_acc:.4f}")
    print(f"\r{message}", end='', flush=True)

def train_fold(model, optimizer, criterion, scaler, train_loader, device, cleanup_interval, fold, trial_number, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader, 1):
        # Move data to GPU
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
        if batch_idx % cleanup_interval == 0:
            partial_train_loss = total_loss / (batch_idx * train_loader.batch_size)
            partial_train_acc = correct.double() / (batch_idx * train_loader.batch_size)
            print_partial_progress(fold, trial_number, epoch, batch_idx, len(train_loader), partial_train_loss, partial_train_acc)
            del X_batch, y_batch, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()
    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct.double() / len(train_loader.dataset)
    return train_loss, train_acc

def validate_fold(model, criterion, scaler, val_loader, device):
    model.eval()
    total_val_loss = 0
    correct_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Move data to GPU
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
    return val_loss, val_acc

def objective(trial, config, X, y, num_classes, model_type, k_folds=5):
    hidden_size = trial.suggest_int('hidden_size', 50, 150, step=25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    val_accuracies = []
    logging.info(f"Trial {trial.number}: Starting training with hidden_size={hidden_size}, "
                 f"dropout_rate={dropout_rate:.2f}, learning_rate={learning_rate:.6f}, batch_size={batch_size}")
    
    # Convert X and y to CPU and NumPy if they're tensors
    if isinstance(X, torch.Tensor):
        X_cpu = X.cpu().numpy()
    else:
        X_cpu = X
    if isinstance(y, torch.Tensor):
        y_cpu = y.cpu().numpy()
    else:
        y_cpu = y
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cpu, y_cpu), 1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        try:
            model = RNNModel(model_type, X_train.shape[2], hidden_size, dropout_rate, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
                persistent_workers=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=6,
                persistent_workers=False
            )
            early_stopping = EarlyStopping(patience=config['patience'])
            scaler = GradScaler()
            cleanup_interval = max(1, len(train_loader) // 10)
            for epoch in range(1, config['epochs'] + 1):
                train_loss, train_acc = train_fold(
                    model, optimizer, criterion, scaler, train_loader, device,
                    cleanup_interval, fold, trial.number, epoch, config['epochs']
                )
                val_loss, val_acc = validate_fold(model, criterion, scaler, val_loader, device)
                print_epoch_progress(fold, epoch, config['epochs'], train_loss, train_acc, val_loss, val_acc)
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    logging.info(f"Trial {trial.number}, Fold {fold}: Early stopping at epoch {epoch}")
                    break
            print()
            val_accuracies.append(val_acc.item())
            del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset, scaler
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logging.warning(f"Trial {trial.number}, Fold {fold}: OOM encountered. Pruning trial.")
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                raise e
    torch.cuda.empty_cache()
    avg_val_acc = np.mean(val_accuracies)
    logging.info(f"Trial {trial.number}: Average Validation Accuracy: {avg_val_acc:.4f}")
    return avg_val_acc

def evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler, model_save_path):
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,  # Re-enabled pin_memory
        num_workers=4      # Adjust based on your CPU cores
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move data to GPU
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
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test Loss: {test_loss:.4f}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to: {model_save_path}")
    scaler_path = os.path.splitext(model_save_path)[0] + '_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to: {scaler_path}")
    label_encoder_path = os.path.splitext(model_save_path)[0] + '_label_encoder.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logging.info(f"Label Encoder saved to: {label_encoder_path}")

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
    logging.info(f"X_seq shape: {X_seq.shape}")
    logging.info(f"y_seq shape: {y_seq.shape}")
    num_classes = len(label_encoder.classes_)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, config, X_seq, y_seq, num_classes, args.model, k_folds=5), n_trials=config['max_trials'])
    logging.info(f"\nBest hyperparameters: {study.best_params}")
    best_trial = study.best_trial
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, random_state=42, stratify=y_seq
    )
    logging.info(f"X_train_full shape: {X_train_full.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    train_dataset = TensorDataset(X_train_full, y_train_full)
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=True,
        pin_memory=True,  # Re-enabled pin_memory
        num_workers=4      # Adjust based on your CPU cores
    )
    scaler_final = GradScaler()
    early_stopping_final = EarlyStopping(patience=config['patience'])
    logging.info("Starting final training on combined training and validation sets...")
    cleanup_interval = max(1, len(train_loader) // 10)
    model = RNNModel(
        args.model,
        X_train_full.shape[2],
        best_trial.params['hidden_size'],
        best_trial.params['dropout_rate'],
        num_classes
    ).to(device)
    logging.info(f"Model input_size: {X_train_full.shape[2]}")
    logging.info(f"Sequence length: {X_train_full.shape[1]}")
    logging.info(f"Number of features: {X_train_full.shape[2]}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_trial.params['learning_rate'])
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        train_loss, train_acc = train_fold(
            model, optimizer, criterion, scaler_final, train_loader, device,
            cleanup_interval, fold='Final', trial_number=study.best_trial.number, epoch=epoch, total_epochs=config['epochs']
        )
        val_loss, val_acc = validate_fold(model, criterion, scaler_final, train_loader, device)
        print_epoch_progress('Final', epoch, config['epochs'], train_loss, train_acc, val_loss, val_acc)
        end_time = time.time()
        epoch_time = end_time - start_time
        early_stopping_final(val_loss)
        if early_stopping_final.early_stop:
            logging.info(f"Final Training: Early stopping triggered at epoch {epoch}")
            break
    print()
    evaluate_and_save_model(
        model, X_test, y_test, label_encoder, scaler,
        model_save_path=os.path.join('./data/model/deep_learning/', f'{args.model}.pt')
    )
    del model, optimizer, criterion, train_loader, train_dataset, scaler_final
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = [
        "rms", "rmse", "x_min", "x_max", "y_min", "y_max",
        "z_min", "z_max", "magnitude_min", "magnitude_max", "sma",
        "displacement_2d"
    ]
    main()
