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
import time
import multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_calculations import feature_functions

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

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
    if not os.path.exists(absolute_config_path):
        print(f"Configuration file {absolute_config_path} does not exist.")
        sys.exit(1)
    with open(absolute_config_path, 'r') as file:
        config = json.load(file)
    if 'max_trials' not in config:
        config['max_trials'] = 50
        print("'max_trials' not found in config. Using default value of 50.")
    if 'sequence_length' not in config:
        config['sequence_length'] = 30
        print("'sequence_length' not found in config. Using default value of 30.")
    if 'learning_rate_scheduler' not in config:
        config['learning_rate_scheduler'] = {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "verbose": True
        }
        print("'learning_rate_scheduler' not found in config. Using default ReduceLROnPlateau settings.")
    if 'gradient_clipping' not in config:
        config['gradient_clipping'] = 1.0
        print("'gradient_clipping' not found in config. Using default value of 1.0.")
    if 'mixed_precision' not in config:
        config['mixed_precision'] = True
        print("'mixed_precision' not found in config. Enabling mixed precision by default.")
    return config

def load_data(dataset_dir):
    accel_path = os.path.join(dataset_dir, 'accel.csv')
    keys_path = os.path.join(dataset_dir, 'keys.csv')
    if not os.path.exists(accel_path):
        print(f"Accelerometer data file {accel_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(keys_path):
        print(f"Keys data file {keys_path} does not exist.")
        sys.exit(1)
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)
    accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'])
    keys_data['timestamp'] = pd.to_datetime(keys_data['timestamp'])
    accel_data = accel_data.sort_values(by='timestamp').reset_index(drop=True)
    keys_data = keys_data.sort_values(by='timestamp').reset_index(drop=True)
    return accel_data, keys_data

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
    ).dropna(subset=["key"]).reset_index(drop=True)
    for axis in ['x', 'y', 'z']:
        denoised_col = f'{axis}_denoised'
        merged_data[denoised_col] = apply_fft_denoise(merged_data[axis].values, cutoff=0.1, sampling_rate=sampling_rate)
    for axis in ['x', 'y', 'z']:
        merged_data[axis] = merged_data[f'{axis}_denoised']
    for feature in selected_features:
        if feature in ['x', 'y', 'z']:
            continue
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
    return features_df.dropna().reset_index(drop=True)

def prepare_data(merged_data):
    X = merged_data[selected_features].values
    y = merged_data["key"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if np.isnan(X_scaled).any():
        print("NaN values found in features.")
        sys.exit(1)
    if np.isinf(X_scaled).any():
        print("Infinite values found in features.")
        sys.exit(1)
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
            print(f"Unsupported model type: {model_type}")
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
    def __init__(self, patience=10):
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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_dataloader(dataset, batch_size, shuffle, num_workers=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False
    )

def print_progress(trial_number, fold, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, batch_idx=None, total_batches=None, partial_train_loss=None, partial_train_acc=None):
    if batch_idx is not None and total_batches is not None and partial_train_loss is not None and partial_train_acc is not None:
        message = (f"Trial {trial_number}, Fold {fold}, Epoch {epoch} - "
                   f"Batch {batch_idx}/{total_batches} | "
                   f"Train Loss: {partial_train_loss:.4f}, Train Acc: {partial_train_acc:.4f}")
    else:
        message = (f"Trial {trial_number}, Fold {fold}, Epoch {epoch}/{total_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"\r{message}", end='', flush=True)

def train_fold(model, optimizer, criterion, scaler, train_loader, device, cleanup_interval, fold, trial_number, epoch, total_epochs, gradient_clipping):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader, 1):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == y_batch)
        if batch_idx % cleanup_interval == 0:
            partial_train_loss = total_loss / (batch_idx * train_loader.batch_size)
            partial_train_acc = correct.double() / (batch_idx * train_loader.batch_size)
            print_progress(trial_number, fold, epoch, total_epochs, total_loss, correct, None, None, batch_idx, len(train_loader), partial_train_loss, partial_train_acc)
            del X_batch, y_batch, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()
    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct.double() / len(train_loader.dataset)
    return train_loss, train_acc

def validate_fold(model, criterion, scaler, val_loader, device, scheduler=None):
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
    if scheduler is not None:
        scheduler.step(val_loss)
    return val_loss, val_acc

def objective(trial, config, X, y, num_classes, model_type, k_folds=5):
    hidden_size = trial.suggest_int('hidden_size', 50, 150, step=25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epochs = config['epochs']
    patience = config['patience']
    gradient_clipping = config['gradient_clipping']
    lr_scheduler_config = config['learning_rate_scheduler']
    mixed_precision = config['mixed_precision']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    val_accuracies = []

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
        model = RNNModel(model_type, X_train.shape[2], hidden_size, dropout_rate, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = GradScaler(enabled=mixed_precision)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = get_dataloader(train_dataset, batch_size, shuffle=True, num_workers=mp.cpu_count() // 2)
        val_loader = get_dataloader(val_dataset, batch_size, shuffle=False, num_workers=mp.cpu_count() // 2)
        early_stopping = EarlyStopping(patience=patience)
        if lr_scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_config.get('mode', 'min'),
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5),
                verbose=lr_scheduler_config.get('verbose', True)
            )
        else:
            print(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
            sys.exit(1)
        cleanup_interval = max(1, len(train_loader) // 10)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_fold(
                model, optimizer, criterion, scaler, train_loader, device,
                cleanup_interval, fold, trial.number, epoch, epochs, gradient_clipping
            )
            val_loss, val_acc = validate_fold(model, criterion, scaler, val_loader, device, scheduler)
            print_progress(trial.number, fold, epoch, epochs, train_loss, train_acc, val_loss, val_acc)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Trial {trial.number}, Fold {fold}: Early stopping at epoch {epoch}")
                break
            trial.report(val_acc.item(), epoch)
            if trial.should_prune():
                print(f"Trial {trial.number}, Fold {fold}: Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()
        print()
        val_accuracies.append(val_acc.item())
        del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset, scaler, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    avg_val_acc = np.mean(val_accuracies)
    print(f"Trial {trial.number}: Average Validation Accuracy: {avg_val_acc:.4f}")
    return avg_val_acc

def evaluate_and_save_model(model, X_test, y_test, label_encoder, scaler, model_save_path, mixed_precision):
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=mp.cpu_count() // 2)
    criterion = nn.CrossEntropyLoss()
    scaler_eval = GradScaler(enabled=mixed_precision)
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
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    num_classes = len(label_encoder.classes_)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(lambda trial: objective(trial, config, X_seq, y_seq, num_classes, args.model, k_folds=5), n_trials=config['max_trials'], n_jobs=2)
    print(f"\nBest hyperparameters: {study.best_params}")
    best_trial = study.best_trial
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, random_state=42, stratify=y_seq
    )
    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"X_test shape: {X_test.shape}")
    train_dataset = TensorDataset(X_train_full, y_train_full)
    train_loader = get_dataloader(train_dataset, batch_size=best_trial.params['batch_size'], shuffle=True, num_workers=mp.cpu_count() // 2)
    model = RNNModel(
        args.model,
        X_train_full.shape[2],
        best_trial.params['hidden_size'],
        best_trial.params['dropout_rate'],
        num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_trial.params['learning_rate'])
    scaler_final = GradScaler(enabled=config['mixed_precision'])
    early_stopping_final = EarlyStopping(patience=config['patience'])
    lr_scheduler_config = config['learning_rate_scheduler']
    if lr_scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_scheduler_config.get('mode', 'min'),
            factor=lr_scheduler_config.get('factor', 0.5),
            patience=lr_scheduler_config.get('patience', 5),
            verbose=lr_scheduler_config.get('verbose', True)
        )
    else:
        print(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
        sys.exit(1)
    print("Starting final training on combined training and validation sets...")
    cleanup_interval = max(1, len(train_loader) // 10)
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        train_loss, train_acc = train_fold(
            model, optimizer, criterion, scaler_final, train_loader, device,
            cleanup_interval, fold='Final', trial_number='Final', epoch=epoch, total_epochs=config['epochs'],
            gradient_clipping=config['gradient_clipping']
        )
        val_loss, val_acc = validate_fold(model, criterion, scaler_final, train_loader, device, scheduler)
        print_progress('Final', 'Final', epoch, config['epochs'], train_loss, train_acc, val_loss, val_acc)
        early_stopping_final(val_loss)
        if early_stopping_final.early_stop:
            print(f"Final Training: Early stopping triggered at epoch {epoch}")
            break
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds.")
    print()
    model_save_path = os.path.join('./data/model/deep_learning/', f'{args.model}.pt')
    evaluate_and_save_model(
        model, X_test, y_test, label_encoder, scaler, model_save_path, mixed_precision=config['mixed_precision']
    )
    del model, optimizer, criterion, train_loader, train_dataset, scaler_final, scheduler
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
