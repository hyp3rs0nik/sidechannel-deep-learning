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
import gc
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
import multiprocessing as mp
import signal
import resource

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from feature_calculations import feature_functions

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def set_ulimit(new_limit):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if new_limit > hard:
            print(f"Requested ulimit {new_limit} exceeds hard limit {hard}. Setting to hard limit.")
            new_limit = hard
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        print(f"Set ulimit -n: {new_limit}")
    except ValueError as ve:
        print(f"Error setting ulimit: {ve}")
    except Exception as e:
        print(f"Unexpected error setting ulimit: {e}")

def signal_handler(sig, frame):
    print('Interrupt received, saving current state...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

set_seed()

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, "lstm_config.json")
    if not os.path.exists(absolute_config_path):
        print(f"Configuration file {absolute_config_path} does not exist.")
        sys.exit(1)
    with open(absolute_config_path, "r") as file:
        config = json.load(file)

    config.setdefault("max_trials", 50)
    config.setdefault("sequence_length", 30)
    config.setdefault(
        "learning_rate_scheduler",
        {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "verbose": True,
        },
    )
    config.setdefault("gradient_clipping", 1.0)
    config.setdefault("mixed_precision", True)
    config.setdefault("epochs", 50)
    config.setdefault("patience", 10)
    config.setdefault("merge_tolerance", "100ms")
    config.setdefault("sampling_rate", 100)
    config.setdefault("batch_size", [32, 64, 128, 256, 512, 1024, 2048, 4096])
    config.setdefault("learning_rate", [0.006, 0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625])
    config.setdefault("num_workers", 4)  
    config.setdefault("ulimit_n", 4096)
    config.setdefault("n_folds", 5)
    config.setdefault("denoise", False)
    return config

def load_data(dataset_dir):
    accel_path = os.path.join(dataset_dir, "accel.csv")
    keys_path = os.path.join(dataset_dir, "keys.csv")
    if not os.path.exists(accel_path):
        print(f"Accelerometer data file {accel_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(keys_path):
        print(f"Keys data file {keys_path} does not exist.")
        sys.exit(1)
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)
    accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"])
    keys_data["timestamp"] = pd.to_datetime(keys_data["timestamp"])
    accel_data = accel_data.sort_values(by="timestamp").reset_index(drop=True)
    keys_data = keys_data.sort_values(by="timestamp").reset_index(drop=True)
    return accel_data, keys_data

def apply_fft_denoise_cpu(data, cutoff=0.1, sampling_rate=100):
    data_fft = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / sampling_rate)
    data_fft[np.abs(frequencies) > cutoff] = 0
    denoised = np.fft.irfft(data_fft, n=len(data))
    return denoised

def merge_data(
    accel_data, keys_data, tolerance="100ms", sampling_rate=100, selected_features=None, denoise=False 
):
    if selected_features is None:
        selected_features = ["x", "y", "z"]
    merged_data = (
        pd.merge_asof(
            accel_data.sort_values("timestamp"),
            keys_data.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )
        .dropna(subset=["key"])
        .reset_index(drop=True)
    )
    if denoise:
        for axis in ["x", "y", "z"]:
            denoised_col = f"{axis}_denoised"
            denoised = apply_fft_denoise_cpu(
                merged_data[axis].values, cutoff=0.1, sampling_rate=sampling_rate
            )
            merged_data[denoised_col] = denoised
        for axis in ["x", "y", "z"]:
            merged_data[axis] = merged_data[f"{axis}_denoised"]
    for feature in selected_features:
        if feature in ["x", "y", "z"]:
            continue
        else:
            if feature in feature_functions:
                merged_data[feature] = feature_functions[feature](merged_data)
            else:
                print(f"Feature '{feature}' not found in feature_functions.")
                sys.exit(1)
    required_features = selected_features + ["key"]
    missing_features = [
        feat for feat in required_features if feat not in merged_data.columns
    ]
    if missing_features:
        print(f"Missing features after computation: {missing_features}")
        sys.exit(1)
    return merged_data

def extract_features_dynamically_cpu(data, selected_features):
    if not selected_features:
        print("No features selected for extraction.")
        sys.exit(1)
    features_df = data[selected_features].copy()
    features_df["key"] = data["key"].values
    return features_df.dropna().reset_index(drop=True)

def prepare_data_cpu(features_df, selected_features, device):
    X = features_df[selected_features].values
    y = features_df["key"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    return X_tensor, y_tensor, label_encoder, scaler

def create_sequences_cpu(X, y, sequence_length):
    sequences = X.unfold(0, sequence_length, 1)[:-1]
    labels = y[sequence_length:]
    return sequences, labels

class RNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_rate,
        num_classes,
        num_layers=2,
        bidirectional=True,
    ):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_optimized_dataloader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        generator=torch.Generator(device="cpu"),
    )

def print_epoch_report(
    trial_number,
    fold,
    epoch,
    total_epochs,
    max_trials,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    show_insights=False,
    timing_info=None,
):
    message = (
        f"Trial {trial_number}/{max_trials}, Fold {fold}, Epoch {epoch}/{total_epochs} - "
        f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}"
    )
    print(message.ljust(120), end="\r", flush=True)
    if show_insights and timing_info:
        timing_message = " | ".join([f"{k}: {v:.2f} ms" for k, v in timing_info.items()])
        print(f"Insights: {timing_message}")

def train_fold(
    model,
    optimizer,
    criterion,
    scaler,
    train_loader,
    device,
    cleanup_interval,
    fold,
    trial_number,
    epoch,
    total_epochs,
    gradient_clipping,
    accumulation_steps=4,
    use_progress_bar=True,
    show_insights=False, 
):
    model.train()
    total_loss = 0
    correct = 0
    optimizer.zero_grad()

    forward_time = 0
    backward_time = 0
    optimization_time = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader, 1):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        start_forward = time.perf_counter()
        with autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) / accumulation_steps
        end_forward = time.perf_counter()
        forward_time += (end_forward - start_forward)

        start_backward = time.perf_counter()
        scaler.scale(loss).backward()
        end_backward = time.perf_counter()
        backward_time += (end_backward - start_backward)

        if batch_idx % accumulation_steps == 0:
            start_opt = time.perf_counter()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            end_opt = time.perf_counter()
            optimization_time += (end_opt - start_opt)

        total_loss += loss.item() * accumulation_steps * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == y_batch)

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct.double() / len(train_loader.dataset)

    timing_info = {}
    if show_insights:
        timing_info = {
            "Forward Pass": forward_time * 1000, 
            "Backward Pass": backward_time * 1000,
            "Optimization Step": optimization_time * 1000,
        }

    return train_loss, train_acc, timing_info

def validate_fold(model, criterion, scaler, val_loader, device, scheduler=None, use_progress_bar=True, show_insights=False):
    model.eval()
    total_val_loss = 0
    correct_val = 0

    forward_time = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            start_forward = time.perf_counter()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            end_forward = time.perf_counter()
            forward_time += (end_forward - start_forward)

            total_val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == y_batch)

    val_loss = total_val_loss / len(val_loader.dataset)
    val_acc = correct_val.double() / len(val_loader.dataset)
    if scheduler is not None:
        scheduler.step(val_loss)

    timing_info = {}
    if show_insights:
        timing_info = {
            "Validation Forward Pass": forward_time * 1000,
        }

    return val_loss, val_acc, timing_info

def objective(trial, config, X, y, num_classes, k_folds=5, device="cuda", use_progress_bar=True, show_insights=False):

    batch_size = trial.suggest_categorical("batch_size", config["batch_size"])
    learning_rate = trial.suggest_categorical("learning_rate", config["learning_rate"])

  
    print(f"Selected Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    epochs = config["epochs"]
    patience = config["patience"]
    gradient_clipping = config["gradient_clipping"]
    lr_scheduler_config = config["learning_rate_scheduler"]
    mixed_precision = config["mixed_precision"]
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)
    val_accuracies = []
    X_cpu = X.cpu().numpy()
    y_cpu = y.cpu().numpy()
    step_counter = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cpu, y_cpu), 1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        model = RNNModel(
            X_train.shape[2], 
            hidden_size=trial.suggest_int("hidden_size", 100, 300, step=50),
            dropout_rate=trial.suggest_float("dropout_rate", 0.2, 0.5),
            num_classes=num_classes,
            num_layers=2,
            bidirectional=True,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler_obj = GradScaler(enabled=mixed_precision)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = get_optimized_dataloader(
            train_dataset, batch_size, shuffle=True, num_workers=config["num_workers"]
        )
        val_loader = get_optimized_dataloader(
            val_dataset, batch_size, shuffle=False, num_workers=config["num_workers"]
        )
        early_stopping = EarlyStopping(patience=patience)
        if lr_scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_config.get("mode", "min"),
                factor=lr_scheduler_config.get("factor", 0.5),
                patience=lr_scheduler_config.get("patience", 5),
                verbose=lr_scheduler_config.get("verbose", True),
            )
        else:
            print(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
            sys.exit(1)
        cleanup_interval = max(1, len(train_loader) // 10)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.perf_counter() 
            try:
                train_loss, train_acc, train_timing = train_fold(
                    model,
                    optimizer,
                    criterion,
                    scaler_obj,
                    train_loader,
                    device,
                    cleanup_interval,
                    fold,
                    trial.number,
                    epoch,
                    epochs,
                    gradient_clipping,
                    accumulation_steps=4,
                    use_progress_bar=use_progress_bar,
                    show_insights=show_insights,
                )
                val_loss, val_acc, val_timing = validate_fold(
                    model, criterion, scaler_obj, val_loader, device, scheduler, use_progress_bar=use_progress_bar, show_insights=show_insights
                )
                epoch_end_time = time.perf_counter() 
                epoch_duration_ms = (epoch_end_time - epoch_start_time) * 1000 

                if show_insights:
                    combined_timing = {
                        "Epoch Time": epoch_duration_ms
                    }
                    combined_timing.update(train_timing)
                    combined_timing.update(val_timing)
                else:
                    combined_timing = None

                print_epoch_report(
                    trial.number,
                    fold,
                    epoch,
                    epochs,
                    config["max_trials"],
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    show_insights=show_insights,
                    timing_info=combined_timing,
                )
            
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(
                        f"\nTrial {trial.number}, Fold {fold}: Early stopping at epoch {epoch}"
                    )
                    break
                step_counter += 1
                trial.report(val_acc.item(), step_counter)
                if trial.should_prune():
                    print(
                        f"\nTrial {trial.number}, Fold {fold}: Pruned at epoch {epoch}"
                    )
                    raise optuna.exceptions.TrialPruned()
            except Exception as e:
                print(f"Error during training: {e}")
                raise e
        val_accuracies.append(val_acc.item())

        del (
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
            scaler_obj,
            scheduler,
        )
        gc.collect()
        torch.cuda.empty_cache()

    avg_val_acc = np.mean(val_accuracies)
    print(f"Trial {trial.number}: Average Validation Accuracy: {avg_val_acc:.4f}")
    return avg_val_acc

def evaluate_and_save_model(
    model,
    X_test,
    y_test,
    label_encoder,
    scaler,
    model_save_path,
    mixed_precision,
    device="cuda",
):
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = get_optimized_dataloader(
        test_dataset, batch_size=128, shuffle=False, num_workers=config["num_workers"]
    )
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
    scaler_path = os.path.splitext(model_save_path)[0] + "_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    label_encoder_path = os.path.splitext(model_save_path)[0] + "_label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label Encoder saved to: {label_encoder_path}")

def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to: {filepath}")

def load_checkpoint(checkpoint_dir, filename, device):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint
    else:
        return None

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="./data/training/",
        help="Path to dataset folder containing accel.csv and keys.csv",
    )
    parser.add_argument(
        "--report",
        choices=["detailed", "simple"],
        default="simple",
        help="Reporting mode: 'detailed' for progress bar (default), 'simple' for per-epoch reports without progress bar",
    )
    parser.add_argument(
        "--insights",
        action="store_true",
        help="Enable detailed timing insights for each epoch",
    )
    args = parser.parse_args()
    report_mode = args.report
    use_progress_bar = False 
    show_insights = args.insights 

    config = load_config()

    set_ulimit(config.get("ulimit_n", 4096)) 

    accel_data, keys_data = load_data(args.dataset)
    merged_data = merge_data(
        accel_data,
        keys_data,
        tolerance=config.get("merge_tolerance", "100ms"),
        sampling_rate=config.get("sampling_rate", 100),
        selected_features=selected_features,
        denoise=config.get("denoise", False) 
    )
    features_df = extract_features_dynamically_cpu(merged_data, selected_features)
    X, y, label_encoder, scaler = prepare_data_cpu(
        features_df, selected_features, device
    )
    X_seq, y_seq = create_sequences_cpu(X, y, config["sequence_length"])

    num_classes = len(torch.unique(y_seq))

    storage_path = os.path.join("./data/optuna_studies/", "lstm_optuna_study.db")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="lstm_optuna_study",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )


    study.optimize(
        lambda trial: objective(
            trial,
            config,
            X_seq,
            y_seq,
            num_classes,
            k_folds=config["n_folds"],
            device=device,
            use_progress_bar=use_progress_bar,
            show_insights=show_insights,
        ),
        n_trials=config["max_trials"],
        n_jobs=1,
    )

    print(f"\nBest hyperparameters: {study.best_params}")
    best_trial = study.best_trial

    batch_size = best_trial.params["batch_size"]
    learning_rate = best_trial.params["learning_rate"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, random_state=42, stratify=y_seq
    )
    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"X_test shape: {X_test.shape}")
    train_dataset = TensorDataset(X_train_full, y_train_full)
    train_loader = get_optimized_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"], 
    )
    model = RNNModel(
        X_train_full.shape[2],
        hidden_size=best_trial.params.get("hidden_size", 150), 
        dropout_rate=best_trial.params.get("dropout_rate", 0.3),
        num_classes=num_classes,
        num_layers=2,
        bidirectional=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler_final = GradScaler(enabled=config["mixed_precision"])


    checkpoint_dir = "./checkpoints/final_training/"
    checkpoint_filename = "final_training_checkpoint.pth"

    checkpoint = load_checkpoint(checkpoint_dir, checkpoint_filename, device)
    start_epoch = 1

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler_final.load_state_dict(checkpoint['scaler_state_dict'])
        early_stopping_final = checkpoint['early_stopping']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["learning_rate_scheduler"].get("mode", "min"),
            factor=config["learning_rate_scheduler"].get("factor", 0.5),
            patience=config["learning_rate_scheduler"].get("patience", 5),
            verbose=config["learning_rate_scheduler"].get("verbose", True),
        )
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        early_stopping_final = EarlyStopping(patience=config["patience"])
        lr_scheduler_config = config["learning_rate_scheduler"]
        if lr_scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_config.get("mode", "min"),
                factor=lr_scheduler_config.get("factor", 0.5),
                patience=lr_scheduler_config.get("patience", 5),
                verbose=lr_scheduler_config.get("verbose", True),
            )
        else:
            print(f"Unsupported scheduler type: {lr_scheduler_config['type']}")
            sys.exit(1)

    print("Starting final training on combined training and validation sets...")
    cleanup_interval = max(1, len(train_loader) // 10)
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start_time = time.perf_counter() 
        try:
            train_loss, train_acc, train_timing = train_fold(
                model,
                optimizer,
                criterion,
                scaler_final,
                train_loader,
                device,
                cleanup_interval,
                fold="Final",
                trial_number="Final",
                epoch=epoch,
                total_epochs=config["epochs"],
                gradient_clipping=config["gradient_clipping"],
                accumulation_steps=4,
                use_progress_bar=use_progress_bar,
                show_insights=show_insights,
            )
            val_loss, val_acc, val_timing = validate_fold(
                model, criterion, scaler_final, train_loader, device, scheduler, use_progress_bar=use_progress_bar, show_insights=show_insights
            )
            epoch_end_time = time.perf_counter() 
            epoch_duration_ms = (epoch_end_time - epoch_start_time) * 1000

            if show_insights:
                combined_timing = {
                    "Epoch Time": epoch_duration_ms
                }
                combined_timing.update(train_timing)
                combined_timing.update(val_timing)
            else:
                combined_timing = None

            print_epoch_report(
                "Final",
                "Final",
                epoch,
                config["epochs"],
                config["max_trials"], 
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                show_insights=show_insights,
                timing_info=combined_timing,
            )
            early_stopping_final(val_loss)
            if early_stopping_final.early_stop:
                print(f"\nFinal Training: Early stopping triggered at epoch {epoch}")
                break

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler_final.state_dict(),
                'early_stopping': early_stopping_final,
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_dir, checkpoint_filename)

        except Exception as e:
            print(f"Error during final training: {e}")
 
            del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset, scaler_final, scheduler
            gc.collect()
            torch.cuda.empty_cache()
            break

    print()

    if not early_stopping_final.early_stop:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Checkpoint {checkpoint_path} removed after successful training.")

    model_save_path = os.path.join("./data/models/deep_learning/", "lstm.pt")
    evaluate_and_save_model(
        model,
        X_test,
        y_test,
        label_encoder,
        scaler,
        model_save_path,
        mixed_precision=config["mixed_precision"],
        device=device,
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_features = [
        "displacement_2d",
        "magnitude_min",
        "magnitude_max",
        "rms",
        "rmse",
        "sma",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ]
    main()
