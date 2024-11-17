import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import random
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset

from behaveformer_model import BehaviorNetModel
from gru_attn_model import AttentionGRUModel
from lstm_attn_model import AttentionLSTMModel


optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("./data/models", exist_ok=True)
os.makedirs("./docs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TRIALS = 50
NUM_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
MIN_EPOCHS = 20
NUM_WORKERS = 4  # Adjust based on your system
WINDOW_SIZE = 64
OVERFITTING_THRESHOLD = 0.0001
AUGMENTATION_LEVEL = 1
STRIDE = 16

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

seed_everything()

parser = argparse.ArgumentParser(description="Select the model architecture.")
parser.add_argument(
    "--model",
    type=str,
    default="gru",
    choices=["gru", "lstm", "behavenet"],
    help="Model type to use: gru, lstm, behavenet",
)
args = parser.parse_args()

class EpochTimer:
    def __init__(self):
        self.epoch_times = []

    def start_epoch(self):
        self.start_time = time.time()

    def end_epoch(self):
        if not hasattr(self, 'start_time'):
            raise RuntimeError("You must call start_epoch() before calling end_epoch().")
        end_time = time.time()
        duration = end_time - self.start_time
        self.epoch_times.append(duration)
        return duration

    def get_average_time(self):
        if not self.epoch_times:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)

def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10
    else:
        return -1

def generate_samples(sensor_df, keys_df, window_size=32, stride=32):
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])

    keys_df = keys_df[keys_df["key"] != "Backspace"].copy()
    keys_df['label'] = keys_df['key'].apply(key_to_label)

    sensor_sequences = []
    labels = []

    sensor_df.set_index('timestamp', inplace=True)
    keys_df.set_index('timestamp', inplace=True)
    sensor_df.sort_index(inplace=True)
    sensor_df.reset_index(inplace=True)
    keys_df.reset_index(inplace=True)

    for start in range(0, len(sensor_df) - window_size + 1, stride):
        end = start + window_size
        sensor_window = sensor_df.iloc[start:end][
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        ].values

        window_start_time = sensor_df.iloc[start]['timestamp']
        window_end_time = sensor_df.iloc[end - 1]['timestamp']

        keys_in_window = keys_df[(keys_df['timestamp'] >= window_start_time) & (keys_df['timestamp'] <= window_end_time)]

        if not keys_in_window.empty:
            label = keys_in_window.iloc[-1]['label']
            labels.append(label)
            sensor_sequences.append(sensor_window)

    sensor_sequences = np.array(sensor_sequences)
    labels = np.array(labels)

    return sensor_sequences, labels

def get_model(model_type, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
    if model_type == "gru":
        return AttentionGRUModel(
            input_dim, hidden_dim, output_dim, num_layers, dropout_rate
        )
    elif model_type == "lstm":
        return AttentionLSTMModel(
            input_dim, hidden_dim, output_dim, num_layers, dropout_rate
        )
    elif model_type == "behavenet":
        return BehaviorNetModel(
            input_dim, hidden_dim, output_dim, num_layers, dropout_rate
        )
    else:
        raise ValueError("Invalid model type selected.")

class SensorDataset(Dataset):
    def __init__(self, sequences, labels, augment=False, noise_intensity=0.01):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.noise_intensity = noise_intensity

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        sequence = torch.tensor(sequence, dtype=torch.float32)
        if self.augment:
            noise = torch.randn_like(sequence) * self.noise_intensity
            scale_factor = torch.empty(1).uniform_(0.9, 1.1)
            sequence = sequence * scale_factor + noise
        label = torch.tensor(label, dtype=torch.long)
        return sequence, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()
    early_stopping_epoch = num_epochs
    last_train_accuracy = 0.0
    last_train_loss = 0.0
    last_val_accuracy = 0.0
    last_val_loss = 0.0

    timer = EpochTimer()

    for epoch in range(num_epochs):
        timer.start_epoch()
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
        last_train_accuracy = total_correct / total_samples
        last_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_samples += batch_y.size(0)
        last_val_accuracy = val_correct / val_samples
        last_val_loss = val_loss / len(val_loader)

        if last_val_loss < best_val_loss - OVERFITTING_THRESHOLD:
            best_val_loss = last_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
            early_stopping_epoch = epoch + 1
        else:
            epochs_no_improve += 1

        avg_epoch_train_time = timer.end_epoch()

        if epoch >= MIN_EPOCHS and epochs_no_improve >= patience:
            model.load_state_dict(best_model_wts)
            break

    average_time = timer.get_average_time()
    return model, early_stopping_epoch, last_train_accuracy, last_train_loss, last_val_accuracy, last_val_loss, average_time

def color_accuracy(acc):
    if isinstance(acc, str):
        return acc
    if acc < 0.40:
        color = '\033[91m'  # Red
    elif acc < 0.55:
        color = '\033[33m'  # Orange (using ANSI code for yellow)
    elif acc < 0.65:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[92m'  # Green
    return f"{color}{acc:.3f}\033[0m"

def log_status(best_accuracy, trial_number, total_trials, fold_number, total_folds,
               train_accuracy, train_loss, val_loss, val_accuracy, avg_epoch_train_time,
               early_stopping_info=None):
    line_length = 80

    # Remove console clearing to reduce overhead
    os.system('cls' if os.name == 'nt' else 'clear')

    best_accuracy_str = f"{color_accuracy(best_accuracy)}" if isinstance(best_accuracy, float) else best_accuracy

    train_acc_str = color_accuracy(train_accuracy)
    val_acc_str = color_accuracy(val_accuracy)

    print('=' * line_length)
    print(f"Best Accuracy: {best_accuracy_str}, Avg Epoch Time: {avg_epoch_train_time:.3f}s")
    print('=' * line_length)
    print(f"Trial {trial_number}/{total_trials}, Fold {fold_number}/{total_folds}")
    print(f"Train Acc: {train_acc_str}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc_str}")
    print('=' * line_length)
    if early_stopping_info:
        print(f"Early Stopping Info: {early_stopping_info}")
        print('=' * line_length)

def main():
    keys_df = pd.read_csv("./data/training/clean_v3.keystrokes.csv")
    sensor_df = pd.read_csv("./data/training/clean_v3.sensors.csv")

    sensor_sequences, labels = generate_samples(
        sensor_df, keys_df, WINDOW_SIZE, STRIDE
    )

    num_features = sensor_sequences.shape[2]
    model_input = sensor_sequences

    # Precompute normalization
    model_input_flat = model_input.reshape(-1, num_features)
    mean = model_input_flat.mean(axis=0)
    std = model_input_flat.std(axis=0)
    model_input_normalized = (model_input - mean) / std

    total_trials = NUM_TRIALS

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        noise_intensity = trial.suggest_float("noise_intensity", 0.005, 0.02)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        avg_accuracy = 0
        total_folds = kf.get_n_splits()

        try:
            best_accuracy_overall = trial.study.best_value
        except ValueError:
            best_accuracy_overall = 'N/A'

        for fold, (train_idx, val_idx) in enumerate(kf.split(model_input_normalized), 1):
            model = get_model(
                args.model, num_features, hidden_dim,
                11, num_layers, dropout_rate
            ).to(device)

            X_train = model_input_normalized[train_idx]
            y_train = labels[train_idx]
            X_val = model_input_normalized[val_idx]
            y_val = labels[val_idx]

            train_dataset = SensorDataset(X_train, y_train, augment=False)
            val_dataset = SensorDataset(X_val, y_val, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)

            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            model, early_stopping_epoch, last_train_accuracy, last_train_loss, last_val_accuracy, last_val_loss, avg_epoch_train_time = train_model(
                model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
            )

            avg_accuracy += last_val_accuracy

            early_stopping_info = f"Trial {trial.number+1}, Epoch {early_stopping_epoch} at {last_val_accuracy:.3f}"

            # Call log_status after each fold
            log_status(best_accuracy_overall, trial.number+1, total_trials, fold, total_folds,
                       last_train_accuracy, last_train_loss, last_val_loss, last_val_accuracy,
                       avg_epoch_train_time, early_stopping_info=early_stopping_info)

        avg_accuracy /= total_folds
        return avg_accuracy

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best validation accuracy: {study.best_value:.4f}")

    hidden_dim = best_params['hidden_dim']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    noise_intensity = best_params['noise_intensity']

    model = get_model(
        args.model, num_features, hidden_dim,
        11, num_layers, dropout_rate
    ).to(device)

    # Use the normalized data
    full_dataset = SensorDataset(model_input_normalized, labels, augment=True, noise_intensity=noise_intensity)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model, _, _, _, _, _, _ = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)

    unseen_keys_df = pd.read_csv("./data/unseen/keys.csv")
    unseen_sensor_df = pd.read_csv("./data/unseen/sensors.csv")

    unseen_sequences, unseen_labels = generate_samples(
        unseen_sensor_df, unseen_keys_df, WINDOW_SIZE, STRIDE
    )

    # Normalize the unseen data using training mean and std
    unseen_sequences_normalized = (unseen_sequences - mean) / std

    unseen_dataset = SensorDataset(unseen_sequences_normalized, unseen_labels, augment=False)
    unseen_loader = DataLoader(unseen_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in unseen_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    print("\nClassification Report on Unseen Set:")
    print(classification_report(all_targets, all_preds, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix on Unseen Set")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig('./docs/confusion_matrix.png')
    plt.close()

    torch.save(model.state_dict(), f"./data/models/{args.model}_best_model.pth")
    print(f"\nModel saved to ./data/models/{args.model}_best_model.pth")

if __name__ == "__main__":
    main()
