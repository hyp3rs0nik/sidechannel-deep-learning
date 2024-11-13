import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import random
import copy
import torch.backends.cudnn as cudnn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from behaveformer_model import BehaviorNetModel
from gru_attn_model import AttentionGRUModel
from lstm_attn_model import AttentionLSTMModel

# Create the models directory if it doesn't exist
os.makedirs("./data/models", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters and constants
NUM_TRIALS = 100
NUM_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 6
WINDOW_SIZE = 8
OVERFITTING_THRESHOLD = 0.0001  # Small threshold for validation loss improvement
AUGMENTATION_LEVEL = 0  # Set to 1 to enable augmentation
STRIDE = 1  # Decrease stride to increase the number of samples

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

seed_everything()

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Select the model architecture.")
parser.add_argument(
    "--model",
    type=str,
    default="gru",
    choices=["gru", "lstm", "behavenet"],
    help="Model type to use: gru, lstm, or behavenet",
)
args = parser.parse_args()

# Define key_to_label function early
def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10
    else:
        raise ValueError(f"Unexpected key: {key}")

def generate_samples(sensor_df, keys_df, window_size, stride):
    # Ensure 'timestamp' columns exist
    if 'timestamp' not in sensor_df.columns or 'timestamp' not in keys_df.columns:
        raise ValueError("Both sensor_df and keys_df must have a 'timestamp' column.")
    
    # Filter 'Backspace' and prepare labels
    keys_df = keys_df[keys_df["key"] != "Backspace"].copy()
    keys_df.loc[:, 'label'] = keys_df['key'].apply(key_to_label)
    
    # Precompute magnitudes
    sensor_df["accel_magnitude"] = np.linalg.norm(sensor_df[["accel_x", "accel_y", "accel_z"]], axis=1)
    sensor_df["gyro_magnitude"] = np.linalg.norm(sensor_df[["gyro_x", "gyro_y", "gyro_z"]], axis=1)

    # Prepare variables
    sensor_sequences, labels = [], []
    sensor_timestamps = sensor_df['timestamp'].values
    key_timestamps = keys_df['timestamp'].values
    key_labels = keys_df['label'].values

    print("Generating sensor sequences and labels...")

    # Map each sensor window to a label based on the closest timestamped key press
    key_indices = np.searchsorted(sensor_timestamps, key_timestamps, side="right") - 1

    # Iterate with windowed sampling and label assignment
    for i in range(0, len(sensor_df) - window_size + 1, stride):
        # Extract the sensor window
        sensor_window = sensor_df.iloc[i:i + window_size][
            ["accel_x", "accel_y", "accel_z",
             "gyro_x", "gyro_y", "gyro_z",
             "accel_magnitude", "gyro_magnitude"]
        ].values
        sensor_sequences.append(sensor_window)
        
        # Find relevant keys in the window
        window_start_time = sensor_timestamps[i]
        window_end_time = sensor_timestamps[i + window_size - 1]
        keys_in_window = key_labels[(key_timestamps >= window_start_time) & (key_timestamps <= window_end_time)]
        
        # Assign the label for the last key press in the window or -1 if none
        label = keys_in_window[-1] if keys_in_window.size > 0 else -1
        labels.append(label)

    # Convert to arrays and filter invalid labels
    sensor_sequences = np.array(sensor_sequences)
    labels = np.array(labels)
    valid_indices = labels != -1
    sensor_sequences = sensor_sequences[valid_indices]
    labels = labels[valid_indices]

    print(f"Total generated sensor sequences: {sensor_sequences.shape[0]}")

    return sensor_sequences, labels

# Load data
keys_df = pd.read_csv("./data/training/keys.csv")
sensor_df = pd.read_csv("./data/training/sensor_v2_denoise_2.25hz.csv")

# Generate samples
sensor_sequences, labels = generate_samples(sensor_df, keys_df, WINDOW_SIZE, STRIDE)

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

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = get_model(
        args.model, sensor_sequences.shape[2], hidden_dim,
        11, num_layers, dropout_rate
    ).to(device)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    avg_accuracy = 0
    best_accuracy = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(sensor_sequences, labels)):
        # Extract training and validation data
        X_train = sensor_sequences[train_idx]
        y_train = labels[train_idx]
        X_val = sensor_sequences[val_idx]
        y_val = labels[val_idx]

        # Reshape for scaling
        num_train_samples, window_size, num_features = X_train.shape
        X_train_flat = X_train.reshape(-1, num_features)

        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_flat_scaled = scaler.fit_transform(X_train_flat)

        # Transform training data
        X_train_scaled = X_train_flat_scaled.reshape(num_train_samples, window_size, num_features)

        # Transform validation data
        num_val_samples = X_val.shape[0]
        X_val_flat = X_val.reshape(-1, num_features)
        X_val_flat_scaled = scaler.transform(X_val_flat)
        X_val_scaled = X_val_flat_scaled.reshape(num_val_samples, window_size, num_features)

        # Apply data augmentation to training data
        if AUGMENTATION_LEVEL == 1:
            noise = np.random.normal(0, 0.01, X_train_scaled.shape)
            X_train_scaled += noise

        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = GradScaler()

        model.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_correct = 0
            total_samples = 0
            total_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    total_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

            train_accuracy = total_correct / total_samples
            avg_train_loss = total_loss / len(train_loader)

            # Validation phase
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_outputs = model(batch_X)
                    loss = criterion(val_outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(val_outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)

            val_accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)

            print(f"Trial {trial.number + 1}/{NUM_TRIALS}, Fold {fold + 1}/{kfold.get_n_splits()}, "
                  f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}", end="\r")

            # Early stopping based on validation loss
            if avg_val_loss < best_val_loss - OVERFITTING_THRESHOLD:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            unique_step = epoch + fold * NUM_EPOCHS
            trial.report(avg_val_loss, step=unique_step)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\033[93m\nEarly stopping at epoch {epoch + 1}\033[0m")
                break

        avg_accuracy += val_accuracy

    avg_accuracy /= 5  # Average over all folds

    # Save the best model if it's the best so far
    if trial.number == 0 or avg_accuracy > trial.study.best_value:
        torch.save(best_model_state, f"./data/models/{args.model}_best_model.pth")
        print(f"\nNew best model saved for trial {trial.number} with average validation accuracy: {avg_accuracy:.4f}")

    return avg_accuracy

def main():
    study = optuna.create_study(
        direction="maximize",
        study_name=f"dl_hyperparameter_tuning_{args.model}",
        storage="sqlite:///data/db.sqlite",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=2)

if __name__ == "__main__":
    main()
