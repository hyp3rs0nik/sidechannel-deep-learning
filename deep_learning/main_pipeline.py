import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import random
import torch.backends.cudnn as cudnn

from sklearn.model_selection import StratifiedKFold, train_test_split
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
NUM_TRIALS = 50
NUM_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 8
MIN_EPOCHS = 20  # Minimum number of epochs before early stopping
NUM_WORKERS = 12
WINDOW_SIZE = 64
OVERFITTING_THRESHOLD = 0.0001  # Small threshold for validation loss improvement
AUGMENTATION_LEVEL = 1  # Set to 1 or higher to enable augmentation
STRIDE = 16  # Adjust stride as needed

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

def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10
    else:
        return -1  # Assign -1 to unexpected keys

def generate_samples(sensor_df, keys_df, window_size=32, stride=32):
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])

    keys_df = keys_df[keys_df["key"] != "Backspace"].copy()
    keys_df['label'] = keys_df['key'].apply(key_to_label)

    sensor_sequences = []
    labels = []

    for start in range(0, len(sensor_df) - window_size + 1, stride):
        end = start + window_size
        sensor_window = sensor_df.iloc[start:end][
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        ].values

        window_start_time = sensor_df.iloc[start]['timestamp']
        window_end_time = sensor_df.iloc[end - 1]['timestamp']

        # Find keys within this window
        keys_in_window = keys_df[(keys_df['timestamp'] >= window_start_time) & (keys_df['timestamp'] <= window_end_time)]

        if not keys_in_window.empty:
            # Assign the label of the last key in the window
            label = keys_in_window.iloc[-1]['label']
            labels.append(label)
        else:
            labels.append(-1)

        sensor_sequences.append(sensor_window)

    sensor_sequences = np.array(sensor_sequences)
    labels = np.array(labels)

    total_sequences = len(sensor_sequences)
    discarded_sequences = np.sum(labels == -1)
    valid_sequences = total_sequences - discarded_sequences

    print(f"Total Sequences: {total_sequences}")
    print(f"Discarded Sequences (no key events): {discarded_sequences}")
    print(f"Valid Sequences (with key events): {valid_sequences}")

    valid_indices = labels != -1
    sensor_sequences = sensor_sequences[valid_indices]
    labels = labels[valid_indices]

    print(f"Total Training Samples: {len(labels)}")

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

def main():
    keys_df = pd.read_csv("./data/training/keys.csv")
    sensor_df = pd.read_csv("./data/training/sensor_v2_denoise_2.25hz.csv")

    print(f'Total keys: {len(keys_df)}')
    print(f'Total sensor data points: {len(sensor_df)}')

    sensor_sequences, labels = generate_samples(sensor_df, keys_df, WINDOW_SIZE, STRIDE)

    # Split the data into training and holdout sets
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        sensor_sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def objective(trial):
        # Adjusted hyperparameter ranges
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        model = get_model(
            args.model, X_train_full.shape[2], hidden_dim,
            11, num_layers, dropout_rate
        ).to(device)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        avg_accuracy = 0

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]

            num_train_samples, window_size, num_features = X_train.shape
            X_train_flat = X_train.reshape(-1, num_features)

            scaler = StandardScaler()
            X_train_flat_scaled = scaler.fit_transform(X_train_flat)
            X_train_scaled = X_train_flat_scaled.reshape(num_train_samples, window_size, num_features)

            num_val_samples = X_val.shape[0]
            X_val_flat = X_val.reshape(-1, num_features)
            X_val_flat_scaled = scaler.transform(X_val_flat)
            X_val_scaled = X_val_flat_scaled.reshape(num_val_samples, window_size, num_features)

            # Data augmentation
            if AUGMENTATION_LEVEL >= 1:
                # Gaussian noise
                noise = np.random.normal(0, 0.01, X_train_scaled.shape)
                X_train_scaled += noise

            if AUGMENTATION_LEVEL >= 2:
                # Time shifting
                shift = np.random.randint(-2, 2)
                X_train_scaled = np.roll(X_train_scaled, shift, axis=1)

            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scaler = GradScaler()

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

                print(f"Trial {trial.number + 1}/{NUM_TRIALS}, Fold {fold + 1}/5, "
                      f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}", end="\r")

                # Early stopping based on validation loss
                if avg_val_loss < best_val_loss - OVERFITTING_THRESHOLD:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                unique_step = epoch + fold * NUM_EPOCHS
                trial.report(avg_val_loss, step=unique_step)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if epoch >= MIN_EPOCHS and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            avg_accuracy += val_accuracy

        avg_accuracy /= 5  # Average over all folds

        return avg_accuracy

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name=f"dl_hyperparameter_tuning_{args.model}",
        storage="sqlite:///data/db.sqlite",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=1)

    best_params = study.best_params
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best validation accuracy: {study.best_value:.4f}")

    # Retrain the model on the full training data with best hyperparameters
    print("Retraining model on full training data with best hyperparameters...")

    # Extract best hyperparameters
    hidden_dim = best_params['hidden_dim']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']

    # Create the model
    model = get_model(
        args.model, X_train_full.shape[2], hidden_dim,
        11, num_layers, dropout_rate
    ).to(device)

    # Scale the training data
    num_train_samples, window_size, num_features = X_train_full.shape
    X_train_flat = X_train_full.reshape(-1, num_features)

    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_flat_scaled.reshape(num_train_samples, window_size, num_features)

    # Transform the holdout data
    num_holdout_samples = X_holdout.shape[0]
    X_holdout_flat = X_holdout.reshape(-1, num_features)
    X_holdout_flat_scaled = scaler.transform(X_holdout_flat)
    X_holdout_scaled = X_holdout_flat_scaled.reshape(num_holdout_samples, window_size, num_features)

    # Data augmentation on the full training data
    if AUGMENTATION_LEVEL >= 1:
        # Gaussian noise
        noise = np.random.normal(0, 0.01, X_train_scaled.shape)
        X_train_scaled += noise

    if AUGMENTATION_LEVEL >= 2:
        # Time shifting
        shift = np.random.randint(-2, 2)
        X_train_scaled = np.roll(X_train_scaled, shift, axis=1)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.long)
    X_holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32)
    y_holdout_tensor = torch.tensor(y_holdout, dtype=torch.long)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    holdout_dataset = TensorDataset(X_holdout_tensor, y_holdout_tensor)
    holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    # Compute class weights
    classes = np.unique(y_train_full)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_full)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    # Train the model
    model.train()
    epochs_no_improve = 0
    best_val_loss = float('inf')

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

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}", end="\r")

        # Optionally implement early stopping in retraining
        # If you have a validation set, you can compute validation loss here and apply early stopping

    print("\nTraining complete.")

    # Evaluate on the holdout set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in holdout_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Print classification report
    from sklearn.metrics import classification_report

    print("\nClassification Report on Holdout Set:")
    print(classification_report(all_targets, all_preds, digits=4))

    # Save the model
    torch.save(model.state_dict(), f"./data/models/{args.model}_best_model.pth")
    print(f"\nModel saved to ./data/models/{args.model}_best_model.pth")

if __name__ == "__main__":
    main()
