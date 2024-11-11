import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from behaveformer_model import BehaviorNetModel
from gru_attn_model import AttentionGRUModel
from lstm_attn_model import AttentionLSTMModel


os.makedirs("./data/models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_TRIALS = 128
NUM_EPOCHS = 128
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 4
WINDOW_SIZE = 8
OVERFITTING_THRESHOLD = 0.04
AUGMENTATION_LEVEL = 1


parser = argparse.ArgumentParser(description="Select the model architecture.")
parser.add_argument(
    "--model",
    type=str,
    default="gru",
    choices=["gru", "lstm", "behavenet"],
    help="Model type to use: gru, lstm, or behavenet",
)
args = parser.parse_args()


keys_df = pd.read_csv("./data/training/keys.csv")
sensor_df = pd.read_csv("./data/training/sensor_v2_denoise_2.25hz.csv")


scaler = StandardScaler()
sensor_df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]] = (
    scaler.fit_transform(
        sensor_df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]]
    )
)

def augment_sensor_data(sensor_data):
    if AUGMENTATION_LEVEL == 1:
        noise = np.random.normal(0, 0.01, sensor_data.shape)
        augmented_data = sensor_data + noise
        return augmented_data
    return sensor_data


sensor_df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]] = (
    augment_sensor_data(
        sensor_df[
            ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        ].values
    )
)

key_coordinates = {
    "1": (0, 0),
    "2": (0, 1),
    "3": (0, 2),
    "4": (1, 0),
    "5": (1, 1),
    "6": (1, 2),
    "7": (2, 0),
    "8": (2, 1),
    "9": (2, 2),
    "0": (2.5, 1),
    "Enter": (2.5, 3),
}

def compute_displacement_vector(key1, key2):
    if key1 in key_coordinates and key2 in key_coordinates:
        x1, y1 = key_coordinates[key1]
        x2, y2 = key_coordinates[key2]
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        return None, None


keys_df = keys_df[keys_df["key"] != "Backspace"]
displacement_vectors = []
previous_key = None

for key in keys_df["key"]:
    if previous_key is not None:
        dx, dy = compute_displacement_vector(previous_key, key)
        if dx is not None and dy is not None:
            displacement_vectors.append((dx, dy))
    previous_key = key


sensor_df["accel_magnitude"] = np.sqrt(
    sensor_df["accel_x"] ** 2 + sensor_df["accel_y"] ** 2 + sensor_df["accel_z"] ** 2
)
sensor_df["gyro_magnitude"] = np.sqrt(
    sensor_df["gyro_x"] ** 2 + sensor_df["gyro_y"] ** 2 + sensor_df["gyro_z"] ** 2
)


window_size = WINDOW_SIZE
sensor_sequences = []

for i in range(len(displacement_vectors)):
    start_idx = i * window_size
    end_idx = start_idx + window_size
    if end_idx <= len(sensor_df):
        sensor_window = sensor_df.iloc[start_idx:end_idx][
            [
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_magnitude",
                "gyro_magnitude",
            ]
        ].values
        sensor_sequences.append(sensor_window)

sensor_sequences = np.array(sensor_sequences)
displacement_features = np.repeat(
    np.array(displacement_vectors).reshape(-1, 1, 2), window_size, axis=1
)
model_input = np.concatenate((sensor_sequences, displacement_features), axis=2)

X = torch.tensor(model_input, dtype=torch.float32)


def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10
    else:
        raise ValueError(f"Unexpected key: {key}")


y = torch.tensor(
    [key_to_label(key) for key in keys_df["key"][: len(displacement_vectors)]],
    dtype=torch.long,
)

dataset = TensorDataset(X, y)

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

def objective(trial, study):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = get_model(
        args.model, model_input.shape[2], hidden_dim,
        11, num_layers, dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    avg_accuracy = 0
    best_accuracy = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model.train()
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

            print(f"Trial {trial.number + 1}/{NUM_TRIALS}, Fold {fold + 1}/{kfold.get_n_splits(X, y)}, "
                  f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.2f}, Train Acc: {train_accuracy:.2f}, "
                  f"Val Loss: {avg_val_loss:.2f}, Val Acc: {val_accuracy:.2f}", end="\r")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\033[93m\nEarly stopping at epoch {epoch + 1}\033[0m")
                break

        avg_accuracy += best_accuracy

    avg_accuracy /= 5

    try:
        if not study.best_trial or avg_accuracy > study.best_value:
            torch.save(best_model_state, f"./data/models/{args.model}_best_model.pth")
            print(f"New global best model saved for trial {trial.number} with validation accuracy: {avg_accuracy:.2f}")
    except ValueError:
        print(f"No completed trials yet. Saving the model for trial {trial.number} with accuracy {avg_accuracy:.2f}")
        torch.save(best_model_state, f"./data/models/{args.model}_best_model.pth")

    return avg_accuracy

def main():
    study = optuna.create_study(direction="maximize", study_name=f"dl_hyperparameter_tuning_{args.model}", storage="sqlite:///data/db.sqlite", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, study), n_trials=NUM_TRIALS)

main()

