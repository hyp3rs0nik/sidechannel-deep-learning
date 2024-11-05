import os
import sys
import json
import argparse
import logging
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import optuna
from torch.cuda.amp import GradScaler, autocast
from scipy.fft import fft

torch.backends.cudnn.enabled = True

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_PATH = './data/models/deep_learning'
os.makedirs(MODEL_PATH, exist_ok=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def load_data():
    keys_df = pd.read_csv('./data/training/keys.csv')
    sensors_df = pd.read_csv('./data/training/sensors.csv')
    return keys_df, sensors_df

# Enhanced alignment logic
def align_keys_to_sensors(keys_df, sensors_df, window_ms=550):
    keys_df['timestamp'] = keys_df['timestamp'].astype(int)
    sensors_df['timestamp'] = sensors_df['timestamp'].astype(int)

    keys_df['time_diff'] = keys_df['timestamp'].diff().fillna(0)
    aligned_df = pd.merge_asof(sensors_df, keys_df, on='timestamp', direction='nearest', tolerance=window_ms)
    aligned_df.dropna(subset=['key'], inplace=True)
    logger.info(f"Aligned Data Shape: {aligned_df.shape}")
    return aligned_df

# Comprehensive feature extraction
def extract_features(aligned_data, window_size=25, step_size=12):
    sequences = []
    labels = []

    for key, group in aligned_data.groupby('key'):
        for i in range(0, len(group) - window_size + 1, step_size):
            window_data = group.iloc[i:i + window_size]

            # Temporal features
            time_diff = window_data['timestamp'].diff().fillna(0).values
            accel_diff_x = window_data['accel_x'].diff().fillna(0).values
            accel_diff_y = window_data['accel_y'].diff().fillna(0).values
            accel_diff_z = window_data['accel_z'].diff().fillna(0).values
            gyro_diff_x = window_data['gyro_x'].diff().fillna(0).values
            gyro_diff_y = window_data['gyro_y'].diff().fillna(0).values
            gyro_diff_z = window_data['gyro_z'].diff().fillna(0).values
            rot_diff_x = window_data['rotation_x'].diff().fillna(0).values
            rot_diff_y = window_data['rotation_y'].diff().fillna(0).values
            rot_diff_z = window_data['rotation_z'].diff().fillna(0).values
            rot_diff_w = window_data['rotation_w'].diff().fillna(0).values

            # Frequency-domain features (FFT)
            fft_accel_x = fft(window_data['accel_x'])
            fft_accel_y = fft(window_data['accel_y'])
            fft_accel_z = fft(window_data['accel_z'])
            fft_gyro_x = fft(window_data['gyro_x'])
            fft_gyro_y = fft(window_data['gyro_y'])
            fft_gyro_z = fft(window_data['gyro_z'])
            fft_rotation_x = fft(window_data['rotation_x'])
            fft_rotation_y = fft(window_data['rotation_y'])
            fft_rotation_z = fft(window_data['rotation_z'])
            fft_rotation_w = fft(window_data['rotation_w'])

            # Statistical and frequency-based features
            features = {
                # Accelerometer statistics
                'rms_accel': np.sqrt(np.mean(window_data[['accel_x', 'accel_y', 'accel_z']] ** 2)),
                'mean_accel_x': window_data['accel_x'].mean(),
                'mean_accel_y': window_data['accel_y'].mean(),
                'mean_accel_z': window_data['accel_z'].mean(),
                'std_accel_x': window_data['accel_x'].std(),
                'std_accel_y': window_data['accel_y'].std(),
                'std_accel_z': window_data['accel_z'].std(),
                
                # Gyroscope statistics
                'rms_gyro': np.sqrt(np.mean(window_data[['gyro_x', 'gyro_y', 'gyro_z']] ** 2)),
                'mean_gyro_x': window_data['gyro_x'].mean(),
                'mean_gyro_y': window_data['gyro_y'].mean(),
                'mean_gyro_z': window_data['gyro_z'].mean(),
                'std_gyro_x': window_data['gyro_x'].std(),
                'std_gyro_y': window_data['gyro_y'].std(),
                'std_gyro_z': window_data['gyro_z'].std(),

                # Rotation data statistics (fusion sensor)
                'rms_rotation': np.sqrt(np.mean(window_data[['rotation_x', 'rotation_y', 'rotation_z', 'rotation_w']] ** 2)),
                'mean_rotation_x': window_data['rotation_x'].mean(),
                'mean_rotation_y': window_data['rotation_y'].mean(),
                'mean_rotation_z': window_data['rotation_z'].mean(),
                'mean_rotation_w': window_data['rotation_w'].mean(),
                'std_rotation_x': window_data['rotation_x'].std(),
                'std_rotation_y': window_data['rotation_y'].std(),
                'std_rotation_z': window_data['rotation_z'].std(),
                'std_rotation_w': window_data['rotation_w'].std(),

                # FFT mean values for frequency analysis
                'fft_mean_accel_x': np.mean(np.abs(fft_accel_x)),
                'fft_mean_accel_y': np.mean(np.abs(fft_accel_y)),
                'fft_mean_accel_z': np.mean(np.abs(fft_accel_z)),
                'fft_mean_gyro_x': np.mean(np.abs(fft_gyro_x)),
                'fft_mean_gyro_y': np.mean(np.abs(fft_gyro_y)),
                'fft_mean_gyro_z': np.mean(np.abs(fft_gyro_z)),
                'fft_mean_rotation_x': np.mean(np.abs(fft_rotation_x)),
                'fft_mean_rotation_y': np.mean(np.abs(fft_rotation_y)),
                'fft_mean_rotation_z': np.mean(np.abs(fft_rotation_z)),
                'fft_mean_rotation_w': np.mean(np.abs(fft_rotation_w)),

                # Temporal statistics
                'time_diff_mean': np.mean(time_diff),
                'accel_diff_mean_x': np.mean(accel_diff_x),
                'accel_diff_mean_y': np.mean(accel_diff_y),
                'accel_diff_mean_z': np.mean(accel_diff_z),
                'gyro_diff_mean_x': np.mean(gyro_diff_x),
                'gyro_diff_mean_y': np.mean(gyro_diff_y),
                'gyro_diff_mean_z': np.mean(gyro_diff_z),
                'rot_diff_mean_x': np.mean(rot_diff_x),
                'rot_diff_mean_y': np.mean(rot_diff_y),
                'rot_diff_mean_z': np.mean(rot_diff_z),
                'rot_diff_mean_w': np.mean(rot_diff_w),
            }

            sequences.append(features)
            labels.append(key)

    features_df = pd.DataFrame(sequences)
    features_df['key'] = labels
    logger.info(f"Shape of features after extraction: {features_df.shape}")
    return features_df



def preprocess_data():
    keys_df, sensors_df = load_data()
    logger.info('Aligning keys to sensors...')
    aligned_data = align_keys_to_sensors(keys_df, sensors_df)
    logger.info(f'Number of samples after alignment: {len(aligned_data)}')

    logger.info('Extracting features...')
    features_df = extract_features(aligned_data)

    label_distribution = features_df['key'].value_counts()
    logger.info("Label distribution:\n" + str(label_distribution))

    X = features_df.drop(columns=['key']).values
    y = features_df['key'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, label_encoder

# BiLSTM with attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, lstm_out):
        weights = torch.tanh(self.attention_weights(lstm_out)).squeeze(-1)
        attention_weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        weighted_output = (lstm_out * attention_weights).sum(dim=1)
        return weighted_output

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(BiLSTMWithAttention, self).__init__()
        self.num_layers = num_layers
        # Apply dropout only if num_layers > 1
        dropout_value = dropout_prob if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_value)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: (batch_size, seq_length, hidden_size * 2)

        # Compute attention scores
        attention_scores = torch.tanh(self.attention(lstm_out))  # Shape: (batch_size, seq_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_length, 1)

        # Apply attention weights to LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: (batch_size, hidden_size * 2)

        # Apply dropout and fully connected layer
        out = self.dropout(context_vector)
        out = self.fc(out)  # Shape: (batch_size, output_size)

        return out


def train_lstm(n_trials=50, n_jobs=1, num_workers=4):
    X_scaled, y_encoded, scaler, label_encoder = preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 128, 512)
        num_layers = trial.suggest_int('num_layers', 1, 4) 
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_prob = trial.suggest_float('dropout_prob', 0.2, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])


        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        model = BiLSTMWithAttention(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=len(label_encoder.classes_), dropout_prob=dropout_prob).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = GradScaler()

        num_epochs = 100
        patience = 7
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            model.eval()
            val_loss, all_preds, all_labels = 0.0, [], []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered due to no improvement in validation loss.")
                break

        return f1

    # Optimize the hyperparameters using Optuna
    study = optuna.create_study(
        study_name="lstm_hyperparameter_tuning",
        storage='sqlite:///optuna_lstm_study.db',
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    # Log and save the best hyperparameters
    if study.best_trial:
        best_params = study.best_trial.params
        logger.info(f"Best trial value: {study.best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
    else:
        logger.error("No successful trials were completed.")
        sys.exit(1)

    # Train the final model with the best parameters
    model = BiLSTMWithAttention(
        input_size=X_scaled.shape[1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        output_size=len(label_encoder.classes_),
        dropout_prob=best_params['dropout_prob']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scaler = GradScaler()
    batch_size = best_params['batch_size']

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Final Training Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered during final training.")
            break

    # Save the model and artifacts
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "bilstm_attention_model.pth"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))
    with open(os.path.join(MODEL_PATH, "bilstm_attention_best_params.json"), "w") as f:
        json.dump(best_params, f)
    logger.info("Model, scaler, label encoder, and best parameters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM model with attention for typing pattern recognition.")
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs for Optuna.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of subprocesses for data loading.')
    args = parser.parse_args()

    train_lstm(n_trials=args.n_trials, n_jobs=args.n_jobs, num_workers=args.num_workers)

