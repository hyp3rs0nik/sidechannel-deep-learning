import os
import argparse
import optuna
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


# Constants
TRAINING_DATA_PATH = './data/training'
TEST_DATA_PATH = './data/test'
MODEL_SAVE_PATH = './data/models'
SENSOR_COLUMNS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
WINDOW_SIZE = 50
STEP_SIZE = 10
FS = 50  # Sampling rate
CUTOFF = 17  # Cutoff frequency
PATIENCE = 10  # Early stopping patience


# Functions
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def load_data(data_path, scaler=None, label_mapping=None, is_training=True):
    keystrokes_path = os.path.join(data_path, 'v3.keystrokes.csv')
    sensors_path = os.path.join(data_path, 'v3.sensors.csv')
    keystrokes_df = pd.read_csv(keystrokes_path)
    sensors_df = pd.read_csv(sensors_path)

    sensors_df[SENSOR_COLUMNS] = sensors_df[SENSOR_COLUMNS].apply(
        lambda col: butter_lowpass_filter(col, CUTOFF, FS)
    )

    if is_training:
        scaler = StandardScaler()
        sensors_df[SENSOR_COLUMNS] = scaler.fit_transform(sensors_df[SENSOR_COLUMNS])
    else:
        sensors_df[SENSOR_COLUMNS] = scaler.transform(sensors_df[SENSOR_COLUMNS])

    keystrokes_df = keystrokes_df.sort_values('timestamp').reset_index(drop=True)
    sensors_df = sensors_df.sort_values('timestamp').reset_index(drop=True)

    if is_training:
        unique_keys = keystrokes_df['key'].unique()
        label_mapping = {key: idx for idx, key in enumerate(unique_keys)}
    keystrokes_df['label'] = keystrokes_df['key'].map(label_mapping)

    if keystrokes_df['label'].isnull().any():
        missing_keys = keystrokes_df[keystrokes_df['label'].isnull()]['key'].unique()
        raise ValueError(f"Found keys not present in label mapping: {missing_keys}")

    sensors_df['label'] = np.nan
    keystroke_idx = 0
    last_label = np.nan
    for i in range(len(sensors_df)):
        sensor_time = sensors_df.at[i, 'timestamp']
        while (keystroke_idx < len(keystrokes_df) and
               keystrokes_df.at[keystroke_idx, 'timestamp'] <= sensor_time):
            last_label = keystrokes_df.at[keystroke_idx, 'label']
            keystroke_idx += 1
        sensors_df.at[i, 'label'] = last_label

    sensors_df = sensors_df.dropna(subset=['label']).reset_index(drop=True)
    sensors_df['label'] = sensors_df['label'].astype(int)

    X = []
    y = []
    for start_idx in range(0, len(sensors_df) - WINDOW_SIZE + 1, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        sequence = sensors_df.iloc[start_idx:end_idx]
        label = sequence['label'].values[-1]
        sequence_data = sequence[SENSOR_COLUMNS].values.T
        X.append(sequence_data)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, label_mapping

def define_model(num_classes, best_params):
    activation_name = best_params['activation']
    activation = getattr(nn, activation_name)()
    dropout_rate = best_params['dropout_rate']
    lstm_hidden_size = best_params['lstm_hidden_size']

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(6, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                activation,
                nn.MaxPool1d(2),
                nn.Dropout(p=dropout_rate),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                activation,
                nn.MaxPool1d(2),
                nn.Dropout(p=dropout_rate)
            )
            self.lstm = nn.LSTM(32, lstm_hidden_size, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_size, 32),
                activation,
                nn.Dropout(p=dropout_rate),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = self.fc(x)
            return x

    return Model()

def train_and_validate(X, y, args, device):
    num_classes = len(np.unique(y))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 32, 128)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])

        activation = getattr(nn, activation_name)()

        class HybridModel(nn.Module):
            def __init__(self):
                super(HybridModel, self).__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(6, 16, kernel_size=3, padding=1),
                    nn.BatchNorm1d(16),
                    activation,
                    nn.MaxPool1d(2),
                    nn.Dropout(p=dropout_rate),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    activation,
                    nn.MaxPool1d(2),
                    nn.Dropout(p=dropout_rate)
                )
                self.lstm = nn.LSTM(32, lstm_hidden_size, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(lstm_hidden_size, 32),
                    activation,
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                x = self.cnn(x)
                x = x.permute(0, 2, 1)
                x, _ = self.lstm(x)
                x = x[:, -1, :]
                x = self.fc(x)
                return x

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        cv_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor)):
            X_train_fold = X_tensor[train_idx]
            y_train_fold = y_tensor[train_idx]
            X_val_fold = X_tensor[val_idx]
            y_val_fold = y_tensor[val_idx]

            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            val_dataset = TensorDataset(X_val_fold, y_val_fold)

            class_counts = np.bincount(y_train_fold.numpy())
            class_weights = 1. / class_counts
            sample_weights = class_weights[y_train_fold.numpy()]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            model = HybridModel().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_accuracy = 0
            epochs_no_improve = 0

            for epoch in range(args.epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X = val_X.to(device)
                        val_y = val_y.to(device)
                        val_outputs = model(val_X)
                        _, val_predicted = torch.max(val_outputs, 1)
                        val_total += val_y.size(0)
                        val_correct += (val_predicted == val_y).sum().item()

                val_accuracy = val_correct / val_total

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:
                    break

                trial.report(val_accuracy, epoch + fold * args.epochs)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            cv_accuracies.append(best_val_accuracy)

        mean_accuracy = np.mean(cv_accuracies)
        return mean_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.num_trials)

    best_trial = study.best_trial
    best_params = best_trial.params

    # Save best_params
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_PATH, 'best_params.pkl'), 'wb') as f:
        pickle.dump(best_params, f)

    return best_params

def test_model(best_params, scaler, label_mapping, device):
    X_test, y_test, _, _ = load_data(TEST_DATA_PATH, scaler=scaler, label_mapping=label_mapping, is_training=False)
    num_classes = len(label_mapping)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    model = define_model(num_classes, best_params).to(device)
    model_path = os.path.join(MODEL_SAVE_PATH, 'final_hybrid_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [str(inverse_label_mapping[i]) for i in range(num_classes)]
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

def main():
    parser = argparse.ArgumentParser(description='Hybrid Model Training and Testing')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    if args.test:
        # Load scaler, label mapping, and best_params
        with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_SAVE_PATH, 'label_mapping.pkl'), 'rb') as f:
            label_mapping = pickle.load(f)
        with open(os.path.join(MODEL_SAVE_PATH, 'best_params.pkl'), 'rb') as f:
            best_params = pickle.load(f)

        test_model(best_params, scaler, label_mapping, device)
    else:
        X, y, scaler, label_mapping = load_data(TRAINING_DATA_PATH)
        best_params = train_and_validate(X, y, args, device)

        # Save scaler and label mapping
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODEL_SAVE_PATH, 'label_mapping.pkl'), 'wb') as f:
            pickle.dump(label_mapping, f)

        # Train final model on training data
        num_classes = len(np.unique(y))
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        class_counts = np.bincount(y_tensor.numpy())
        class_weights = 1. / class_counts
        sample_weights = class_weights[y_tensor.numpy()]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], sampler=sampler)

        model = define_model(num_classes, best_params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        best_train_accuracy = 0
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            model.train()
            train_correct = 0
            train_total = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_accuracy = train_correct / train_total

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"Epoch [{epoch + 1}/{args.epochs}], Train Acc: {train_accuracy:.4f}")

            if epochs_no_improve >= PATIENCE:
                break

        # Save the trained model
        model_save_path = os.path.join(MODEL_SAVE_PATH, 'final_hybrid_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at '{model_save_path}'.")

if __name__ == '__main__':
    main()
