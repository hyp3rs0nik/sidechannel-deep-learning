# Deep Learning Model for Keystroke Prediction using Motion Sensors

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import optuna
import argparse
import pickle
from scipy.stats import skew, kurtosis
import requests
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

executor = ThreadPoolExecutor(max_workers=1)

def send_best_trial_to_server(study, trial, model):
    if study.best_trial.number == trial.number:
        def send_request():
            url = 'http://192.168.1.139:3000/best_study'
            headers = {'Content-Type': 'application/json'}
            data = {
                'model': model,
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            try:
                requests.put(url, headers=headers, data=json.dumps(data))
            except Exception as e:
                print(f"Request failed: {e}")

        executor.submit(send_request)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Keystroke Prediction Model')
    parser.add_argument('--data_dir', type=str, default='./data/training', help='Path to training data directory')
    parser.add_argument('--test_data_dir', type=str, default='./data/test', help='Path to test data directory')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lstm', 'gru'], help='Model type to use')
    parser.add_argument('--num_trials', type=int, default=50, help='Number of trials for Optuna')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--num_splits', type=int, default=3, help='Number of splits for cross-validation')
    parser.add_argument('--window_size_before', type=int, default=160, help='Window size before keystroke in milliseconds')
    parser.add_argument('--window_size_after', type=int, default=160, help='Window size after keystroke in milliseconds')
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of timesteps in each sample')
    parser.add_argument('--test', action='store_true', help='Run the script in test mode')
    args = parser.parse_args()

    print('Starting Keystroke Prediction Model...')
    print('Arguments:', args)

    model_dir = Path('./data/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'tuned_{args.model}.pth'
    label_encoder_path = model_dir / 'label_encoder.pkl'

    if args.test:
        # Test mode
        if not model_path.exists():
            print(f"No existing trained model tuned_{args.model}, training instead")
            model, le = train_and_save_model(args, model_dir, model_path, label_encoder_path)
        else:
            # Load model and label encoder
            model = load_model(model_path)
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f)
        # Proceed to load test data and evaluate
        evaluate_on_test_data(args, model, le)
    else:
        # Training mode
        # Run training routine
        model, le = train_and_save_model(args, model_dir, model_path, label_encoder_path)
        evaluate_on_test_data(args, model, le)

def train_and_save_model(args, model_dir, model_path, label_encoder_path):
    # Load Data
    data_dir = Path(args.data_dir)
    print('Loading training data...')
    keystrokes = load_and_merge_all_versions(data_dir, 'keystrokes')
    sensors = load_and_merge_all_versions(data_dir, 'sensors')

    print(f'Total keystrokes: {len(keystrokes)}, Total sensor data points: {len(sensors)}')

    # Prepare Dataset
    print('Creating dataset...')
    X_raw, X_engineered, y = create_dataset(
        keystrokes, sensors,
        args.window_size_before,
        args.window_size_after,
        args.num_timesteps
    )

    print(f'Dataset created with {len(X_raw)} samples.')

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f'Number of classes: {num_classes}')

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Hyperparameter Optimization with Optuna
    print('Starting hyperparameter optimization with Optuna...')

    def objective(trial):
        # Hyperparameters to tune
        num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        # Define the model
        if args.model == 'cnn':
            model = DualInputCNNModel(
                num_features=X_raw.shape[2],
                num_engineered_features=X_engineered.shape[1],
                num_classes=num_classes,
                num_timesteps=args.num_timesteps,
                num_filters=num_filters,
                dropout_rate=dropout_rate
            ).to(device)
        elif args.model == 'lstm':
            model = DualInputLSTMModel(
                num_features=X_raw.shape[2],
                num_engineered_features=X_engineered.shape[1],
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)
        elif args.model == 'gru':
            model = DualInputGRUModel(
                num_features=X_raw.shape[2],
                num_engineered_features=X_engineered.shape[1],
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Cross-validation
        skf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=42)
        val_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_encoded)):
            X_train_raw_fold, X_val_raw_fold = X_raw[train_idx], X_raw[val_idx]
            X_train_engineered_fold, X_val_engineered_fold = X_engineered[train_idx], X_engineered[val_idx]
            y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

            train_dataset = KeystrokeDataset(X_train_raw_fold, X_train_engineered_fold, y_train_fold)
            val_dataset = KeystrokeDataset(X_val_raw_fold, X_val_engineered_fold, y_val_fold)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # Training loop
            for epoch in range(args.num_epochs):
                model.train()
                for (inputs_raw, inputs_engineered), labels in train_loader:
                    inputs_raw = inputs_raw.to(device)
                    inputs_engineered = inputs_engineered.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs_raw, inputs_engineered)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Validation accuracy
            model.eval()
            val_corrects = 0
            val_total = 0
            with torch.no_grad():
                for (inputs_raw, inputs_engineered), labels in val_loader:
                    inputs_raw = inputs_raw.to(device)
                    inputs_engineered = inputs_engineered.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs_raw, inputs_engineered)
                    _, preds = torch.max(outputs, 1)

                    val_corrects += torch.sum(preds == labels.data)
                    val_total += labels.size(0)
            val_acc = val_corrects.double() / val_total
            val_accuracies.append(val_acc.item())

            # Optuna trial pruning
            trial.report(val_acc.item(), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_val_acc = np.mean(val_accuracies)
        return mean_val_acc

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(
        objective,
        n_trials=args.num_trials,
        callbacks=[lambda study, trial: send_best_trial_to_server(study, trial, args.model)]
    )

    print('Best hyperparameters:', study.best_params)

    # Retrain the Model with Best Hyperparameters
    print('Retraining the model with best hyperparameters...')
    best_params = study.best_params

    # Split data into training and validation sets
    X_train_raw, X_val_raw, X_train_engineered, X_val_engineered, y_train_encoded, y_val_encoded = train_test_split(
        X_raw, X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Create datasets with data augmentation for training
    train_dataset = KeystrokeDataset(X_train_raw, X_train_engineered, y_train_encoded, transform=data_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = KeystrokeDataset(X_val_raw, X_val_engineered, y_val_encoded)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'cnn':
        model = DualInputCNNModel(
            num_features=X_raw.shape[2],
            num_engineered_features=X_engineered.shape[1],
            num_classes=num_classes,
            num_timesteps=args.num_timesteps,
            num_filters=best_params['num_filters'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif args.model == 'lstm':
        model = DualInputLSTMModel(
            num_features=X_raw.shape[2],
            num_engineered_features=X_engineered.shape[1],
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif args.model == 'gru':
        model = DualInputGRUModel(
            num_features=X_raw.shape[2],
            num_engineered_features=X_engineered.shape[1],
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    print('Training model...')
    train_model(model, criterion, optimizer, train_loader, val_loader, args.num_epochs, patience=10)

    # Save the model and label encoder
    print(f"Saving model to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'model_type': args.model,
        'num_features': X_raw.shape[2],
        'num_engineered_features': X_engineered.shape[1],
        'num_classes': num_classes,
        'num_timesteps': args.num_timesteps
    }, model_path)

    print(f"Saving label encoder to {label_encoder_path}")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le, f)

    return model, le

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model_type = checkpoint['model_type']
    num_features = checkpoint['num_features']
    num_engineered_features = checkpoint['num_engineered_features']
    num_classes = checkpoint['num_classes']
    num_timesteps = checkpoint['num_timesteps']
    best_params = checkpoint['best_params']

    if model_type == 'cnn':
        model = DualInputCNNModel(
            num_features=num_features,
            num_engineered_features=num_engineered_features,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            num_filters=best_params['num_filters'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif model_type == 'lstm':
        model = DualInputLSTMModel(
            num_features=num_features,
            num_engineered_features=num_engineered_features,
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif model_type == 'gru':
        model = DualInputGRUModel(
            num_features=num_features,
            num_engineered_features=num_engineered_features,
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_on_test_data(args, model, le):
    test_data_dir = Path(args.test_data_dir)
    print('Loading test data...')

    # Find all versions based on keystroke files
    keystroke_files = list(test_data_dir.glob('v*.keystrokes.csv'))
    versions = set(f.stem.split('.')[0] for f in keystroke_files)

    for version in versions:
        keystroke_file = test_data_dir / f"{version}.keystrokes.csv"
        sensor_file = test_data_dir / f"{version}.sensors.csv"

        if not keystroke_file.exists() or not sensor_file.exists():
            print(f"Missing files for version {version}, skipping...")
            continue

        print(f"Processing version {version}...")

        # Load data for this version
        keystrokes_test = pd.read_csv(keystroke_file)
        sensors_test = pd.read_csv(sensor_file)

        # Prepare Test Dataset
        print(f'Creating test dataset for {version}...')
        X_test_raw, X_test_engineered, y_test = create_dataset(
            keystrokes_test, sensors_test,
            args.window_size_before,
            args.window_size_after,
            args.num_timesteps
        )

        if len(X_test_raw) == 0:
            print(f"No test samples for version {version}, skipping...")
            continue

        y_test_encoded = le.transform(y_test)

        test_dataset = KeystrokeDataset(X_test_raw, X_test_engineered, y_test_encoded)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Evaluate on Test Data
        print(f'Evaluating model on test data for {version}...')
        criterion = nn.CrossEntropyLoss()
        evaluate_model(model, criterion, test_loader, le)

def load_and_merge_all_versions(data_dir, file_type: str):
    files = list(data_dir.glob(f'v*.{file_type}.csv'))
    if not files:
        raise FileNotFoundError(f"No files found for type: {file_type} in {data_dir}")

    data_frames = [pd.read_csv(file) for file in files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    merged_data = merged_data.sort_values(by='timestamp').reset_index(drop=True)
    return merged_data

def create_dataset(keystrokes, sensors, window_size_before, window_size_after, num_timesteps):
    print('Extracting features...')
    X_raw = []
    X_engineered = []
    y = []
    sensor_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    sensors_array = sensors[sensor_columns].values
    sensors_timestamps = sensors['timestamp'].values

    for idx, row in keystrokes.iterrows():
        key_time = row['timestamp']
        key_label = row['key']
        # Define the window
        start_time = key_time - window_size_before
        end_time = key_time + window_size_after
        # Get sensor data within the window
        idx_mask = (sensors_timestamps >= start_time) & (sensors_timestamps <= end_time)
        window_data = sensors_array[idx_mask]
        if window_data.size == 0:
            continue
        # Resample or interpolate to fixed number of timesteps
        current_timesteps = window_data.shape[0]
        if current_timesteps == num_timesteps:
            sampled_data = window_data
        else:
            # Interpolate to the desired number of timesteps
            indices = np.linspace(0, current_timesteps - 1, num_timesteps)
            sampled_data = np.array([np.interp(indices, np.arange(current_timesteps), window_data[:, i]) for i in range(window_data.shape[1])]).T
        X_raw.append(sampled_data)
        y.append(key_label)

        # Compute engineered features
        features = []
        # Mean and standard deviation for each axis
        for i in range(window_data.shape[1]):
            axis_data = window_data[:, i]
            features.append(np.mean(axis_data))
            features.append(np.std(axis_data))
                        
            rms = np.sqrt(np.mean(axis_data**2))
            features.append(rms)
            
            sma = np.sum(np.abs(axis_data))
            features.append(sma)            
            zero_crossings = np.where(np.diff(np.sign(axis_data)))[0]
            zcr = len(zero_crossings) / len(axis_data)
            features.append(zcr)
            
            fft_coeffs = np.fft.fft(axis_data)
            fft_magnitude = np.abs(fft_coeffs)
            features.append(np.mean(fft_magnitude))
            features.append(np.std(fft_magnitude))
            # Spectral entropy
            psd = fft_magnitude ** 2
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))
            features.append(spectral_entropy)
            # Dominant frequency
            freqs = np.fft.fftfreq(len(axis_data), d=1)
            dominant_freq = freqs[np.argmax(fft_magnitude)]
            features.append(dominant_freq)

            
        X_engineered.append(features)
    return np.array(X_raw), np.array(X_engineered), np.array(y)

# Data Augmentation Function
def data_augmentation(x):
    # Adding Gaussian Noise
    noise = torch.randn_like(x) * 0.01
    x_noisy = x + noise
    return x_noisy

# Define Dataset Class
class KeystrokeDataset(Dataset):
    def __init__(self, X_raw, X_engineered, y, transform=None):
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)  # Shape: (num_samples, num_timesteps, num_features)
        self.X_engineered = torch.tensor(X_engineered, dtype=torch.float32)  # Shape: (num_samples, num_engineered_features)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_raw = self.X_raw[idx]
        x_engineered = self.X_engineered[idx]
        y = self.y[idx]
        if self.transform:
            x_raw = self.transform(x_raw)
        return (x_raw, x_engineered), y
# Define Models
class DualInputCNNModel(nn.Module):
    def __init__(self, num_features, num_engineered_features, num_classes, num_timesteps, num_filters=64, dropout_rate=0.5):
        super(DualInputCNNModel, self).__init__()
        # Raw data processing
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.dropout = nn.Dropout(dropout_rate)
        # Engineered features processing
        self.fc_engineered = nn.Sequential(
            nn.Linear(num_engineered_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Combined fully connected layer
        self.fc_combined = nn.Linear(num_filters * 2 * num_timesteps + 64, num_classes)

    def forward(self, x_raw, x_engineered):
        # x_raw shape: (batch_size, num_timesteps, num_features)
        x = x_raw.permute(0, 2, 1)  # Shape: (batch_size, num_features, num_timesteps)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        # Process engineered features
        x_eng = self.fc_engineered(x_engineered)
        # Concatenate both representations
        x_combined = torch.cat((x, x_eng), dim=1)
        # Final classification layer
        x_out = self.fc_combined(x_combined)
        return x_out

class DualInputLSTMModel(nn.Module):
    def __init__(self, num_features, num_engineered_features, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(DualInputLSTMModel, self).__init__()
        # Raw data processing
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        # Engineered features processing
        self.fc_engineered = nn.Sequential(
            nn.Linear(num_engineered_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Combined fully connected layer
        self.fc_combined = nn.Linear(hidden_size + 64, num_classes)

    def forward(self, x_raw, x_engineered):
        # x_raw shape: (batch_size, num_timesteps, num_features)
        h_0 = torch.zeros(self.lstm.num_layers, x_raw.size(0), self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers, x_raw.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x_raw, (h_0, c_0))  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # Use the last time step
        # Process engineered features
        x_eng = self.fc_engineered(x_engineered)
        # Concatenate both representations
        x_combined = torch.cat((out, x_eng), dim=1)
        # Final classification layer
        x_out = self.fc_combined(x_combined)
        return x_out

class DualInputGRUModel(nn.Module):
    def __init__(self, num_features, num_engineered_features, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(DualInputGRUModel, self).__init__()
        # Raw data processing
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        # Engineered features processing
        self.fc_engineered = nn.Sequential(
            nn.Linear(num_engineered_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Combined fully connected layer
        self.fc_combined = nn.Linear(hidden_size + 64, num_classes)

    def forward(self, x_raw, x_engineered):
        # x_raw shape: (batch_size, num_timesteps, num_features)
        h_0 = torch.zeros(self.gru.num_layers, x_raw.size(0), self.gru.hidden_size).to(device)
        out, _ = self.gru(x_raw, h_0)  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # Use the last time step
        # Process engineered features
        x_eng = self.fc_engineered(x_engineered)
        # Concatenate both representations
        x_combined = torch.cat((out, x_eng), dim=1)
        # Final classification layer
        x_out = self.fc_combined(x_combined)
        return x_out

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience=10):
    print('Training...')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = model.state_dict()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        model.train()
        for (inputs_raw, inputs_engineered), labels in train_loader:
            inputs_raw = inputs_raw.to(device)
            inputs_engineered = inputs_engineered.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs_raw, inputs_engineered)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs_raw.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        # Evaluate on validation set
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0
        with torch.no_grad():
            for (inputs_raw, inputs_engineered), labels in val_loader:
                inputs_raw = inputs_raw.to(device)
                inputs_engineered = inputs_engineered.to(device)
                labels = labels.to(device)

                outputs = model(inputs_raw, inputs_engineered)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs_raw.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_running_corrects.double() / val_total

        print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
            epoch+1, num_epochs, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc
        ))

        # Early stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            epochs_no_improve = 0
            # Save the best model weights
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

def evaluate_model(model, criterion, test_loader, le):
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (inputs_raw, inputs_engineered), labels in test_loader:
            inputs_raw = inputs_raw.to(device)
            inputs_engineered = inputs_engineered.to(device)
            labels = labels.to(device)

            outputs = model(inputs_raw, inputs_engineered)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            test_corrects += torch.sum(preds == labels.data)
            test_total += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_epoch_loss = test_loss / test_total
    test_epoch_acc = test_corrects.double() / test_total

    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(
        test_epoch_loss, test_epoch_acc
    ))

    # Detailed Metrics
    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[str(cls) for cls in le.classes_]
    )

    print('Classification Report:\n', report)

if __name__ == '__main__':
    main()
