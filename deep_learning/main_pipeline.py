# Deep Learning Model for Keystroke Prediction using Motion Sensors

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import optuna
import argparse
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            user_input = input(f"No existing trained model tuned_{args.model} found, do you want to train? [y/n]: ")
            if user_input.lower() != 'y':
                print("Exiting...")
                return
            else:
                # Run training routine
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
    X, y = create_dataset(
        keystrokes, sensors,
        args.window_size_before,
        args.window_size_after,
        args.num_timesteps
    )

    print(f'Dataset created with {len(X)} samples.')

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f'Number of classes: {num_classes}')

    # Define Dataset Class
    # Moved KeystrokeDataset class definition outside the main function

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
            model = CNNModel(
                num_features=X.shape[2],
                num_classes=num_classes,
                num_timesteps=args.num_timesteps,
                num_filters=num_filters,
                dropout_rate=dropout_rate
            ).to(device)
        elif args.model == 'lstm':
            model = LSTMModel(
                num_features=X.shape[2],
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)
        elif args.model == 'gru':
            model = GRUModel(
                num_features=X.shape[2],
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Cross-validation
        skf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=42)
        val_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

            train_dataset = KeystrokeDataset(X_train_fold, y_train_fold)
            val_dataset = KeystrokeDataset(X_val_fold, y_val_fold)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # Training loop
            for epoch in range(args.num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Validation accuracy
            model.eval()
            val_corrects = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
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
    study.optimize(objective, n_trials=args.num_trials)

    print('Best hyperparameters:', study.best_params)

    # Retrain the Model with Best Hyperparameters
    print('Retraining the model with best hyperparameters...')
    best_params = study.best_params

    # Using all data for training
    train_dataset = KeystrokeDataset(X, y_encoded)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == 'cnn':
        model = CNNModel(
            num_features=X.shape[2],
            num_classes=num_classes,
            num_timesteps=args.num_timesteps,
            num_filters=best_params['num_filters'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif args.model == 'lstm':
        model = LSTMModel(
            num_features=X.shape[2],
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif args.model == 'gru':
        model = GRUModel(
            num_features=X.shape[2],
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    print('Training model...')
    train_model(model, criterion, optimizer, train_loader, args.num_epochs)

    # Save the model and label encoder
    print(f"Saving model to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'model_type': args.model,
        'num_features': X.shape[2],
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
    num_classes = checkpoint['num_classes']
    num_timesteps = checkpoint['num_timesteps']
    best_params = checkpoint['best_params']

    if model_type == 'cnn':
        model = CNNModel(
            num_features=num_features,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            num_filters=best_params['num_filters'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif model_type == 'lstm':
        model = LSTMModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
    elif model_type == 'gru':
        model = GRUModel(
            num_features=num_features,
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
        X_test, y_test = create_dataset(
            keystrokes_test, sensors_test,
            args.window_size_before,
            args.window_size_after,
            args.num_timesteps
        )

        if len(X_test) == 0:
            print(f"No test samples for version {version}, skipping...")
            continue

        y_test_encoded = le.transform(y_test)

        test_dataset = KeystrokeDataset(X_test, y_test_encoded)
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
    X = []
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
        X.append(sampled_data)
        y.append(key_label)
    return np.array(X), np.array(y)

# Define Dataset Class
class KeystrokeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, num_timesteps, num_features)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Models
class CNNModel(nn.Module):
    def __init__(self, num_features, num_classes, num_timesteps, num_filters=64, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * 2 * num_timesteps, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_timesteps, num_features)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, num_features, num_timesteps)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_timesteps, num_features)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

class GRUModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_timesteps, num_features)
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(device)
        out, _ = self.gru(x, h_0)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    print('Training...')
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Shape: (batch_size, num_timesteps, num_features)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # Shape: (batch_size, num_classes)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print('Epoch {}/{}: Loss: {:.4f}, Acc: {:.4f}'.format(
            epoch+1, num_epochs, epoch_loss, epoch_acc
        ))

def evaluate_model(model, criterion, test_loader, le):
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            test_total += labels.size(0)
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
