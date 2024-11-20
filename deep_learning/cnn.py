# Deep Learning Model for Keystroke Prediction using Motion Sensors

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
data_dir = Path('./data/training')

def load_and_merge_all_versions(file_type: str):
    files = list(data_dir.glob(f'v*.{file_type}.csv'))
    if not files:
        raise FileNotFoundError(f"No files found for type: {file_type}")
    
    data_frames = [pd.read_csv(file) for file in files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    
    merged_data = merged_data.sort_values(by='timestamp').reset_index(drop=True)
    
    return merged_data

keystrokes = load_and_merge_all_versions('keystrokes')
sensors = load_and_merge_all_versions('sensors')

print(f'Total keystrokes: {len(keystrokes)}, Total sensor data points: {len(sensors)}')

# Parameters
window_size_before = 160  # milliseconds
window_size_after = 160   # milliseconds
num_timesteps = 50        # number of timesteps in each sample

# Prepare Dataset
def create_dataset(keystrokes, sensors, window_size_before, window_size_after, num_timesteps):
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

X, y = create_dataset(keystrokes, sensors, window_size_before, window_size_after, num_timesteps)

print(f'Dataset created with {len(X)} samples.')

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f'Number of classes: {num_classes}')

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Define Dataset Class
class KeystrokeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, num_timesteps, num_features)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Data Loaders
batch_size = 64

train_dataset = KeystrokeDataset(X_train, y_train)
val_dataset = KeystrokeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_features, num_classes, num_filters=64, kernel_size=3, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size//2)
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

num_features = X.shape[2]  # Number of sensor features
model = CNNModel(num_features=num_features, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
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

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects.double() / val_total

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')

        print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
            epoch+1, num_epochs, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc
        ), end='\r')
    print('Best Validation Accuracy: {:.4f}'.format(best_val_acc))

# Train the Model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20)

# Load the Best Model
model.load_state_dict(torch.load('best_model.pth'))

# Load Test Data
test_data_dir = Path('./data/test')

def load_test_data(file_type: str):
    files = list(test_data_dir.glob(f'v*.{file_type}.csv'))
    if not files:
        raise FileNotFoundError(f"No test files found for type: {file_type}")
    
    data_frames = [pd.read_csv(file) for file in files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    
    merged_data = merged_data.sort_values(by='timestamp').reset_index(drop=True)
    
    return merged_data

keystrokes_test = load_test_data('keystrokes')
sensors_test = load_test_data('sensors')

# Prepare Test Dataset
X_test, y_test = create_dataset(keystrokes_test, sensors_test, window_size_before, window_size_after, num_timesteps)
y_test_encoded = le.transform(y_test)

test_dataset = KeystrokeDataset(X_test, y_test_encoded)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate on Test Data
def evaluate_model(model, criterion, test_loader):
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

# Hyperparameter Optimization with Optuna
def objective(trial):
    # Hyperparameters to tune
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
    kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Define the model
    model = CNNModel(num_features=num_features, num_classes=num_classes, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 10  # Use fewer epochs for tuning
    for epoch in range(num_epochs):
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
    return val_acc.item()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

print('Best hyperparameters:', study.best_params)

# Retrain the Model with Best Hyperparameters
best_params = study.best_params

model = CNNModel(
    num_features=num_features,
    num_classes=num_classes,
    num_filters=best_params['num_filters'],
    kernel_size=best_params['kernel_size'],
    dropout_rate=best_params['dropout_rate']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)

# Load the Best Model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on Test Data
evaluate_model(model, criterion, test_loader)
