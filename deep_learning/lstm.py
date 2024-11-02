import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import json
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_PATH = './data/models/deep_learning'
os.makedirs(MODEL_PATH, exist_ok=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from machine_learning.rfc import load_data, align_data, extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1]) if self.num_layers > 1 else h_n[-1]
        out = self.fc(out)
        return out

def objective(trial):
    print("Starting a new Optuna trial...")
    keys_df, sensors_df = load_data()
    aligned_df = align_data(keys_df, sensors_df)
    features_df = extract_features(aligned_df)

    X = features_df.drop(columns=['key']).values
    y = features_df['key'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Parameter suggestions
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.2, 0.4) 
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=len(label_encoder.classes_), dropout_prob=dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered due to no improvement in validation loss.")
            break

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

def train_lstm():
    study = optuna.create_study(
        study_name="lstm_hyperparameter_tuning",
        storage='sqlite:///optuna_lstm_study.db',
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
    best_params = study.best_params

    # Reload data to train final model
    keys_df, sensors_df = load_data()
    aligned_df = align_data(keys_df, sensors_df)
    features_df = extract_features(aligned_df)

    X = features_df.drop(columns=['key']).values
    y = features_df['key'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train final model with best parameters
    model = LSTMModel(
        input_size=X_scaled.shape[1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        output_size=len(label_encoder.classes_),
        dropout_prob=best_params['dropout_prob']
    ).to(device)

    # Save the model, scaler, and label encoder
    torch.save(model.state_dict(), f"{MODEL_PATH}/lstm_model.pth")
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_PATH}/label_encoder.pkl")

    # Save best parameters to JSON
    with open(f"{MODEL_PATH}/lstm_best_params.json", "w") as f:
        json.dump(best_params, f)

    print("Model, scaler, label encoder, and best parameters saved.")

if __name__ == "__main__":
    train_lstm()
