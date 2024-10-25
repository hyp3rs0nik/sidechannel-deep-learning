import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import optuna
from optuna.integration import TFKerasPruningCallback

def load_and_preprocess_data(accel_path, keys_path):
    # Load data
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)
    
    # Convert timestamps to datetime
    accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"])
    keys_data["timestamp"] = pd.to_datetime(keys_data["timestamp"])
    
    # Aggregate accelerometer data by timestamp (mean values)
    cleaned_data = accel_data.groupby("timestamp", as_index=False).agg(
        {"x": "mean", "y": "mean", "z": "mean"}
    )
    
    # Calculate RMS (Root Mean Square)
    cleaned_data["rms"] = np.sqrt((cleaned_data[["x", "y", "z"]] ** 2).sum(axis=1))
    
    merged_data = pd.merge_asof(
        cleaned_data.sort_values("timestamp"),
        keys_data.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    merged_data.dropna(subset=["key"], inplace=True)
    
    return merged_data

def extract_features(cleaned_data):
    # Initialize the DataFrame to store features
    features_df = pd.DataFrame(index=cleaned_data.index)
    
    # RMS features
    features_df["rms"] = cleaned_data["rms"]
    features_df["rmse"] = features_df["rms"].rolling(window=5, min_periods=1).apply(
        lambda x: np.sqrt(np.mean((x - x.mean()) ** 2)), raw=False
    )
    features_df["rms_cross_rate"] = (
        (features_df["rms"].shift(1) - features_df["rms"].mean()) *
        (features_df["rms"] - features_df["rms"].mean())
        < 0
    ).astype(int)
    
    # Function to calculate number of peaks and crests
    def count_peaks_crests(series):
        # Detect all peaks
        peaks, _ = find_peaks(series)
        # Detect crests with prominence > 1 (adjust as needed)
        crests, _ = find_peaks(series, prominence=1)
        return len(peaks), len(crests)
    
    # Axis-wise features
    for axis in ["x", "y", "z"]:
        # Rolling window
        rolling_series = cleaned_data[axis].rolling(window=5, min_periods=1)
        
        # Minimum and Maximum
        features_df[f"{axis}_min"] = rolling_series.min()
        features_df[f"{axis}_max"] = rolling_series.max()
        
        # Number of Peaks and Crests
        features_df[f"{axis}_num_peaks"] = rolling_series.apply(
            lambda x: count_peaks_crests(x)[0], raw=False
        )
        features_df[f"{axis}_num_crests"] = rolling_series.apply(
            lambda x: count_peaks_crests(x)[1], raw=False
        )
    
    # Magnitude features based on RMS
    rolling_rms = features_df["rms"].rolling(window=5, min_periods=1)
    features_df["magnitude_min"] = rolling_rms.min()
    features_df["magnitude_max"] = rolling_rms.max()
    features_df["magnitude_num_peaks"] = rolling_rms.apply(
        lambda x: count_peaks_crests(x)[0], raw=False
    )
    features_df["magnitude_num_crests"] = rolling_rms.apply(
        lambda x: count_peaks_crests(x)[1], raw=False
    )
    
    # Signal Magnitude Area (SMA)
    features_df["sma"] = (
        cleaned_data[["x", "y", "z"]].abs().sum(axis=1).rolling(window=5, min_periods=1).sum()
    )
    
    # Handle any remaining NaN values by filling them with zeros
    features_df.fillna(0, inplace=True)
    
    return features_df

def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length + 1):
        X_seq = X.iloc[i:i+sequence_length].values
        y_seq = y[i+sequence_length-1]
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    return np.array(X_sequences), np.array(y_sequences)

def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, num_classes, class_weight_dict, sequence_length, num_features):
    # Suggest hyperparameters
    units = trial.suggest_categorical('units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Define the model
    model = Sequential()
    model.add(GRU(units, input_shape=(sequence_length, num_features), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weight_dict,
        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
    return val_accuracy

def train_and_evaluate_gru_with_optuna(X, y):
    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Create sequences
    sequence_length = 50  # Adjust as needed
    X_sequences, y_sequences = create_sequences(X, y_encoded, sequence_length)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
    )

    # Reshape data for scaling
    num_samples_train, seq_len_train, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)

    num_samples_val, seq_len_val, _ = X_val.shape
    X_val_reshaped = X_val.reshape(-1, num_features)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)

    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(num_samples_train, seq_len_train, num_features)
    X_val_scaled = X_val_scaled.reshape(num_samples_val, seq_len_val, num_features)

    # Compute class weights to handle class imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    # Optimize hyperparameters with Optuna
    def objective_wrapper(trial):
        return objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, num_classes, class_weight_dict, sequence_length, num_features)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_wrapper, n_trials=500)

    print(f"\nBest Hyperparameters: {study.best_params}")

    # Train final model with best hyperparameters
    best_params = study.best_params
    units = best_params['units']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    # Define the model
    model = Sequential()
    model.add(GRU(units, input_shape=(sequence_length, num_features), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
    print('\nValidation Accuracy:', val_accuracy)

    # Predict on validation set
    y_pred_probs = model.predict(X_val_scaled)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Fix the error by converting class labels to strings
    target_names = [str(label) for label in label_encoder.classes_]

    # Classification report
    report = classification_report(y_val, y_pred_classes, target_names=target_names)
    print('\nClassification Report:\n', report)

    return model, history, label_encoder

def save_model(model, model_name, path="./data/models/deep_learning"):
    model_path = os.path.join(path, f"{model_name}.keras")
    os.makedirs(path, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    # Paths to the data files
    accel_data_path = "./data/training/accel.csv"
    keys_data_path = "./data/training/keys.csv"

    # Load and preprocess data
    data = load_and_preprocess_data(accel_data_path, keys_data_path)
    features = extract_features(data)

    # Define the original feature list
    original_feature_list = [
        "rms",
        "rmse",
        "rms_cross_rate",
        "x_min",
        "x_max",
        "x_num_peaks",
        "x_num_crests",
        "y_min",
        "y_max",
        "y_num_peaks",
        "y_num_crests",
        "z_min",
        "z_max",
        "z_num_peaks",
        "z_num_crests",
        "magnitude_min",
        "magnitude_max",
        "magnitude_num_peaks",
        "magnitude_num_crests",
        "sma",
    ]

    # Verify that all original features are present
    missing_features = [feature for feature in original_feature_list if feature not in features.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing: {missing_features}")

    # Define X and y
    X = features[original_feature_list]
    y = data["key"]

    # Diagnostic: Check columns and data types
    print("Columns in X:", X.columns)
    print("Data types in X:")
    print(X.dtypes)

    print("\nClass distribution in target variable:")
    print(y.value_counts())

    # Train and evaluate GRU model with Optuna hyperparameter tuning
    model, history, label_encoder = train_and_evaluate_gru_with_optuna(X, y)

    # Save the trained model
    save_model(model, "gru_model")
