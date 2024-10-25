import os
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy.fft import rfft, irfft
import optuna

def load_config(config_path='lstm_config.json'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, config_path)
    with open(absolute_config_path, 'r') as file:
        return json.load(file)

def load_data(dataset_dir):
    accel_data = pd.read_csv(os.path.join(dataset_dir, 'accel.csv'))
    keys_data = pd.read_csv(os.path.join(dataset_dir, 'keys.csv'))

    accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'])
    keys_data['timestamp'] = pd.to_datetime(keys_data['timestamp'])

    accel_data = accel_data.sort_values(by='timestamp')
    keys_data = keys_data.sort_values(by='timestamp')

    return accel_data, keys_data

def merge_data(accel_data, keys_data):
    accel_data.set_index('timestamp', inplace=True)
    keys_data.set_index('timestamp', inplace=True)
    merged_data = pd.merge_asof(accel_data, keys_data, left_index=True, right_index=True, direction='backward')
    return merged_data

def apply_fft_denoise(data, cutoff=0.1):
    data_fft = rfft(data)
    frequencies = np.fft.rfftfreq(len(data))
    data_fft[np.abs(frequencies) > cutoff] = 0
    return irfft(data_fft, n=len(data))

# Feature extraction helper functions
def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def calculate_rmse(data):
    return np.sqrt(np.mean((data - np.mean(data))**2))

def calculate_cross_rate(data):
    return np.sum(np.diff(np.sign(data)) != 0)

def calculate_peaks(data):
    return len(find_peaks(data)[0])

def calculate_crests(data):
    return len(find_peaks(-data)[0])

def calculate_sma(data, window_size=5):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def extract_features(merged_data):
    merged_data['x_denoised'] = apply_fft_denoise(merged_data['x'].values)
    merged_data['y_denoised'] = apply_fft_denoise(merged_data['y'].values)
    merged_data['z_denoised'] = apply_fft_denoise(merged_data['z'].values)

    # Calculating magnitude
    merged_data['magnitude'] = np.sqrt(merged_data['x_denoised']**2 + merged_data['y_denoised']**2 + merged_data['z_denoised']**2)

    # Feature extraction
    features = pd.DataFrame({
        "rms": calculate_rms(merged_data['magnitude']),
        "rmse": calculate_rmse(merged_data['magnitude']),
        "rms_cross_rate": calculate_cross_rate(merged_data['magnitude']),
        "x_min": merged_data['x_denoised'].min(),
        "x_max": merged_data['x_denoised'].max(),
        "x_num_peaks": calculate_peaks(merged_data['x_denoised']),
        "x_num_crests": calculate_crests(merged_data['x_denoised']),
        "y_min": merged_data['y_denoised'].min(),
        "y_max": merged_data['y_denoised'].max(),
        "y_num_peaks": calculate_peaks(merged_data['y_denoised']),
        "y_num_crests": calculate_crests(merged_data['y_denoised']),
        "z_min": merged_data['z_denoised'].min(),
        "z_max": merged_data['z_denoised'].max(),
        "z_num_peaks": calculate_peaks(merged_data['z_denoised']),
        "z_num_crests": calculate_crests(merged_data['z_denoised']),
        "magnitude_min": merged_data['magnitude'].min(),
        "magnitude_max": merged_data['magnitude'].max(),
        "magnitude_num_peaks": calculate_peaks(merged_data['magnitude']),
        "magnitude_num_crests": calculate_crests(merged_data['magnitude']),
        "sma": calculate_sma(merged_data['magnitude']).mean()  # Mean value of SMA
    }, index=[0])

    return pd.concat([merged_data.reset_index(drop=True), features], axis=1)

def prepare_data(merged_data):
    X = merged_data[['x', 'y', 'z']].values
    y = merged_data['key'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    return X, y_categorical

def create_sequences(X, y, sequence_length):
    sequences = []
    labels = []
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i + sequence_length])
        labels.append(y[i + sequence_length])
    return np.array(sequences), np.array(labels)

def build_model(trial, input_shape, num_classes):
    model = Sequential()
    
    # Suggest hyperparameters
    units = trial.suggest_int('units', 50, 150, step=25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model.add(LSTM(units=units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    model = build_model(trial, (config['sequence_length'], X_train.shape[2]), y_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=config['epochs'], batch_size=config['batch_size'], 
                        callbacks=[early_stopping], verbose=1)

    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

def evaluate_and_save_model(model, X_test, y_test, model_save_path):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='./data/training/', help='Path to dataset folder containing accel.csv and keys.csv')
    parser.add_argument('--config', default='lstm_config.json', help='Path to config file')
    args = parser.parse_args()

    global config
    config = load_config(args.config)

    accel_data, keys_data = load_data(args.dataset)
    merged_data = merge_data(accel_data, keys_data)
    merged_data = extract_features(merged_data)

    X, y_categorical = prepare_data(merged_data)
    X_seq, y_seq = create_sequences(X, y_categorical, config['sequence_length'])

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f"Best hyperparameters: {study.best_params}")

    best_model = build_model(study.best_trial, (config['sequence_length'], X_train.shape[2]), y_train.shape[1])
    evaluate_and_save_model(best_model, X_test, y_test, './data/model/deep_learning/lstm.keras')

if __name__ == "__main__":
    main()
