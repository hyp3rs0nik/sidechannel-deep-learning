import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy.fft import rfft, irfft
import optuna
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_calculations import feature_functions

def load_config(model_type):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, f'{model_type}_config.json')
    with open(absolute_config_path, 'r') as file:
        return json.load(file)

def load_data(dataset_dir):
    accel_data = pd.read_csv(os.path.join(dataset_dir, 'accel.csv'))
    keys_data = pd.read_csv(os.path.join(dataset_dir, 'keys.csv'))

    accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'])
    keys_data['timestamp'] = pd.to_datetime(keys_data['timestamp'])

    return accel_data.sort_values(by='timestamp'), keys_data.sort_values(by='timestamp')

def merge_data(accel_data, keys_data):
    merged_data = pd.merge_asof(
        accel_data.sort_values("timestamp"),
        keys_data.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("100ms")
    ).dropna(subset=["key"])
    return merged_data


def apply_fft_denoise(data, cutoff=0.1):
    data_fft = rfft(data)
    frequencies = np.fft.rfftfreq(len(data))
    data_fft[np.abs(frequencies) > cutoff] = 0
    return irfft(data_fft, n=len(data))

def extract_features(merged_data):
    selected_features = [
        "rms", "rmse", "x_min", "x_max", "y_min", "y_max",
        "z_min", "z_max", "magnitude_min", "magnitude_max", "sma",
        "displacement_2d"
    ]
    
    feature_data = {
        feature: feature_functions[feature](merged_data).mean() 
        if isinstance(feature_functions[feature](merged_data), pd.Series) else feature_functions[feature](merged_data)
        for feature in selected_features
    }
    features_df = pd.DataFrame([feature_data])
    
    return pd.concat([merged_data.reset_index(drop=True), features_df], axis=1)

def prepare_data(merged_data, apply_denoise=True):
    if apply_denoise:
        merged_data['x'] = apply_fft_denoise(merged_data['x'].values)
        merged_data['y'] = apply_fft_denoise(merged_data['y'].values)
        merged_data['z'] = apply_fft_denoise(merged_data['z'].values)

    X = merged_data[['x', 'y', 'z']].values
    y = merged_data['key'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    return X, y_categorical

def create_sequences(X, y, sequence_length):
    sequences, labels = [], []
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i + sequence_length])
        labels.append(y[i + sequence_length])
    return np.array(sequences), np.array(labels)

def build_model(trial, input_shape, num_classes, model_type):
    model = Sequential()
    units = trial.suggest_int('units', 50, 150, step=25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    if model_type == 'lstm':
        model.add(LSTM(units=units, input_shape=input_shape))
    elif model_type == 'gru':
        model.add(GRU(units=units, input_shape=input_shape))

    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def objective(trial, model_type):
    model = build_model(trial, (config['sequence_length'], X_train.shape[2]), y_train.shape[1], model_type)
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
    parser.add_argument('--model', choices=['lstm', 'gru'], required=True, help='Type of model to use: lstm or gru')
    parser.add_argument('--dataset', default='./data/training/', help='Path to dataset folder containing accel.csv and keys.csv')
    args = parser.parse_args()

    global config
    config = load_config(args.model)

    accel_data, keys_data = load_data(args.dataset)
    merged_data = merge_data(accel_data, keys_data)
    merged_data = extract_features(merged_data)

    X, y_categorical = prepare_data(merged_data)
    X_seq, y_seq = create_sequences(X, y_categorical, config['sequence_length'])

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.model), n_trials=50)

    print(f"Best hyperparameters: {study.best_params}")

    best_model = build_model(study.best_trial, (config['sequence_length'], X_train.shape[2]), y_train.shape[1], args.model)
    model_path = './data/model/deep_learning/' + args.model + '.keras'
    evaluate_and_save_model(best_model, X_test, y_test, model_path)

if __name__ == "__main__":
    main()
