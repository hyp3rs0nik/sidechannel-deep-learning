import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, irfft

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

def extract_features(merged_data):
    merged_data['x_denoised'] = apply_fft_denoise(merged_data['x'].values)
    merged_data['y_denoised'] = apply_fft_denoise(merged_data['y'].values)
    merged_data['z_denoised'] = apply_fft_denoise(merged_data['z'].values)
    
    merged_data['displacement_2d'] = np.sqrt(
        cumulative_trapezoid(merged_data['x_denoised'], initial=0)**2 + 
        cumulative_trapezoid(merged_data['y_denoised'], initial=0)**2
    )
    
    return merged_data

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

def build_model(input_shape, dropout_rate, learning_rate, num_classes):
    model = Sequential()
    model.add(LSTM(units=100, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

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

    config = load_config(args.config)

    accel_data, keys_data = load_data(args.dataset)
    merged_data = merge_data(accel_data, keys_data)
    merged_data = extract_features(merged_data)

    X, y_categorical = prepare_data(merged_data)
    X_seq, y_seq = create_sequences(X, y_categorical, config['sequence_length'])

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    model = build_model((config['sequence_length'], X_train.shape[2]), config['dropout_rate'], config['learning_rate'], y_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test, config['batch_size'], config['epochs'], config['patience'])

    evaluate_and_save_model(model, X_test, y_test, './data/model/deep_learning/lstm.keras')

if __name__ == "__main__":
    main()
