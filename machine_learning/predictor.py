import numpy as np
import pandas as pd
import joblib
from collections import deque
from sklearn.preprocessing import StandardScaler
import keyboard  # For monitoring numpad key press
import socket
import time

# Load the trained model, scaler, and label encoder
model_data = joblib.load('./data/models/machine_learning/rfc.pkl')
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

# Configurations
WINDOW_SIZE = 100  # Size of the sliding window (adjust based on training setup)
sensor_buffer = deque(maxlen=WINDOW_SIZE)  # Circular buffer for sensor data
SERVER_HOST = '0.0.0.0'  # Bind to all network interfaces
SERVER_PORT = 65432       # Port to listen on

# Function to preprocess data in the buffer
def preprocess_buffer(buffer):
    buffer_df = pd.DataFrame(buffer, columns=['timestamp', 'x', 'y', 'z'])
    buffer_df['displacement_2d'] = np.sqrt(np.cumsum(buffer_df['x']**2 + buffer_df['y']**2))
    rolling = buffer_df[['x', 'y', 'z']].rolling(window=10, min_periods=1)
    buffer_df['x_min'], buffer_df['x_max'] = rolling['x'].agg(['min', 'max']).values.T
    buffer_df['y_min'], buffer_df['y_max'] = rolling['y'].agg(['min', 'max']).values.T
    buffer_df['z_min'], buffer_df['z_max'] = rolling['z'].agg(['min', 'max']).values.T
    magnitude = np.sqrt(buffer_df['x']**2 + buffer_df['y']**2 + buffer_df['z']**2)
    buffer_df['magnitude'] = magnitude
    buffer_df['magnitude_min'], buffer_df['magnitude_max'] = magnitude.rolling(window=10, min_periods=1).agg(['min', 'max']).values.T

    features = buffer_df[['x', 'y', 'z', 'displacement_2d', 'x_min', 'x_max', 'y_min', 'y_max',
                          'z_min', 'z_max', 'magnitude_min', 'magnitude_max']]
    features_scaled = scaler.transform(features)
    return features_scaled

# Prediction function
def predict_keystroke():
    features_scaled = preprocess_buffer(sensor_buffer)
    predictions = model.predict(features_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)
    print("Predicted Key:", predicted_labels[-1])  # Last prediction as keystroke inference

# Keyboard event callback
def on_key_event(key_event):
    if key_event.event_type == 'down' and key_event.name in map(str, range(10)):
        print(f"Numpad Key {key_event.name} pressed, processing buffer...")
        keyboard.unhook_all()  # Temporarily disable further key detection
        predict_keystroke()    # Process and predict
        time.sleep(0.1)        # Small delay to prevent multiple detections
        keyboard.hook(on_key_event)  # Re-enable key detection
        print("Resuming sensor data stream...")

# Network listener for sensor data
def start_sensor_stream():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)
        print(f"Listening for sensor data on {SERVER_HOST}:{SERVER_PORT}...")
        
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # Read incoming data
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break  # End connection if no data
                for line in data.strip().split('\n'):
                    # Parse CSV line
                    try:
                        x, y, z = map(float, line.split(','))
                        timestamp = time.time()
                        sensor_buffer.append([timestamp, x, y, z])  # Append to buffer
                    except ValueError:
                        print("Received invalid data format:", line)

# Attach keyboard hook
keyboard.hook(on_key_event)

# Start the network sensor stream
try:
    start_sensor_stream()
except KeyboardInterrupt:
    print("Streaming terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
