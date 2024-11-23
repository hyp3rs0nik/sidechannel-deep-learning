# Traditional Models with Additional Metrics (Recall, F1 Score)

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.fft import fft

# Load Data


def load_and_merge_all_versions(data_dir, file_type: str):
    files = list(data_dir.glob(f'v*.{file_type}.csv'))
    if not files:
        raise FileNotFoundError(f"No files found for type: {file_type} in {data_dir}")
    
    data_frames = [pd.read_csv(file) for file in files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    merged_data = merged_data.sort_values(by='timestamp').reset_index(drop=True)
    return merged_data

# Preprocess Data

def extract_features(keystrokes, sensors, window_size=240):
    features = []
    labels = []
    sensor_data = sensors[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    timestamps = sensors['timestamp'].values

    for idx, row in keystrokes.iterrows():
        key_time = row['timestamp']
        key_label = row['key']
        # Get sensor data within the window around the keystroke
        window_mask = (timestamps >= key_time - window_size) & (timestamps <= key_time + window_size)
        window_data = sensor_data[window_mask]
        if window_data.size == 0:
            continue
        # Aggregate features (mean, rms, fft-based spectral features)
        mean_features = np.mean(window_data, axis=0)  # Mean of each column (sensor axis)
        rms_features = np.sqrt(np.mean(window_data ** 2, axis=0))  # RMS of each column
        fft_coeffs = fft(window_data, axis=0)
        fft_magnitude = np.abs(fft_coeffs)
        spectral_features = np.hstack([
            np.max(fft_magnitude, axis=0),  # Maximum frequency magnitude per sensor axis
            np.sum(fft_magnitude, axis=0)  # Total spectral sum per sensor axis
        ])
        
        # Combine all features
        feature_vector = np.hstack([mean_features, rms_features, spectral_features])
        features.append(feature_vector)
        labels.append(key_label)
    
    return np.array(features), np.array(labels)


data_dir = Path('./data/training')
keystrokes = load_and_merge_all_versions(data_dir, 'keystrokes')
sensors = load_and_merge_all_versions(data_dir, 'sensors')


features, labels = extract_features(keystrokes, sensors, 240)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Features and Labels
X = features
y = labels_encoded

# Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Traditional Models
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

results = []

# Define Scoring Metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

for name, model in models.items():
    pipeline = Pipeline([
        ('model', model)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    results.append({
        'Model': name,
        'Mean Accuracy': np.mean(scores['test_accuracy']),
        'Std Accuracy': np.std(scores['test_accuracy']),
        'Mean Recall': np.mean(scores['test_recall_macro']),
        'Std Recall': np.std(scores['test_recall_macro']),
        'Mean F1 Score': np.mean(scores['test_f1_macro']),
        'Std F1 Score': np.std(scores['test_f1_macro']),
    })

# Display Results
results_df = pd.DataFrame(results)
print(results_df)
