# Import Necessary Libraries
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.fft import fft
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Load Data
def load_and_merge_all_versions(data_dir, file_pattern: str):
    files = list(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern} in {data_dir}")
    
    data_frames = [pd.read_csv(file) for file in files]
    merged_data = pd.concat(data_frames, ignore_index=True)
    if 'timestamp' in merged_data.columns:
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

# Paths to Data
training_data_dir = Path('./data/training')
test_data_dir = Path('./data/test')

# Load Training Data
keystrokes = load_and_merge_all_versions(training_data_dir, 'v*.keystrokes.csv')
sensors = load_and_merge_all_versions(training_data_dir, 'v*.sensors.csv')

# Extract Features and Labels
features, labels = extract_features(keystrokes, sensors, 160)

# Encode Labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = labels_encoded

# Define Models
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

# Hyperparameter Tuning with Optuna
def objective(trial, model_name, X, y):
    if model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        model = RandomForestClassifier(**params)
    elif model_name == 'SVM':
        params = {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
            'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e-1),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        }
        model = SVC(**params)
    elif model_name == 'LogisticRegression':
        params = {
            'C': trial.suggest_loguniform('C', 1e-3, 1e2),
            'penalty': trial.suggest_categorical('penalty', ['l2', None]),
            'solver': 'lbfgs'
        }
        # Remove 'penalty' if None
        if params['penalty'] is None:
            params.pop('penalty')
        model = LogisticRegression(max_iter=1000, **params)
    else:
        raise ValueError("Model not supported for hyperparameter tuning.")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1, error_score='raise')
    return np.mean(scores['test_score'])

# Tuning and Training Models
best_models = {}
for name, model in models.items():
    print(f"Tuning {name}...")
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, name, X, y)
    study.optimize(func, n_trials=30)
    best_params = study.best_params
    print(f"Best parameters for {name}: {best_params}")
    # Update model with best parameters
    if name == 'RandomForest':
        best_model = RandomForestClassifier(**best_params)
    elif name == 'SVM':
        best_model = SVC(**best_params, probability=True)
    elif name == 'LogisticRegression':
        # Remove 'penalty' if None
        if best_params.get('penalty', 'l2') is None:
            best_params.pop('penalty')
        best_model = LogisticRegression(max_iter=1000, **best_params)
    best_models[name] = best_model

# Define Scoring Metrics
scoring = ['accuracy', 'recall_macro', 'f1_macro']

# Training Results
training_results = []

for name, model in best_models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    training_results.append({
        'Model': name,
        'Mean Accuracy': np.mean(scores['test_accuracy']),
        'Mean Recall': np.mean(scores['test_recall_macro']),
        'Mean F1 Score': np.mean(scores['test_f1_macro']),
    })

training_results_df = pd.DataFrame(training_results)
print("Training Results:")
print(training_results_df)

# Load Test Data
test_keystrokes = pd.read_csv(test_data_dir / 'v3.keystrokes.csv')
test_sensors = pd.read_csv(test_data_dir / 'v3.sensors.csv')

# Extract Features and Labels for Test Data
X_test_raw, y_test_raw = extract_features(test_keystrokes, test_sensors, 160)
y_test_raw_encoded = le.transform(y_test_raw)  # Use the same label encoder

# Scale Test Features
X_test = scaler.transform(X_test_raw)
y_test = y_test_raw_encoded

# Test Results
# Test Results
test_results = []

for name, model in best_models.items():
    # Fit the model on the entire training data
    model.fit(X, y)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Append results
    test_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Recall (macro)': recall,
        'F1 Score (macro)': f1,
        'Weighted Precision': precision_weighted,
        'Weighted Recall': recall_weighted,
        'Weighted F1': f1_weighted,
    })

    # For RandomForest, extract feature importances
    if name == 'RandomForest':
        feature_importances = model.feature_importances_
        # Get feature names
        num_sensor_axes = 6  # ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        feature_names = []
        # Mean features
        feature_names.extend([f'mean_{axis}' for axis in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']])
        # RMS features
        feature_names.extend([f'rms_{axis}' for axis in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']])
        # Spectral features
        feature_names.extend([f'max_fft_{axis}' for axis in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']])
        feature_names.extend([f'sum_fft_{axis}' for axis in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']])

        # Plot feature importances
        importances = pd.Series(feature_importances, index=feature_names)
        importances_sorted = importances.sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        importances_sorted.plot(kind='bar')
        plt.title('Feature Importances from Random Forest')
        plt.ylabel('Importance')
        plt.tight_layout()
        # Save the plot
        plt.savefig('./docs/best_features.png')
        plt.close()

# Create Test Results DataFrame
test_results_df = pd.DataFrame(test_results)
print("\nTest Results:")
print(test_results_df)

# Save Results to CSV
training_results_df.to_csv('training_results.csv', index=False)
test_results_df.to_csv('test_results.csv', index=False)
