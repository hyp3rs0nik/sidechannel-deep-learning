import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model
import os

# Load the accelerometer and keypress data
accel_data_path = './data/training/v2/accel.csv'
keys_data_path = './data/training/v2/keys.csv'

# Read the data
accel_data = pd.read_csv(accel_data_path)
keys_data = pd.read_csv(keys_data_path)

# Convert timestamps to datetime for both datasets
accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'])
keys_data['timestamp'] = pd.to_datetime(keys_data['timestamp'])

# Clean accelerometer data by aggregating duplicate timestamps
cleaned_accel_data = accel_data.groupby('timestamp').agg({
    'x': 'mean',
    'y': 'mean',
    'z': 'mean'
}).reset_index()

# Compute the RMS of the accelerometer data
cleaned_accel_data['rms'] = (cleaned_accel_data['x']**2 + cleaned_accel_data['y']**2 + cleaned_accel_data['z']**2)**0.5

# Merge with keypress data using the nearest timestamps
merged_data_features = pd.merge_asof(cleaned_accel_data, 
                                     keys_data, 
                                     on='timestamp', 
                                     direction='nearest', 
                                     tolerance=pd.Timedelta('100ms'))

# RMS cross rate (RCR)
cleaned_accel_data['rms_cross_rate'] = ((cleaned_accel_data['rms'].shift(1) - cleaned_accel_data['rms'].mean()) *
                                        (cleaned_accel_data['rms'] - cleaned_accel_data['rms'].mean()) < 0).astype(int)

# Root Mean Square Error (RMSE) over a window
cleaned_accel_data['rmse'] = (cleaned_accel_data['rms'].rolling(window=5, min_periods=1).mean() - cleaned_accel_data['rms'])**2
cleaned_accel_data['rmse'] = cleaned_accel_data['rmse'].rolling(window=5, min_periods=1).mean()**0.5

# Extract dimensional features (x, y, z) and magnitude features
for axis in ['x', 'y', 'z']:
    cleaned_accel_data[f'{axis}_min'] = cleaned_accel_data[axis].rolling(window=5, min_periods=1).min()
    cleaned_accel_data[f'{axis}_max'] = cleaned_accel_data[axis].rolling(window=5, min_periods=1).max()
    cleaned_accel_data[f'{axis}_num_peaks'] = cleaned_accel_data[axis].rolling(window=5, min_periods=1).apply(lambda x: len(find_peaks(x)[0]), raw=True)
    cleaned_accel_data[f'{axis}_num_crests'] = cleaned_accel_data[axis].rolling(window=5, min_periods=1).apply(lambda x: len(find_peaks(-x)[0]), raw=True)

# Euclidean magnitude features (combined x, y, z)
cleaned_accel_data['magnitude'] = (cleaned_accel_data['x']**2 + cleaned_accel_data['y']**2 + cleaned_accel_data['z']**2)**0.5
cleaned_accel_data['magnitude_min'] = cleaned_accel_data['magnitude'].rolling(window=5, min_periods=1).min()
cleaned_accel_data['magnitude_max'] = cleaned_accel_data['magnitude'].rolling(window=5, min_periods=1).max()
cleaned_accel_data['magnitude_num_peaks'] = cleaned_accel_data['magnitude'].rolling(window=5, min_periods=1).apply(lambda x: len(find_peaks(x)[0]), raw=True)
cleaned_accel_data['magnitude_num_crests'] = cleaned_accel_data['magnitude'].rolling(window=5, min_periods=1).apply(lambda x: len(find_peaks(-x)[0]), raw=True)

# Signal Magnitude Area (SMA)
cleaned_accel_data['sma'] = cleaned_accel_data[['x', 'y', 'z']].abs().sum(axis=1).rolling(window=5, min_periods=1).sum()

# Prepare the feature set based strictly on the paper's features
X_strict = cleaned_accel_data[['rms', 'rmse', 'rms_cross_rate', 'x_min', 'x_max', 'x_num_peaks', 'x_num_crests',
                               'y_min', 'y_max', 'y_num_peaks', 'y_num_crests', 'z_min', 'z_max', 'z_num_peaks', 'z_num_crests',
                               'magnitude_min', 'magnitude_max', 'magnitude_num_peaks', 'magnitude_num_crests', 'sma']]

# Drop any rows with missing target labels (key)
y_strict = merged_data_features['key'].dropna()
X_strict = X_strict.loc[y_strict.index]

# Fill missing values in the features
X_strict = X_strict.fillna(0)

# Split the data into training and test sets
X_train_strict, X_test_strict, y_train_strict, y_test_strict = train_test_split(X_strict, y_strict, test_size=0.2, random_state=42)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_strict, y_train_strict)

# Get the best parameters and retrain the model
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train the Random Forest with the best hyperparameters
best_rfc = RandomForestClassifier(**best_params, random_state=42)
best_rfc.fit(X_train_strict, y_train_strict)

# Predict on the test set
y_pred_best = best_rfc.predict(X_test_strict)

# Evaluate the model's performance
accuracy_best = accuracy_score(y_test_strict, y_pred_best)
classification_rep_best = classification_report(y_test_strict, y_pred_best)

print(f"Accuracy after Hyperparameter Tuning: {accuracy_best}")
print(f"Classification Report after Hyperparameter Tuning:\n{classification_rep_best}")

# Save the model
model_name = 'random_forest_best_model'
model_save_path = f'./data/models/machine_learning/{model_name}.pkl'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(best_rfc, model_save_path)
print(f"Model saved to: {model_save_path}")
