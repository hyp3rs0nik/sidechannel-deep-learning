import pandas as pd
import numpy as np
import time
import optuna
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    start_time = time.time()
    accel_df = pd.read_csv('./data/training/accel.csv')
    keys_df = pd.read_csv('./data/training/keys.csv')
    accel_df['timestamp'] = pd.to_datetime(accel_df['timestamp'])
    keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])
    print(f"Data Loading Time: {time.time() - start_time:.2f} seconds")
    return accel_df, keys_df

def segment_data_refined(accel_df, keys_df, idle_threshold=1.5):
    segmentation_start_time = time.time()
    accel_df['key'] = 'idle'
    num_keys = len(keys_df)

    for i in range(num_keys):
        key = keys_df.iloc[i]['key']
        key_time = keys_df.iloc[i]['timestamp']
        
        seg_start_time = accel_df['timestamp'].min() if i == 0 else keys_df.iloc[i - 1]['timestamp'] + (key_time - keys_df.iloc[i - 1]['timestamp']) / 2
        seg_end_time = accel_df['timestamp'].max() if i == num_keys - 1 else key_time + (keys_df.iloc[i + 1]['timestamp'] - key_time) / 2

        mask = (accel_df['timestamp'] >= seg_start_time) & (accel_df['timestamp'] < seg_end_time)
        accel_df.loc[mask, 'key'] = key
    
    for i in range(num_keys - 1):
        current_end = keys_df.iloc[i]['timestamp'] + pd.Timedelta(seconds=idle_threshold / 2)
        next_start = keys_df.iloc[i + 1]['timestamp'] - pd.Timedelta(seconds=idle_threshold / 2)
        
        if (next_start - current_end).total_seconds() > idle_threshold:
            idle_mask = (accel_df['timestamp'] > current_end) & (accel_df['timestamp'] < next_start)
            accel_df.loc[idle_mask, 'key'] = 'idle'
    
    print(f"Segmentation Time: {time.time() - segmentation_start_time:.2f} seconds")
    return accel_df

def extract_features(df):
    start_time = time.time()
    
    df['displacement_2d'] = np.sqrt(np.cumsum(df['x']**2 + df['y']**2))
    rolling = df[['x', 'y', 'z']].rolling(window=10, min_periods=1)
    
    df['x_min'], df['x_max'] = rolling['x'].agg(['min', 'max']).values.T
    df['y_min'], df['y_max'] = rolling['y'].agg(['min', 'max']).values.T
    df['z_min'], df['z_max'] = rolling['z'].agg(['min', 'max']).values.T
    
    magnitude = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['magnitude'] = magnitude
    df['magnitude_min'], df['magnitude_max'] = magnitude.rolling(window=10, min_periods=1).agg(['min', 'max']).values.T

    print(f"Feature Extraction Time: {time.time() - start_time:.2f} seconds")
    return df[['x', 'y', 'z', 'displacement_2d', 'x_min', 'x_max', 'y_min', 'y_max', 
               'z_min', 'z_max', 'magnitude_min', 'magnitude_max']]

def prepare_data():
    start_time = time.time()
    accel_df, keys_df = load_data()
    segmented_data = segment_data_refined(accel_df, keys_df, idle_threshold=1.5)
    features = extract_features(segmented_data)
    labels = segmented_data['key'].astype(str)
    
    print(f"Data Preparation Time: {time.time() - start_time:.2f} seconds")
    return features, labels

# Prepare data once, outside of the objective function
X, y = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    # Initialize the model with suggested hyperparameters
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Use Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(rfc, X_train, y_train, cv=skf, scoring='accuracy')
    return scores.mean()

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=-1)

# Best parameters from the study
print("Best Parameters:", study.best_params)
print("Best Cross-Validation Accuracy:", study.best_value)

# Train the model with the best parameters
best_rfc = RandomForestClassifier(**study.best_params, random_state=42)
best_rfc.fit(X_train, y_train)

# Save the tuned model
model_path = './data/models/machine_learning/rfc.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_rfc, model_path, compress=8)

# Check the file size and prompt if it exceeds 100MB
file_size_mb = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
if file_size_mb > 100:
    response = input(f"The model file size is {file_size_mb:.2f} MB, which exceeds 100MB. Do you still want to save it? (y/n): ")
    if response.lower() != 'y':
        os.remove(model_path)  # Delete the file if user chooses 'n'
        print("Model file deleted due to size constraints.")
    else:
        print(f"Model saved to {model_path} with size {file_size_mb:.2f} MB.")
else:
    print(f"Model saved to {model_path} with size {file_size_mb:.2f} MB.")

# Evaluate the model
y_pred = best_rfc.predict(X_test)
print(f"Optuna-Tuned RFC Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
