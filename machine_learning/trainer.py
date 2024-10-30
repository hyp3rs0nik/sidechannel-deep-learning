import pandas as pd
import numpy as np
import time
import optuna
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data():
    start_time = time.time()
    accel_df = pd.read_csv('./data/training/accel.csv')
    keys_df = pd.read_csv('./data/training/keys.csv')
    accel_df['timestamp'] = pd.to_datetime(accel_df['timestamp'])
    keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])
    print(f"Data Loading Time: {time.time() - start_time:.2f} seconds")
    return accel_df, keys_df

def merge_data(accel_df, keys_df, idle_threshold=1.5):
    merged_df = pd.merge_asof(
        accel_df.sort_values('timestamp'),
        keys_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=idle_threshold)
    )
    merged_df['key'] = merged_df['key'].fillna('idle')
    return merged_df

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
    features = df[['x', 'y', 'z', 'displacement_2d', 'x_min', 'x_max', 'y_min', 'y_max', 
                   'z_min', 'z_max', 'magnitude_min', 'magnitude_max']]
    labels = df['key'].astype(str)
    return features, labels

def prepare_data():
    start_time = time.time()
    accel_df, keys_df = load_data()
    merged_df = merge_data(accel_df, keys_df, idle_threshold=1.5)
    features, labels = extract_features(merged_df)
    print(f"Data Preparation Time: {time.time() - start_time:.2f} seconds")
    return features, labels

# Prepare data
X, y = prepare_data()
X = X.dropna().reset_index(drop=True)
y = y[X.index]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 1: Initial Stratified Split for Holdout Set
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

# Step 2: Further Stratified Split for Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
)

# Step 3: Scale Training and Validation Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)

# Define the Optuna objective function with cross-validation
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 170, 190)
    max_depth = trial.suggest_int('max_depth', 20, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 4)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)
    max_features = trial.suggest_categorical('max_features', [None])
    
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Use StratifiedKFold for cross-validation within the training data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rfc, X_train_scaled, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Step 4: Hyperparameter Optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, n_jobs=-1)

print("Best Parameters:", study.best_params)
print("Best Cross-Validation Accuracy:", study.best_value)

# Step 5: Train the Best Model on the Full Training Set
best_rfc = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
best_rfc.fit(X_train_scaled, y_train)

# Step 6: Evaluate on the Holdout Set
y_holdout_pred = best_rfc.predict(X_holdout_scaled)
holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)

holdout_data = pd.DataFrame(X_holdout_scaled, columns=[f'feature_{i}' for i in range(X_holdout_scaled.shape[1])])
holdout_data['true_label'] = y_holdout
holdout_data['predicted_label'] = y_holdout_pred

# Save to CSV for visualization purposes
holdout_data.to_csv('./data/training/holdout_data.csv', index=False)

print("Holdout data and predictions saved to './data/training/holdout_data.csv'")

print(f"Holdout Set Accuracy: {holdout_accuracy:.4f}")
print("Holdout Set Classification Report:")
print(classification_report(label_encoder.inverse_transform(y_holdout), label_encoder.inverse_transform(y_holdout_pred)))
print("Holdout Set Confusion Matrix:")
print(confusion_matrix(y_holdout, y_holdout_pred))

# Save the model, scaler, and label encoder
model_path = './data/models/machine_learning/rfc.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump({'model': best_rfc, 'scaler': scaler, 'label_encoder': label_encoder}, model_path, compress=9)
