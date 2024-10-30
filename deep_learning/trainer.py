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
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer for Feature Extraction
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window=10):
        self.window = window
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Calculate displacement
        df['displacement_2d'] = np.sqrt(np.cumsum(df['x']**2 + df['y']**2))
        
        # Rolling window calculations
        rolling = df[['x', 'y', 'z']].rolling(window=self.window, min_periods=1)
        df['x_min'], df['x_max'] = rolling['x'].agg(['min', 'max']).values.T
        df['y_min'], df['y_max'] = rolling['y'].agg(['min', 'max']).values.T
        df['z_min'], df['z_max'] = rolling['z'].agg(['min', 'max']).values.T
        
        # Magnitude calculations
        magnitude = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df['magnitude'] = magnitude
        df['magnitude_min'], df['magnitude_max'] = magnitude.rolling(window=self.window, min_periods=1).agg(['min', 'max']).values.T
        
        # Handle NaNs with backfill and forward fill
        df = df.bfill().ffill()
        
        # Select relevant features
        features = df[['x', 'y', 'z', 'displacement_2d', 'x_min', 'x_max', 'y_min', 'y_max', 
                       'z_min', 'z_max', 'magnitude_min', 'magnitude_max']]
        return features

def load_data():
    start_time = time.time()
    accel_df = pd.read_csv('./data/training/accel.csv')
    keys_df = pd.read_csv('./data/training/keys.csv')
    accel_df['timestamp'] = pd.to_datetime(accel_df['timestamp'])
    keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])
    print(f"Data Loading Time: {time.time() - start_time:.2f} seconds")
    return accel_df, keys_df

def merge_data(accel_df, keys_df, idle_threshold=1.5):
    # Merge accelerometer data with keypress data using asof merge for nearest timestamp within tolerance
    merged_df = pd.merge_asof(
        accel_df.sort_values('timestamp'),
        keys_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=idle_threshold)
    )
    # Assign 'idle' to rows without a keypress
    merged_df['key'] = merged_df['key'].fillna('idle')
    return merged_df

def extract_features(df):
    # This function is now integrated into the FeatureExtractor transformer
    pass

def prepare_data():
    start_time = time.time()
    accel_df, keys_df = load_data()
    merged_df = merge_data(accel_df, keys_df, idle_threshold=1.5)
    # Feature extraction will be handled within the pipeline
    print(f"Data Preparation Time: {time.time() - start_time:.2f} seconds")
    return merged_df

# Prepare data
merged_df = prepare_data()

# Remove any rows with NaN values (if any) before splitting
merged_df = merged_df.dropna().reset_index(drop=True)

# Separate features and labels
X = merged_df.drop('key', axis=1)
y = merged_df['key'].astype(str)

# Encode labels based on training data
label_encoder = LabelEncoder()

# Split the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize Label Encoder and fit on training labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define the pipeline
pipeline = Pipeline([
    ('feature_extraction', FeatureExtractor(window=10)),
    ('scaler', StandardScaler()),
    ('rfc', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('rfc__n_estimators', 50, 200)
    max_depth = trial.suggest_int('rfc__max_depth', 5, 30)
    min_samples_split = trial.suggest_int('rfc__min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('rfc__min_samples_leaf', 1, 5)
    max_features = trial.suggest_categorical('rfc__max_features', ['sqrt', 'log2', None])
    
    # Set the hyperparameters
    pipeline.set_params(
        rfc__n_estimators=n_estimators,
        rfc__max_depth=max_depth,
        rfc__min_samples_split=min_samples_split,
        rfc__min_samples_leaf=min_samples_leaf,
        rfc__max_features=max_features
    )
    
    # Use StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Create and optimize the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)

print("Best Parameters:", study.best_params)
print("Best Cross-Validation Accuracy:", study.best_value)

# Set the best parameters to the pipeline
pipeline.set_params(**study.best_params)

# Train the pipeline on the entire training set
pipeline.fit(X_train, y_train_encoded)

# Save the pipeline, scaler, and label encoder
model_path = './data/models/machine_learning/rfc_pipeline.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump({'model_pipeline': pipeline, 'label_encoder': label_encoder}, model_path, compress=9)

# Check Model Size
file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
if file_size_mb > 100:
    response = input(f"The model file size is {file_size_mb:.2f} MB, which exceeds 100MB. Do you still want to save it? (y/n): ")
    if response.lower() != 'y':
        os.remove(model_path) 
        print("Model file deleted due to size constraints.")
    else:
        print(f"Model saved to {model_path} with size {file_size_mb:.2f} MB.")
else:
    print(f"Model saved to {model_path} with size {file_size_mb:.2f} MB.")

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
print(f"Optuna-Tuned RFC Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))
