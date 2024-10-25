import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define individual calculation functions
def calculate_rms(data):
    return np.sqrt((data[["x", "y", "z"]] ** 2).mean(axis=1))

def calculate_rmse(data, window=5):
    rms = calculate_rms(data)
    return rms.rolling(window=window, min_periods=1).apply(
        lambda x: np.sqrt(np.mean((x - x.mean()) ** 2)), raw=False
    )

def calculate_min(data, axis):
    return data[axis].rolling(window=5, min_periods=1).min()

def calculate_max(data, axis):
    return data[axis].rolling(window=5, min_periods=1).max()

def calculate_magnitude(data):
    return np.sqrt((data[["x", "y", "z"]] ** 2).sum(axis=1))

def calculate_sma(data, window=5):
    return data[["x", "y", "z"]].abs().sum(axis=1).rolling(window=window, min_periods=1).sum()

def calculate_displacement_2d(data):
    return np.sqrt(
        cumulative_trapezoid(data["x"], initial=0)**2 + cumulative_trapezoid(data["y"], initial=0)**2
    )

# Map feature names to calculation functions
feature_functions = {
    "rms": calculate_rms,
    "rmse": lambda data: calculate_rmse(data),
    "x_min": lambda data: calculate_min(data, "x"),
    "x_max": lambda data: calculate_max(data, "x"),
    "y_min": lambda data: calculate_min(data, "y"),
    "y_max": lambda data: calculate_max(data, "y"),
    "z_min": lambda data: calculate_min(data, "z"),
    "z_max": lambda data: calculate_max(data, "z"),
    "magnitude_min": lambda data: calculate_min(data.assign(magnitude=calculate_magnitude(data)), "magnitude"),
    "magnitude_max": lambda data: calculate_max(data.assign(magnitude=calculate_magnitude(data)), "magnitude"),
    "sma": calculate_sma,
    "displacement_2d": calculate_displacement_2d
}

# Feature extraction function
def extract_features_dynamically(data, selected_features):
    feature_data = {feature: feature_functions[feature](data) for feature in selected_features}
    features_df = pd.DataFrame(feature_data)
    features_df["key"] = data["key"]
    return features_df.dropna()

# Example selected features
selected_features = [
    "rms", "rmse", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "magnitude_min", "magnitude_max", "sma",
    "displacement_2d"
]

# Load data and merge with keys to include the "key" column
accel_data_path = "./data/training/accel.csv"  # Replace with actual path
keys_data_path = "./data/training/keys.csv"    # Replace with actual path
accel_data = pd.read_csv(accel_data_path)
keys_data = pd.read_csv(keys_data_path)

# Convert timestamps to datetime and merge datasets
accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"])
keys_data["timestamp"] = pd.to_datetime(keys_data["timestamp"])

# Merge accel and key data
merged_data = pd.merge_asof(
    accel_data.sort_values("timestamp"),
    keys_data.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("100ms")
).dropna(subset=["key"])  # Drop rows without key data

# Extract features from the merged dataset
features_denoise_core = extract_features_dynamically(merged_data, selected_features)

# Split into features and labels
X = features_denoise_core.drop(columns="key")
y = features_denoise_core["key"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=50)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Feature Importance Evaluation
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Feature Importances:\n", feature_importance_df)
