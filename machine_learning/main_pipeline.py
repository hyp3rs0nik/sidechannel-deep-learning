import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# File paths
SENSOR_FILE = "./data/training/v3.sensors.csv"
KEYSTROKES_FILE = "./data/training/v3.keystrokes.csv"

# Load datasets
sensors_df = pd.read_csv(SENSOR_FILE)
keystrokes_df = pd.read_csv(KEYSTROKES_FILE)

# Ensure timestamps are sorted
sensors_df = sensors_df.sort_values(by="timestamp").reset_index(drop=True)
keystrokes_df = keystrokes_df.sort_values(by="timestamp").reset_index(drop=True)

# Parameters for sliding window
WINDOW_SIZE = 50
STEP_SIZE = 10
SENSOR_COLUMNS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


# Features to include
features_to_include = [
    "mean",
    # "std",
    # "median",
    # "skewness",
    # "kurtosis",
    "rms",
    "dominant_frequency",
    "total_spectral_energy",
    # "displacement_2d",
    # "displacement_3d",
    # "keystroke_rate",
    # "hold_time",
    # "flight_time",
    # "inter_key_interval",
]


# Feature extraction function
def extract_features(window, keystroke_window, features_to_include):
    features = []

    for col in SENSOR_COLUMNS:
        data = window[col].values

        # Statistical features
        if "mean" in features_to_include:
            features.append(np.mean(data))
        if "std" in features_to_include:
            features.append(np.std(data))
        if "median" in features_to_include:
            features.append(np.median(data))
        if "skewness" in features_to_include:
            features.append(skew(data))
        if "kurtosis" in features_to_include:
            features.append(kurtosis(data))

        # RMS feature
        if "rms" in features_to_include:
            features.append(np.sqrt(np.mean(data**2)))

        # Frequency-domain features
        if "dominant_frequency" in features_to_include:
            fft_coeffs = fft(data)
            fft_magnitude = np.abs(fft_coeffs)
            features.append(np.max(fft_magnitude))
        if "total_spectral_energy" in features_to_include:
            fft_coeffs = fft(data)
            fft_magnitude = np.abs(fft_coeffs)
            features.append(np.sum(fft_magnitude))

    # 2D and 3D displacement features
    if (
        "displacement_2d" in features_to_include
        or "displacement_3d" in features_to_include
    ):
        accel_x = window["accel_x"].values
        accel_y = window["accel_y"].values
        accel_z = window["accel_z"].values

        if "displacement_2d" in features_to_include:
            displacement_2d = np.sqrt((np.diff(accel_x) ** 2) + (np.diff(accel_y) ** 2))
            features.append(np.sum(displacement_2d))

        if "displacement_3d" in features_to_include:
            displacement_3d = np.sqrt(
                (np.diff(accel_x) ** 2)
                + (np.diff(accel_y) ** 2)
                + (np.diff(accel_z) ** 2)
            )
            features.append(np.sum(displacement_3d))

    # Time-domain features based on keystrokes
    if "keystroke_rate" in features_to_include:
        keystroke_count = len(keystroke_window)
        time_span = window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]
        features.append(keystroke_count / time_span if time_span > 0 else 0)

    if "hold_time" in features_to_include:
        hold_times = keystroke_window["timestamp"].diff().dropna().values
        features.append(np.mean(hold_times) if len(hold_times) > 0 else 0)

    if "flight_time" in features_to_include:
        flight_times = keystroke_window["timestamp"].diff(periods=2).dropna().values
        features.append(np.mean(flight_times) if len(flight_times) > 0 else 0)

    if "inter_key_interval" in features_to_include:
        inter_key_intervals = keystroke_window["timestamp"].diff().dropna().values
        features.append(
            np.mean(inter_key_intervals) if len(inter_key_intervals) > 0 else 0
        )

    return features

# Feature and label extraction
X, y = [], []

for start_idx in range(0, len(sensors_df) - WINDOW_SIZE + 1, STEP_SIZE):
    end_idx = start_idx + WINDOW_SIZE
    window = sensors_df.iloc[start_idx:end_idx]

    # Find corresponding keystrokes
    keystroke_window = keystrokes_df[
        (keystrokes_df["timestamp"] >= window["timestamp"].iloc[0])
        & (keystrokes_df["timestamp"] <= window["timestamp"].iloc[-1])
    ]

    # Assign the label (key pressed at midpoint)
    label = keystroke_window["key"].values[-1] if not keystroke_window.empty else None
    if label is not None:
        features = extract_features(window, keystroke_window, features_to_include)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="poly", random_state=42),
}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

# Feature importance ranking for Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Extract feature importances
feature_importances = random_forest.feature_importances_

# Create a mapping of feature names to their importances
feature_names = []
for col in SENSOR_COLUMNS:
    if "mean" in features_to_include:
        feature_names.append(f"{col}_mean")
    if "std" in features_to_include:
        feature_names.append(f"{col}_std")
    if "median" in features_to_include:
        feature_names.append(f"{col}_median")
    if "skewness" in features_to_include:
        feature_names.append(f"{col}_skewness")
    if "kurtosis" in features_to_include:
        feature_names.append(f"{col}_kurtosis")
    if "rms" in features_to_include:
        feature_names.append(f"{col}_rms")
    if "dominant_frequency" in features_to_include:
        feature_names.append(f"{col}_dominant_frequency")
    if "total_spectral_energy" in features_to_include:
        feature_names.append(f"{col}_total_spectral_energy")

# Add names for global features
if "displacement_2d" in features_to_include:
    feature_names.append("displacement_2d")
if "displacement_3d" in features_to_include:
    feature_names.append("displacement_3d")
if "keystroke_rate" in features_to_include:
    feature_names.append("keystroke_rate")
if "hold_time" in features_to_include:
    feature_names.append("hold_time")
if "flight_time" in features_to_include:
    feature_names.append("flight_time")
if "inter_key_interval" in features_to_include:
    feature_names.append("inter_key_interval")


# Combine feature names and their importance scores
feature_ranking = sorted(
    zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
)

# Display the ranked features
print("\nFeature Ranking:")
for feature, importance in feature_ranking:
    print(f"{feature}: {importance:.4f}")
