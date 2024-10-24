import os
import joblib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(accel_path, keys_path):
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)

    accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"])
    keys_data["timestamp"] = pd.to_datetime(keys_data["timestamp"])

    cleaned_data = accel_data.groupby("timestamp", as_index=False).agg(
        {"x": "mean", "y": "mean", "z": "mean"}
    )

    cleaned_data["rms"] = np.sqrt((cleaned_data[["x", "y", "z"]] ** 2).sum(axis=1))

    merged_data = pd.merge_asof(
        cleaned_data,
        keys_data,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )

    return merged_data


def extract_features(cleaned_data):
    rolling_window = 5

    # RMS cross rate
    cleaned_data["rms_cross_rate"] = (
        (cleaned_data["rms"].shift(1) - cleaned_data["rms"].mean())
        * (cleaned_data["rms"] - cleaned_data["rms"].mean())
        < 0
    ).astype(int)

    # Root Mean Square Error (RMSE) over a window
    cleaned_data["rmse"] = (
        cleaned_data["rms"].rolling(window=rolling_window, min_periods=1).mean()
        - cleaned_data["rms"]
    ) ** 2
    cleaned_data["rmse"] = (
        cleaned_data["rmse"].rolling(window=rolling_window, min_periods=1).mean() ** 0.5
    )

    # Extract dimensional features for each axis
    for axis in ["x", "y", "z"]:
        cleaned_data[f"{axis}_min"] = (
            cleaned_data[axis].rolling(window=rolling_window, min_periods=1).min()
        )
        cleaned_data[f"{axis}_max"] = (
            cleaned_data[axis].rolling(window=rolling_window, min_periods=1).max()
        )
        cleaned_data[f"{axis}_num_peaks"] = (
            cleaned_data[axis]
            .rolling(window=rolling_window, min_periods=1)
            .apply(lambda x: len(find_peaks(x)[0]), raw=True)
        )
        cleaned_data[f"{axis}_num_crests"] = (
            cleaned_data[axis]
            .rolling(window=rolling_window, min_periods=1)
            .apply(lambda x: len(find_peaks(-x)[0]), raw=True)
        )

    # Euclidean magnitude features
    cleaned_data["magnitude"] = np.sqrt(
        (cleaned_data[["x", "y", "z"]] ** 2).sum(axis=1)
    )
    for feature in ["min", "max", "num_peaks", "num_crests"]:
        cleaned_data[f"magnitude_{feature}"] = (
            cleaned_data["magnitude"]
            .rolling(window=rolling_window, min_periods=1)
            .agg(
                feature
                if feature in ["min", "max"]
                else lambda x: len(find_peaks(x if feature == "num_peaks" else -x)[0])
            )
        )

    # Signal Magnitude Area (SMA)
    cleaned_data["sma"] = (
        cleaned_data[["x", "y", "z"]]
        .abs()
        .sum(axis=1)
        .rolling(window=rolling_window, min_periods=1)
        .sum()
    )

    return cleaned_data


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_rfc = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    best_rfc.fit(X_train, y_train)
    y_pred = best_rfc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return best_rfc, accuracy, report


def save_model(model, model_name, path="./data/models/machine_learning"):
    model_path = os.path.join(path, f"{model_name}.pkl")
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    accel_data_path = "./data/training/accel.csv"
    keys_data_path = "./data/training/keys.csv"

    data = load_and_preprocess_data(accel_data_path, keys_data_path)
    features = extract_features(data)

    X = features[
        [
            "rms",
            "rmse",
            "rms_cross_rate",
            "x_min",
            "x_max",
            "x_num_peaks",
            "x_num_crests",
            "y_min",
            "y_max",
            "y_num_peaks",
            "y_num_crests",
            "z_min",
            "z_max",
            "z_num_peaks",
            "z_num_crests",
            "magnitude_min",
            "magnitude_max",
            "magnitude_num_peaks",
            "magnitude_num_crests",
            "sma",
        ]
    ].fillna(0)

    y = data["key"].dropna()
    X = X.loc[y.index]

    model, accuracy, report = train_and_evaluate(X, y)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    save_model(model, model_name="random_forest_best_model")
