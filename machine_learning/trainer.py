import os
import joblib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(accel_path, keys_path):
    accel_data = pd.read_csv(accel_path)
    keys_data = pd.read_csv(keys_path)  # Corrected line

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
        tolerance=pd.Timedelta("200ms"),
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


def randomized_search_tuning(X_train, y_train, model_type="RandomForest"):
    if model_type == "RandomForest":
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == "XGBoost":
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
        model = XGBClassifier(random_state=42, eval_metric='mlogloss')

    rand_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rand_search.fit(X_train, y_train)
    print(f"Best Parameters for {model_type}: {rand_search.best_params_}")
    return rand_search.best_estimator_


def train_and_evaluate_stacking(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest and XGBoost with hyperparameter tuning
    rf_model = randomized_search_tuning(X_train, y_train, model_type="RandomForest")
    xgb_model = randomized_search_tuning(X_train, y_train, model_type="XGBoost")

    # Meta features (predictions from both models)
    rf_pred_train = rf_model.predict_proba(X_train)
    xgb_pred_train = xgb_model.predict_proba(X_train)
    
    rf_pred_test = rf_model.predict_proba(X_test)
    xgb_pred_test = xgb_model.predict_proba(X_test)

    # Combine predictions from both models as meta-features
    meta_train = np.hstack((rf_pred_train, xgb_pred_train))
    meta_test = np.hstack((rf_pred_test, xgb_pred_test))

    # Train a Logistic Regression meta-model
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_train, y_train)

    # Meta-model predictions
    y_pred = meta_model.predict(meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Stacking Model Accuracy: {accuracy}")
    print(f"Stacking Model Classification Report:\n{report}")

    return meta_model


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

    # Train and evaluate stacking model
    meta_model = train_and_evaluate_stacking(X, y)

    # Save the stacking model
    save_model(meta_model, model_name="stacking_model")
