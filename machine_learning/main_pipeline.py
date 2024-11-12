import argparse
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


os.makedirs("./data/models", exist_ok=True)

NUM_TRIALS = 100
NUM_FOLDS = 5
WINDOW_SIZE = 8
EARLY_STOPPING_ROUNDS = 10

parser = argparse.ArgumentParser(description="Select the model architecture.")
parser.add_argument(
    "--model",
    type=str,
    default="rfc",
    choices=["rfc", "xgboost"],
    help="Model type to use: rfc or xgboost",
)
args = parser.parse_args()

keys_df = pd.read_csv("./data/training/keys.csv")
sensor_df = pd.read_csv("./data/training/sensor_v2_denoise_2.25hz.csv")

scaler = StandardScaler()
sensor_features = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
sensor_df[sensor_features] = scaler.fit_transform(sensor_df[sensor_features])

key_coordinates = {
    "1": (0, 0), "2": (0, 1), "3": (0, 2),
    "4": (1, 0), "5": (1, 1), "6": (1, 2),
    "7": (2, 0), "8": (2, 1), "9": (2, 2),
    "0": (2.5, 1), "Enter": (2.5, 3),
}

def compute_displacement_vector(key1, key2):
    if key1 in key_coordinates and key2 in key_coordinates:
        x1, y1 = key_coordinates[key1]
        x2, y2 = key_coordinates[key2]
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        return None, None

keys_df = keys_df[keys_df["key"] != "Backspace"]
displacement_vectors = []
previous_key = None

for key in keys_df["key"]:
    if previous_key is not None:
        dx, dy = compute_displacement_vector(previous_key, key)
        if dx is not None and dy is not None:
            displacement_vectors.append((dx, dy))
    previous_key = key


sensor_df["accel_magnitude"] = np.sqrt(
    sensor_df["accel_x"] ** 2 + sensor_df["accel_y"] ** 2 + sensor_df["accel_z"] ** 2
)
sensor_df["gyro_magnitude"] = np.sqrt(
    sensor_df["gyro_x"] ** 2 + sensor_df["gyro_y"] ** 2 + sensor_df["gyro_z"] ** 2
)

sensor_sequences = []
for i in range(len(displacement_vectors)):
    start_idx = i * WINDOW_SIZE
    end_idx = start_idx + WINDOW_SIZE
    if end_idx <= len(sensor_df):
        sensor_window = sensor_df.iloc[start_idx:end_idx][
            sensor_features + ["accel_magnitude", "gyro_magnitude"]
        ].values.flatten()  
        sensor_sequences.append(sensor_window)


sensor_sequences = np.array(sensor_sequences)
displacement_vectors = np.array(displacement_vectors)

X = np.hstack([sensor_sequences, displacement_vectors])

def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10
    else:
        raise ValueError(f"Unexpected key: {key}")

y = np.array([key_to_label(key) for key in keys_df["key"][:len(displacement_vectors)]])

def get_model(model_type, trial):
    if model_type == "rfc":
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "xgboost":
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)
        max_depth = trial.suggest_int("max_depth", 2, 15)
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        return XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            eval_metric="mlogloss",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
    else:
        raise ValueError("Invalid model type selected.")

def objective(trial):
    model = get_model(args.model, trial)
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    accuracies = []

    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if args.model == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

def main():
    study = optuna.create_study(direction="maximize", study_name=f"ml_hyperparameter_tuning_{args.model}", storage="sqlite:///data/db.sqlite", load_if_exists=True)
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print(f"Best trial parameters: {study.best_trial.params}")
    print(f"Best trial accuracy: {study.best_value}")
    
    
    best_model = get_model(args.model, study.best_trial)
    best_model.fit(X, y)
    model_path = f"./data/models/{args.model}_best_model.pkl"
    with open(model_path, "wb") as f:
        import pickle
        pickle.dump(best_model, f)
    print(f"Best model saved to {model_path}")

main()
