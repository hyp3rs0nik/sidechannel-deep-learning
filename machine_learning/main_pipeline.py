import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import DMatrix, train
from xgboost import callback
import optuna
import pickle
import os
from datetime import timedelta


parser = argparse.ArgumentParser(description="Select the model architecture.")
parser.add_argument(
    "--model",
    type=str,
    default="xgboost",
    choices=["rfc", "xgboost"],
    help="Model type to use: rfc or xgboost",
)
args = parser.parse_args()


keys_df = pd.read_csv("./data/training/keys.csv")
sensor_df = pd.read_csv("./data/training/sensor_v2_denoise_2.25hz.csv")


keys_df['timestamp'] = pd.to_datetime(keys_df['timestamp'])
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])


valid_keys_df = keys_df[keys_df["key"] != "Backspace"].reset_index(drop=True)


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


displacement_vectors = []
previous_key = None
for key in valid_keys_df["key"]:
    if previous_key is not None:
        dx, dy = compute_displacement_vector(previous_key, key)
        if dx is not None and dy is not None:
            displacement_vectors.append((dx, dy))
    previous_key = key
displacement_vectors = np.array(displacement_vectors)

WINDOW_SIZE = 8  
STEP_SIZE = 4    


sliding_window_sequences = []
sliding_window_labels = []
sliding_window_displacements = []  

def key_to_label(key):
    if key.isdigit():
        return int(key)
    elif key == "Enter":
        return 10  
    else:
        raise ValueError(f"Unexpected key: {key}")


for i, (key, key_time) in enumerate(zip(valid_keys_df["key"], valid_keys_df["timestamp"])):
    sensor_window = sensor_df[
        (sensor_df['timestamp'] >= key_time - timedelta(milliseconds=150)) &
        (sensor_df['timestamp'] <= key_time + timedelta(milliseconds=150))
    ]
    
    if len(sensor_window) >= WINDOW_SIZE:
        sensor_values = sensor_window[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values

        for start in range(0, len(sensor_values) - WINDOW_SIZE + 1, STEP_SIZE):
            window = sensor_values[start:start + WINDOW_SIZE].flatten()
            sliding_window_sequences.append(window)
            sliding_window_labels.append(key_to_label(key))
            sliding_window_displacements.append(displacement_vectors[i])

X_sensor = np.array(sliding_window_sequences, dtype=np.float32)
X_displacement = np.array(sliding_window_displacements, dtype=np.float32)
y = np.array(sliding_window_labels)


X = np.hstack([X_sensor, X_displacement])


extracted_features_df = pd.DataFrame(X)
extracted_features_df["label"] = y
extracted_features_path = "./data/training/extracted_features.csv"
extracted_features_df.to_csv(extracted_features_path, index=False)
print(f"Extracted features saved to {extracted_features_path}")


X = X[y < 10]
y = y[y < 10]


def get_model(trial):
    if args.model == "rfc":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            random_state=42
        )
    elif args.model == "xgboost":
        return {
            "objective": "multi:softmax",
            "num_class": 10,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "eval_metric": "mlogloss",
            "seed": 42,
        }
    else:
        raise ValueError("Invalid model type selected.")

def objective(trial):
    if args.model == "rfc":
        model = get_model(trial)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        return avg_accuracy

    elif args.model == "xgboost":
        params = get_model(trial)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = DMatrix(X_train, label=y_train)
            dval = DMatrix(X_val, label=y_val)
            early_stop = callback.EarlyStopping(rounds=10, save_best=True, maximize=False)
            evals = [(dval, 'validation')]

            model = train(params, dtrain, num_boost_round=1000, evals=evals, callbacks=[early_stop], verbose_eval=False)
            y_pred = model.predict(dval)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        return avg_accuracy


storage_path = f"sqlite:///data/{args.model}_db.sqlite"
study_name = f"ml_hyperparameter_tuning_{args.model}"
study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_path, load_if_exists=True)
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_trial.params
best_accuracy = study.best_value
print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_accuracy}")

if args.model == "rfc":
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X, y)
else:
    dtrain_full = DMatrix(X, label=y)
    best_model = train(best_params, dtrain_full, num_boost_round=1000, verbose_eval=False)

model_path = f"./data/models/{args.model}_best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {model_path}")

if os.path.exists(f"./data/{args.model}_db.sqlite"):
    os.remove(f"./data/{args.model}_db.sqlite")
    print(f"Study database file for {args.model} deleted after training.")

