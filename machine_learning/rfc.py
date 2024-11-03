import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
import joblib

# Paths
MODEL_PATH = './data/models/machine_learning'
PARAMS_PATH = './docs/best_param_rfc.json'
OPTUNA_DB_PATH = 'sqlite:///optuna_rfc_study.db' 

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

def load_data():
    keys_df = pd.read_csv('./data/training/keys.csv')
    sensors_df = pd.read_csv('./data/training/sensors.csv')
    return keys_df, sensors_df

def align_data(keys_df, sensors_df, window_ms=275):
    keys_df['timestamp'] = keys_df['timestamp'].astype(int)
    sensors_df['timestamp'] = sensors_df['timestamp'].astype(int)
    keys_df.sort_values('timestamp', inplace=True)
    sensors_df.sort_values('timestamp', inplace=True)

    aligned_df = pd.merge_asof(
        sensors_df, keys_df, on='timestamp', direction='nearest', tolerance=window_ms
    )
    aligned_df.dropna(subset=['key'], inplace=True)
    return aligned_df

def extract_features(aligned_df):
    features_list = []
    for _, row in aligned_df.iterrows():
        features = {
            'key': row['key'],
            'accel_x': row['accel_x'],
            'accel_y': row['accel_y'],
            'accel_z': row['accel_z'],
            'gyro_x': row['gyro_x'],
            'gyro_y': row['gyro_y'],
            'gyro_z': row['gyro_z'],
            'rotation_x': row['rotation_x'],
            'rotation_y': row['rotation_y'],
            'rotation_z': row['rotation_z'],
            'rotation_w': row['rotation_w'],
        }
        features_list.append(features)
    return pd.DataFrame(features_list)

def plot_learning_curve(model, X, y, save_path='./docs/rfc_learning_curve.png'):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, val_mean, label='Validation Score')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()


def create_optuna_study(X_train, y_train):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 250, 700) 
        max_depth = trial.suggest_int('max_depth', 12, 18)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 15)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        
        return cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=3), scoring='accuracy').mean()

    study = optuna.create_study(
        study_name="rf_hyperparameter_tuning",
        storage=OPTUNA_DB_PATH,
        direction='maximize',
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=100, n_jobs=1)
    return study

def main():
    keys_df, sensors_df = load_data()
    aligned_df = align_data(keys_df, sensors_df)
    features_df = extract_features(aligned_df)
    features_df.to_csv(f'{MODEL_PATH}/features.csv', index=False)

    X = features_df.drop(columns=['key'])
    y = features_df['key']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    study = create_optuna_study(X_train, y_train)

    best_params = study.best_params
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    test_score = model.score(X_test, y_test)
    print("Test score on hold-out set:", test_score)

    plot_learning_curve(model, X_train, y_train)

    joblib.dump(model, f"{MODEL_PATH}/random_forest_model.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_PATH}/label_encoder.pkl")
    with open(PARAMS_PATH, 'w') as f:
        json.dump(best_params, f)

    print(f"Model, scaler, label encoder saved to {MODEL_PATH}")
    print(f"Best parameters saved to {PARAMS_PATH}")

    return model, best_params, test_score

if __name__ == '__main__':
    model, best_params, test_score = main()
    print("Model:", model)
    print("Test score:", test_score)
