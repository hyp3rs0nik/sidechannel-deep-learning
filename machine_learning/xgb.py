import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, learning_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import optuna
import joblib

# Paths
MODEL_PATH = './data/models/machine_learning'
PARAMS_PATH = './docs/best_param_xgboost.json'
OPTUNA_DB_PATH = 'sqlite:///optuna_xgboost_study.db' 

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

def load_data():
    keys_df = pd.read_csv('./data/training/keys.csv')
    sensors_df = pd.read_csv('./data/training/sensors.csv')
    return keys_df, sensors_df

def align_data(keys_df, sensors_df, pre_window_ms=100, post_window_ms=175):
    keys_df['timestamp'] = keys_df['timestamp'].astype(int)
    sensors_df['timestamp'] = sensors_df['timestamp'].astype(int)
    keys_df.sort_values('timestamp', inplace=True)
    sensors_df.sort_values('timestamp', inplace=True)

    aligned_data = []

    for _, key_row in keys_df.iterrows():
        key_time = key_row['timestamp']
        relevant_sensors = sensors_df[
            (sensors_df['timestamp'] >= key_time - pre_window_ms) &
            (sensors_df['timestamp'] <= key_time + post_window_ms)
        ]
        
        if not relevant_sensors.empty:
            for _, sensor_row in relevant_sensors.iterrows():
                aligned_entry = sensor_row.to_dict()
                aligned_entry['key'] = key_row['key']
                aligned_data.append(aligned_entry)

    aligned_df = pd.DataFrame(aligned_data)
    
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

def plot_learning_curve(model, X, y, save_path='./docs/xgboost_learning_curve.png'):
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
        n_estimators = trial.suggest_int('n_estimators', 200, 500)
        max_depth = trial.suggest_int('max_depth', 5, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        min_child_weight = trial.suggest_int('min_child_weight', 5, 15)
        subsample = trial.suggest_float('subsample', 0.5, 0.8, step=0.05)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.05)
        gamma = trial.suggest_float('gamma', 0, 5, step=0.5)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1, step=0.1)
        reg_lambda = trial.suggest_float('reg_lambda', 1, 10, step=1)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric='logloss',
            random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        return cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy').mean()

    study = optuna.create_study(
        study_name="xgboost_hyperparameter_tuning",
        storage=OPTUNA_DB_PATH,
        direction='maximize',
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=200, n_jobs=-1)
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

    # Split data before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Apply scaling after split
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Optimize hyperparameters using Optuna
    study = create_optuna_study(X_train_scaled, y_train)

    best_params = study.best_params
    model = XGBClassifier(**best_params, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)

    test_score = model.score(X_test_scaled, y_test)
    print("Test score on hold-out set:", test_score)

    plot_learning_curve(model, X_train_scaled, y_train)

    # Save model, scaler, and label encoder
    joblib.dump(model, f"{MODEL_PATH}/xgboost_model.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_PATH}/label_encoder.pkl")
    with open(PARAMS_PATH, 'w') as f:
        json.dump(best_params, f)

    print(f"Model, scaler, label encoder saved to {MODEL_PATH}")
    print(f"Best parameters saved to {PARAMS_PATH}")

    # # Load collected data for testing

    # print("Loading collected data...")

    # collected_keys_df = pd.read_csv('./data/collected/keys.csv')
    # collected_sensors_df = pd.read_csv('./data/collected/sensors.csv')
    # collected_aligned_df = align_data(collected_keys_df, collected_sensors_df)
    # print(f'Aligned data shape: {collected_aligned_df.shape}, columns: {collected_aligned_df.columns}')
    # collected_features_df = extract_features(collected_aligned_df)
    # print(f'Extracted features shape: {collected_features_df.shape}')

    # # Prepare collected features for prediction
    # X_collected = collected_features_df.drop(columns=['key'])
    # X_collected_scaled = pd.DataFrame(scaler.transform(X_collected), columns=X.columns)

    # # Predict using the trained model
    # y_collected_pred = model.predict(X_collected_scaled)
    # y_collected_pred_decoded = label_encoder.inverse_transform(y_collected_pred)

    # # Save predictions
    # collected_features_df['predicted_key'] = y_collected_pred_decoded
    # collected_features_df.to_csv('./data/collected/predictions.csv', index=False)
    # print("Predictions saved to ./data/collected/predictions.csv")

    return model, best_params, test_score

if __name__ == '__main__':
    model, best_params, test_score = main()
    print("Model:", model)
    print("Test score:", test_score)
    os.remove('optuna_xgboost_study.db')
