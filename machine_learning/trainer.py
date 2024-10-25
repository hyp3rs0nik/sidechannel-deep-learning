import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_calculations import feature_functions

def extract_features_dynamically(data, selected_features):
    feature_data = {feature: feature_functions[feature](data) for feature in selected_features}
    features_df = pd.DataFrame(feature_data)
    features_df["key"] = data["key"]
    return features_df.dropna()

selected_features = [
    "rms", "rmse", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "magnitude_min", "magnitude_max", "sma",
    "displacement_2d"
]

accel_data_path = "./data/training/accel.csv" 
keys_data_path = "./data/training/keys.csv" 
accel_data = pd.read_csv(accel_data_path)
keys_data = pd.read_csv(keys_data_path)


accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"])
keys_data["timestamp"] = pd.to_datetime(keys_data["timestamp"])


merged_data = pd.merge_asof(
    accel_data.sort_values("timestamp"),
    keys_data.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("100ms")
).dropna(subset=["key"]) 

features_denoise_core = extract_features_dynamically(merged_data, selected_features)

X = features_denoise_core.drop(columns="key")
y = features_denoise_core["key"]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"),
                           param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)


grid_search.fit(X, y)


best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)


model_save_path = './data/models/machine_learning/best_random_forest_model.pkl'
joblib.dump(best_rf_model, model_save_path)
print(f"Model saved to: {model_save_path}")


importances = best_rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Feature Importances:\n", feature_importance_df)
