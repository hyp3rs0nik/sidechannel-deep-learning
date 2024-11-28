# Deep Learning Model for Keystroke Prediction using Motion Sensors
## Overview
This project predicts keystrokes using motion sensor data (accelerometer and gyroscope). It employs deep learning models (CNN, LSTM, GRU) for time-series classification. Hyperparameter tuning is facilitated by Optuna, and cross-validation ensures model robustness.

## Features
- Input: Sensor data (accelerometer, gyroscope) and timestamps of keystrokes.
- Models: CNN, LSTM, GRU with customizable architecture.
- Optimization: Hyperparameter tuning using Optuna.
- Training: Cross-validation for reliable performance evaluation.
- Output: Saved trained model and label encoder.

## Requirements
- Python 3.8+
- Libraries: torch, numpy, pandas, scikit-learn, optuna, argparse.
Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
### Training
Train the model with:
```python main.py --data_dir ./data/training --model cnn --num_trials 50 --num_epochs 100 --num_trials 50```

## Testing
Evaluate the trained model on test data:
```python main.py --test --test_data_dir ./data/test --model cnn```

## Key Arguments
- --model: Selects the model type (cnn, lstm, gru).
- --data_dir: Directory for training data.
- --test_data_dir: Directory for test data.
- --num_trials: Number of Optuna trials for hyperparameter tuning.
- --num_epochs: Number of training epochs.
- --batch_size: Batch size for DataLoader.
## Data Preparation
- Input Files: .csv files containing sensor data and keystroke timestamps.
- Preprocessing: Time window extraction and interpolation for fixed timesteps.

## Models
Trained model: Saved as tuned_<model_type>_raw_sensor.pth in ./data/models.
Label encoder: Saved as label_encoder.pkl for consistent label mapping.
