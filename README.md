
# ML/DL Numeric Keystroke Inference

This project focuses on inferring numeric keystrokes using data captured from a keylogger and smartwatch sensors. **Machine learning**, specifically a **Random Forest Classifier**, is used to analyze the sensor data, with the goal of identifying specific numeric keystrokes made by the user. The project explores how smartwatch accelerometer data can be used to infer which numeric keys are pressed, providing insights into potential vulnerabilities related to keystroke detection through wearable devices.

## Inference Process

The inference process involves capturing raw data from two sources:

1. **Keylogger (capture-key-app)**: Records keypresses on the user’s device and associates them with precise timestamps.
2. **Smartwatch (capture-sensor-app)**: The smartwatch captures movement data from its accelerometer sensors, which are timestamped and aligned with the captured keypress data.

### Data Processing and Feature Engineering

Key steps in the process:
- **Feature Extraction**: Important features such as Root Mean Square (RMS), RMS cross rate, Signal Magnitude Area (SMA), and Euclidean magnitude are derived from the accelerometer data to capture variations related to keystrokes.
- **Model Training**: The processed data is fed into a **Random Forest Classifier**, which uses hyperparameter tuning via **GridSearchCV** to optimize the model’s performance.
- **Evaluation**: The model is evaluated using accuracy and F1-score metrics to measure its effectiveness in identifying keypresses.

### Machine Learning Model

The current approach uses a **Random Forest Classifier** with hyperparameter tuning. The trained model is saved in:
```
./data/models/machine_learning/{model_name}.pkl
```
This allows for easy retrieval and deployment of the model for future predictions.

### Limitations

This experiment was conducted with a relatively small dataset, which limits its generalizability. Additionally:
- **Sensor Noise**: Inaccuracies or noise in the accelerometer readings may affect data quality.
- **Typing Style Variation**: Differences in individual typing styles can introduce variance in sensor readings.
- **Smartwatch Positioning**: Variations in how the smartwatch is worn can impact accuracy.

As a result, **the current model serves as a proof of concept** and may not perform optimally under all conditions or with diverse users.

## Setup and Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sidechannel-keys-sensor-main.git
   cd sidechannel-keys-sensor
   ```

2. Install the necessary Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Install the necessary dependencies for the API server (assuming Node.js is installed):
   ```sh
   cd ./capture-data-api
   npm install
   node server/js
   ```

## Usage

To train the machine learning model with the processed data:
```sh
python ./machine_learning/trainer.py
```

After training, the model is automatically saved to:
```
./data/models/machine_learning/{model_name}.pkl
```

## Future Work

- **Dataset Expansion**: Expand the dataset to improve model robustness and generalizability.
- **Algorithm Improvements**: Experiment with other machine learning algorithms or deep learning models.
- **Real-time Prediction**: Implement real-time keypress prediction using the trained model.
