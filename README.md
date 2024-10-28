# ML/DL Numeric Keystroke Inference

This project focuses on inferring numeric keystrokes using data captured from a keylogger and smartwatch sensors. The solution incorporates **Machine Learning** using a **Random Forest Classifier** and **Deep Learning** models (LSTM/GRU) to analyze accelerometer data. The goal is to identify specific numeric keystrokes made by the user. This project highlights potential vulnerabilities of keystroke detection through wearable devices, showcasing how accelerometer data can infer keystrokes.

---

## Inference Process

The inference process involves capturing data from two sources:

1. **Keylogger (capture-key-app)**: Records keypresses on the user’s device with precise timestamps.
2. **Smartwatch (capture-sensor-app)**: Captures accelerometer data, timestamped and aligned with the keypress data.

---

## Data Processing and Feature Engineering

Key steps in the process include:

- **Data Segmentation**: Refined segmentation logic aligns accelerometer data with keypress events using timestamps.
- **FFT-Based Denoising**: Applies FFT to filter out high-frequency noise from accelerometer data.
- **Feature Extraction**: Extracts critical features like RMS, SMA, min/max axis values, and magnitude variations.
- **Dynamic Feature Selection**: Additional features are selected and applied based on configuration.

---

## Machine Learning Model

The **Random Forest Classifier** is tuned using **Optuna** for hyperparameter optimization. 

- **Model Training**: The segmented and processed data is fed into the classifier for training.
- **Cross-Validation**: Stratified K-Fold validation is used for reliable model evaluation.
- **Performance Metrics**: Model performance is measured using accuracy and classification reports.

The model is saved to: `./data/models/machine_learning/rfc.pkl`


---

## Deep Learning Model

For advanced analysis, **LSTM/GRU models** are trained using PyTorch, with hyperparameter tuning via Optuna. 

- **Data Preparation**: Tensor-based sequence generation ensures the LSTM/GRU models can capture temporal dependencies.
- **Training Process**: Mixed precision training and gradient clipping are used for stability.
- **Early Stopping**: Monitors validation loss to avoid overfitting.
  
The deep learning models are saved to: `./data/models/deep_learning/{model_name}.pt`


## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sidechannel-keys-sensor-main.git
   cd sidechannel-keys-sensor
   ```
2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage
**Train the Machine Learning model**:
```bash
python ./machine_learning/trainer.py
```
**Train the Deep Learning model (LSTM/GRU)**:
```bash
python ./trainer.py --model lstm --dataset ./data/training/
```

## Limitations

- **Small Dataset**: Results may vary due to limited data.
- **Sensor Noise**: Noisy sensor data may reduce accuracy.
- **Typing Style & Watch Positioning**: Individual typing styles and variations in watch placement may affect model performance.

## Future Work
- Dataset Expansion: Collect more data for improved generalizability.
- Algorithm Improvements: Experiment with CNNs or hybrid models.
- Real-time Prediction: Implement real-time prediction capabilities.




