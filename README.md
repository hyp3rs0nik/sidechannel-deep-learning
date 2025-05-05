# Side-Channel Keystroke Inference via Acoustic & Motion Sensor Data

A research prototype and end-to-end implementation for inferring keyboard input smartwatch motion sensors. This repository contains data-capture tools, preprocessing pipelines, machine-learning baselines, and deep-learning models, alongside scripts for hyperparameter tuning and evaluation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Contributions](#key-contributions)
3. [Repository Structure](#repository-structure)
4. [Data Capture & Sync](#data-capture--sync)
5. [Preprocessing](#preprocessing)
6. [Machine-Learning Pipeline](#machine-learning-pipeline)
7. [Deep-Learning Pipeline](#deep-learning-pipeline)
8. [Evaluation & Results](#evaluation--results)
9. [Installation & Usage](#installation--usage)

---

## Project Overview

* **Motion**: infer keystrokes from accelerometer & gyroscope data streamed by a smartwatch.

Our goal was to evaluate each method’s accuracy and real-world feasibility.

---

## Key Contributions

* **Full-stack capture tools**:

  * **capture-key-app** (HTML/JS) generates balanced 5-digit sequences.
  * **capture-sensor-app** (Kotlin) streams 250 Hz motion data with server time-sync.
  * **capture-data-api** (Node.js + MongoDB) handles high-frequency sensor & keylog ingestion with sub-20 ms latency.

* **Preprocessing & feature engineering**:

  * Windowed alignment (±180 ms) and FFT-based noise filtering for motion sensors.
  * Statistical & frequency-domain features for sensor data.

* **Modeling**:
  * **Classical ML**: Random Forest, SVM, Logistic Regression with Optuna-tuned hyperparameters.
  * **Deep Learning**: CNN, LSTM, GRU on raw sensor windows; five-fold stratified cross-validation.

* **Evaluation**:
  * Sensor-based CNN achieved 70 % accuracy on entirely new user data.

---

## Repository Structure

```
├── capture-key-app/         # JS app for PIN sequence generation  
├── capture-sensor-app/      # Android/Kotlin smartwatch data streamer  
├── capture-data-api/        # Node.js + MongoDB ingestion server  
├── data/                    # Raw & processed CSV datasets  
├── machine_learning/        # Feature extraction + classical ML experiments  
├── deep_learning/           # PyTorch models, training & evaluation scripts  
├── docs/                    # Protocol diagrams, sample waveforms  
├── README.md                # This document  
├── requirements.txt         # Python dependencies  
└── package.json             # API server dependencies  
```

---

## Data Capture & Sync

1. **Balanced sequences**: 0–9 digits, 10 examples per class, grouped in 5-digit pools.
2. **Timestamp alignment**: server round-trip time adjustment ensures <1 ms drift.
3. **Batch & compress**: motion data buffered at 250 ms intervals, then gzipped for throughput.

---

## Preprocessing

* **Motion**:

  * Window centered on keypress timestamp (±180 ms).
  * FFT filter to suppress unrelated hand movements.
  * Time-domain (mean, std, RMS) and frequency-domain (FFT coefficients, spectral energy) features.

---

## Machine-Learning Pipeline

1. **Feature sets** for sensor data → sklearn pipelines.
2. **Optuna** for hyperparameter search (e.g., number of trees, max depth).
3. **5-fold stratified CV** to assess generalization.
4. **Best classical model**: SVM at 51 % accuracy on unseen user data.

---

## Deep-Learning Pipeline

* **Raw windows** fed into PyTorch models without hand-crafted features.
* Architectures:

  * **CNN**: learns spatial & temporal patterns → **70 %** accuracy on new user.
  * **GRU**: lightweight sequential model (outperformed LSTM here).
  * **LSTM**: larger parameter count but prone to overfit on small dataset.
* **Training**: 100 Optuna trials, early stopping, moderate regularization.

---

## Evaluation & Results

| Method         | Controlled Data | Unseen Data (%) |
| -------------- | --------------- | --------------- |
| Motion (SVM)   | —               | 51 %            |
| Motion (CNN)   | —               | **70 %**        |

> Motion-based deep models generalize better, though they require wearing a compromised watch.

---

## Installation & Usage

```bash
# Clone
git clone https://github.com/hyp3rs0nik/sidechannel-deep-learning.git
cd sidechannel-deep-learning

# API server
cd capture-data-api
npm install
# configure MONGODB_URI in .env
npm start

# Python dependencies
pip install -r requirements.txt

# Train deep model
python deep_learning/train.py \
  --data-dir ../data/sensor/raw \
  --model cnn \
  --trials 50 \
  --epochs 100

# Evaluate
python deep_learning/evaluate.py \
  --model-path ./models/cnn_best.pth \
  --test-data ../data/sensor/test
```
