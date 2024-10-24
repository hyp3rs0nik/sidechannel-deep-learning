# ML/DL Numeric Keystroke Inference

This is a class project focused on inferring numeric keystrokes using data captured from a keylogger and smartwatch sensors. It machine learning to analyze the sensor data, with the ultimate goal of identifying specific keystrokes made by the user. The project explores how smartwatch accelerometer and gyroscope data can be used to infer which numeric keys are pressed, providing insights into potential vulnerabilities related to keystroke detection through wearable devices.

The inference process begins by capturing raw data from two sources:

1. **Keylogger (capture-key-app)**: This component runs on the user device, recording keypresses and associating them with precise timestamps.
2. **Smartwatch (capture-sensor-app)**: The smartwatch worn by the user captures movement data from its accelerometer and gyroscope sensors. The movement data is timestamped to align with the captured keystrokes.

The collected data from both the keylogger and smartwatch is processed using the `data-process/process-raw-data.py` script. This script performs data association, aligning the keypress events with the corresponding sensor data based on timestamps. After preprocessing, the aligned data is used to train a neural network model, which attempts to infer the numeric keypresses based on the sensor data.

## Limitations
This experiment used a relatively small dataset, which limits the generalizability of the results. In addition, data quality may be impacted by sensor noise, differences in typing styles, and variations in smartwatch positioning. Due to these constraints, **the current model may not achieve optimal performance across all conditions and should be considered a proof of concept**.

### Setup and Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sidechannel-research-main.git
   cd sidechannel-research-main
   ```
2. Install necessary dependencies
   ```sh
   pip install -r requirements.txt  # (Python dependencies)
   ```
3. Install necessary dependencies on the API server. (assuming you have installed NodeJS)
    ```sh
    cd ./capture-data-api
    npm install
    node server/js
    ```

## Usage
```sh
python .\machine_learning_models\trainer.py --accel .\data\training\v2\sensor_accel.csv --gyro .\data\training\v2\sensor_gyro.csv --keystrokes .\data\training\v2\keystrokes.csv --window 480 --augmentation 2
```

- **window**: Defines the time window for associating sensor data with keystrokes. Default: `280` ms. Increasing this value may capture more context but could introduce noise.
- **augmentation**: Number of times the data need to be augment. If you have a small dataset, augmentation maybe needed.

## Future Work
- **Dataset Expansion**: Collecting more data to improve the robustness and accuracy of the model.
- **Algorithm Improvements**: Experimenting with different neural network architectures.
