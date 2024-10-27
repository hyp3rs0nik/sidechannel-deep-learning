# feature_calculations.py
import numpy as np
from scipy.integrate import cumulative_trapezoid

def calculate_rms(data):
    return np.sqrt((data[["x", "y", "z"]] ** 2).mean(axis=1))

def calculate_rmse(data, window=5):
    rms = calculate_rms(data)
    return rms.rolling(window=window, min_periods=1).apply(lambda x: np.sqrt(np.mean((x - x.mean()) ** 2)), raw=False)

def calculate_min(data, axis):
    return data[axis].rolling(window=5, min_periods=1).min()

def calculate_max(data, axis):
    return data[axis].rolling(window=5, min_periods=1).max()

def calculate_magnitude(data):
    return np.sqrt((data[["x", "y", "z"]] ** 2).sum(axis=1))

def calculate_sma(data, window=5):
    return data[["x", "y", "z"]].abs().sum(axis=1).rolling(window=window, min_periods=1).sum()

def calculate_displacement_2d(data):
    return np.sqrt(cumulative_trapezoid(data["x"], initial=0)**2 + cumulative_trapezoid(data["y"], initial=0)**2)

feature_functions = {
    "rms": calculate_rms,
    "rmse": lambda data: calculate_rmse(data),
    "x_min": lambda data: calculate_min(data, "x"),
    "x_max": lambda data: calculate_max(data, "x"),
    "y_min": lambda data: calculate_min(data, "y"),
    "y_max": lambda data: calculate_max(data, "y"),
    "z_min": lambda data: calculate_min(data, "z"),
    "z_max": lambda data: calculate_max(data, "z"),
    "magnitude_min": lambda data: calculate_min(data.assign(magnitude=calculate_magnitude(data)), "magnitude"),
    "magnitude_max": lambda data: calculate_max(data.assign(magnitude=calculate_magnitude(data)), "magnitude"),
    "sma": lambda data: calculate_sma(data),
    "displacement_2d": lambda data: calculate_displacement_2d(data)
}
