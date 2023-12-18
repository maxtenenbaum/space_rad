import os
import pandas as pd
from src import preprocessing, visualization
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import numpy as np

def load_and_process(filepath, name):
    # Load data
    raw_data = preprocessing.proper_format(filepath)
    # Ground correction
    preprocessing.mean_ground_correction(raw_data,plot='False')
    # Bandpass data
    theta = preprocessing.bandpass_of_interest(raw_data, 4, 8)
    alpha = preprocessing.bandpass_of_interest(raw_data, 8, 13)
    beta = preprocessing.bandpass_of_interest(raw_data, 13, 40)
    return raw_data, theta, alpha, beta, name

def plot_data_and_psd(raw_data, theta, alpha, beta):
    # Plot processed data and PSD
    visualization.plot_data(raw_data, f"{name} Raw Data")
    visualization.plot_data(theta, f"{name}Theta Band Data")
    visualization.plot_data(alpha, f"{name} Alpha Band Data")
    visualization.plot_data(beta, f"{name} Beta Band Data")

    visualization.power_spectral_density(raw_data, title= f"{name} Raw PSD")
    visualization.power_spectral_density(alpha, title= f"{name} Alpha PSD")
    visualization.power_spectral_density(beta, title= f"{name} Beta PSD")
    visualization.power_spectral_density(theta, title= f"{name} Theta PSD")
