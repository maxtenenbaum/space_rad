import os
import pandas as pd
from src import preprocessing, visualization
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import numpy as np

# Load data
rat55_raw = preprocessing.proper_format("/Volumes/Elements/Space Radiation Project/evms_hypercapnia_male1/Rat 55/Normoxic - Hypercapnic/06052023 Normoxic cond Standard housing/rat55_H06052023-105935_16g.csv")

# Plot raw data
rat55_raw_plot = visualization.plot_data(rat55_raw, "Rat55 Raw Data")

# Bandpass raw data
rat55_theta_raw = preprocessing.bandpass_of_interest(rat55_raw, 4, 8)
rat55_alpha_raw = preprocessing.bandpass_of_interest(rat55_raw, 8, 13)
rat55_beta_raw = preprocessing.bandpass_of_interest(rat55_raw, 13, 30)

# Ground correct raw data
rat55_theta_corrected = preprocessing.mean_ground_correction(rat55_theta_raw, plot='False')
rat55_alpha_corrected = preprocessing.mean_ground_correction(rat55_alpha_raw, plot='False')
rat55_beta_corrected = preprocessing.mean_ground_correction(rat55_beta_raw, plot='False')

# Plot processed data

rat55_theta_plot = visualization.plot_data(rat55_theta_corrected, "Rat55 Theta Band Data")
rat55_alpha_plot = visualization.plot_data(rat55_alpha_corrected, "Rat55 Alpha Band Data")
rat55_beta_plot = visualization.plot_data(rat55_beta_corrected, "Rat55 Beta Band Data")

