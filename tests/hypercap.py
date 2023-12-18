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
#rat55_raw_plot = visualization.plot_data(rat55_raw, "Rat55 Raw Data")

# Ground correct raw data
preprocessing.mean_ground_correction(rat55_raw, plot='False')

# Bandpass raw data
rat55_theta = preprocessing.bandpass_of_interest(rat55_raw, 4, 8)
rat55_alpha = preprocessing.bandpass_of_interest(rat55_raw, 8, 13)
rat55_beta = preprocessing.bandpass_of_interest(rat55_raw, 13, 30)

# Plot processed data
#rat55_theta_plot = visualization.plot_data(rat55_theta, "Rat55 Theta Band Data")
#rat55_alpha_plot = visualization.plot_data(rat55_alpha, "Rat55 Alpha Band Data")
#rat55_beta_plot = visualization.plot_data(rat55_beta, "Rat55 Beta Band Data")

# PSD WORK IN PROGRESS
rat55_raw_PSD = visualization.power_spectral_density(rat55_raw, title='Raw PSD')
rat55_alpha_PSD = visualization.power_spectral_density(rat55_alpha, title= "Alpha PSD")
rat55_beta_PSD = visualization.power_spectral_density(rat55_beta, title = 'Beta PSD')
rat55_theta_PSD = visualization.power_spectral_density(rat55_theta, title='Theta PSD')

# Plotting power 
#rat55_theta_power = visualization.plot_power(rat55_theta, 'Power Theta')
#rat55_alpha_power = visualization.plot_power(rat55_alpha, 'Power Alpha')
#rat55_beta_power = visualization.plot_power(rat55_beta, 'Power Beta')
