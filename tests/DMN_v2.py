import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import welch, butter, filtfilt
from scipy import stats
from scipy.fft import fft, ifft
import sys
sys.path.append('/Users/maxtenenbaum/Desktop/space_radiation/space_rad')
from src import preprocessing, analysis, visualization

# Load data
rat83_dmn = preprocessing.proper_format(file_path)
rat84_dmn = preprocessing.proper_format(file_path)

# Bandpass data
rat83_dmn_bandpassed = preprocessing.bandpass_of_interest(rat83_dmn, 1, 40)
rat84_dmn_bandpassed = preprocessing.bandpass_of_interest(rat84_dmn, 1, 40)

# Ground correct
rat83_corrected = preprocessing.mean_ground_correction(rat83_dmn_bandpassed)
rat84_corrected = preprocessing.mean_ground_correction(rat84_dmn_bandpassed)

# Combine like channels 
"""
Due to electrode setup, channels are averaged with eachother in order to reduce variance
"""
