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
rat83_dmn = preprocessing.proper_format('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 83/Baseline Recording Data 20230928/09282023-124713_16g.csv')
rat84_dmn = preprocessing.proper_format("/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 84/Baseline 20230928/09282023-145933_16g.csv")

# Bandpass data
rat83_dmn_bandpassed = preprocessing.bandpass_of_interest(rat83_dmn, 1, 40)
rat84_dmn_bandpassed = preprocessing.bandpass_of_interest(rat84_dmn, 1, 40)

# Ground correct
rat83_corrected = preprocessing.mean_ground_correction(rat83_dmn_bandpassed)
rat84_corrected = preprocessing.mean_ground_correction(rat84_dmn_bandpassed)

# Combine like channels 
"""
Due to electrode setup, channels are averaged with each other in order to reduce variance
"""
rat83_combined = preprocessing.combine_channels(rat83_corrected)
rat84_combined = preprocessing.combine_channels(rat84_corrected)

# Plot Combined Preprocessed Data

### Modified plot function
def plot_combined_data(data, title):
    output_dir = "output/plots/"
    #os.makedirs(output_dir, exist_ok=True)
    channels = ['PFC', 'hippocampus', 'striatum']
    time_vector = np.arange(start=0, stop= (len(data)))
    fig, ax = plt.subplots(nrows=len(channels), ncols=1, figsize=(24,24), sharex=True)
    for chan in channels:
        i = channels.index(chan)
        ax[i].plot(time_vector/1024, data[chan])
        ax[i].set_title(chan, fontsize=24)
        #ax[i].set_ylim(-25,25)
    plt.suptitle(f'{title}', fontsize=48,y=0.93)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid
    plt.close()

plot_83 = plot_combined_data(rat83_combined, "Rat83 Combined Channels")
plot_84 = plot_combined_data(rat84_combined, "Rat84 Combined Channels")

