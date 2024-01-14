"""
This pipeline is different than the first pipeline in the following ways:

1. Average striatum channels
2. Average hippocampal channels
3. Windowing in halves instead of 2 minute intervals

Data loading, bandapss, ground correction, are identical

"""

"""
VISUALIZSTION MODULE AT BOTTOM NOT WORKING
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import welch, butter, filtfilt
from scipy import stats
from scipy.fft import fft, ifft
from scipy.signal import spectrogram
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

# Windowing by splitting in half

### Modified windowing function
def create_windows(data, fs=1024, first=False, second=False):
    duration = len(data)/fs
    if duration % 2 == 0:
        halfway_index = (duration * fs) / 2
    else:
        halfway_index = int(((duration * fs) / 2)-0.5)
    if first == True:
        window = data.iloc[0:halfway_index]
    elif second == True:
        window = data.iloc[halfway_index:]
    return window

rat83_first_half = create_windows(rat83_combined, first=True)
rat83_second_half = create_windows(rat83_combined, second = True)
rat84_first_half = create_windows(rat84_combined, first = True)
rat84_second_half  = create_windows(rat84_combined, second = True)

# Windowed power spectral density 
### Modified function
def power_spectral_density(data, fs=1024, title='Data'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    recording_channels = ['PFC', 'hippocampus','striatum']
    plt.figure(figsize=(24, 24))  # Create a figure
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed
    for i, chan in enumerate(recording_channels):
        f, S = signal.periodogram(data[chan], fs, scaling='density', nfft=1024)
        plt.semilogy(f, S, label=chan, color=colors[i % len(colors)], linewidth=4)
    plt.xlabel('Frequency [Hz]', fontsize=36)
    plt.ylabel('PSD [V**2/Hz]', fontsize=36)
    plt.xlim(1, 40)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.ylim(0.0001,None)
    plt.grid(visible=True, which='major')
    plt.title(title, fontsize=48)
    plt.legend()  # Add a legend to distinguish channels
    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Save the figure
    plt.close()  # Close the figure"""

rat83_combined_first = power_spectral_density(rat83_first_half, title='83 Combined First Half')
rat83_combined_second = power_spectral_density(rat83_second_half, title = '83 Combined Second Half')
rat84_combined_first = power_spectral_density(rat84_first_half, title = '84 Combined First Half')
rat84_combined_second = power_spectral_density(rat84_second_half, title='84 Combined Second Half')

# Calculating significance
def compare_power_between_windows_and_save_table_periodogram(data1, data2, fs=1024, freq_range=(1, 40), alpha=0.05, output_dir='output/tables/', title="Test"):
    channels = ['PFC', 'hippocampus','striatum']
    results_summary = []

    for chan in channels:
        f1, Pxx1 = signal.periodogram(data1[chan], fs)
        f2, Pxx2 = signal.periodogram(data2[chan], fs)

        idx1 = np.logical_and(f1 >= freq_range[0], f1 <= freq_range[1])
        idx2 = np.logical_and(f2 >= freq_range[0], f2 <= freq_range[1])

        # Perform the Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(Pxx1[idx1], Pxx2[idx2])

        # Determine if the result is statistically significant
        significant = "Yes" if p_value < alpha else "No"

        # Determine direction
        direction = "Decrease" if (np.mean(Pxx1[idx1])) > (np.mean(Pxx2[idx2])) else "Increase"

        # Append results to the summary list
        results_summary.append([chan, round(np.mean(Pxx1[idx1]),3), round(np.mean(Pxx2[idx2]),3), p_value, significant, direction])

    # Create a table and save as PNG
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table_data = [["Channel", "Avg Power Window 1", "Avg Power Window 2", "P-Value", "Significant Difference", "Direction"]] + results_summary
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Adjust table style as needed
    table.auto_set_font_size(True)
    #table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{title}comparison_results_periodogram.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return results_summary

# Example usage:
os.makedirs('output/tables/', exist_ok=True)  # Ensure output directory exists
results_rat83 = compare_power_between_windows_and_save_table_periodogram(rat83_first_half, rat83_second_half, output_dir='output/tables/', title="Rat83 Halved")
results_rat84 = compare_power_between_windows_and_save_table_periodogram(rat84_first_half, rat84_second_half, output_dir='output/tables/', title="Rat84 Halved")

# Signal to Noise Ratio

recording_channels = ['PFC', 'hippocampus', 'striatum']
rat83_snr = preprocessing.signal_to_noise(rat83_combined, recording_channels)
rat84_snr = preprocessing.signal_to_noise(rat84_combined, recording_channels)




# Plot data and signals
recording_channels = ['PFC', 'hippocampus', 'striatum']
#rat83_spectrogram = visualization.spectrogram_and_data(rat83_combined, recording_channels, title = "Rat83 First Half Data")
visualization.spectrogram_and_data(rat84_first_half, recording_channels, title = "Rat84 Full Data")