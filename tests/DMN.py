import os
#import pywt
import pandas as pd
#from src import preprocessing, visualization
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft
from scipy.signal import welch
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy import stats

# Functions
def proper_format(file_path):
    headers = ['time', 'PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2',
               'striatum3', 'ground1', 'ground2', 'ground3', 'ground4',
               'x_accel', 'y_accel', 'z_accel']

    # Read CSV, skip top 4 rows, and set headers
    df = pd.read_csv(file_path, skiprows=4, names=headers, on_bad_lines='skip')

    return df

def plot_data(data, title):
    output_dir = "output/plots/"
    #os.makedirs(output_dir, exist_ok=True)
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
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


def bandpass_of_interest(data, lo, hi, fs=1024, plot=False):
    N = 8  # Example order, adjust based on your needs
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3', 'ground1', 'ground2', 'ground3']
    sos = signal.butter(N, (lo, hi), 'bp', fs=fs, output='sos')
    time_vector = np.arange(start=0, stop=(len(data)))
    for chan in channels:
        data[chan] = signal.sosfiltfilt(sos, data[chan])
    if plot:
        fig, ax = plt.subplots(nrows=len(channels), ncols=1, figsize=(24, 24), sharex=True)
        for i, chan in enumerate(channels):
            ax[i].plot(time_vector, data[chan])
            ax[i].set_title(chan, fontsize=24)
        plt.suptitle('Bandpassed Data', fontsize=48, y=0.93)
        plt.show()
    else:
        return data  # Returning data might be more useful than just 'return'

def mean_ground_correction(data, plot=False):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3','ground1', 'ground2', 'ground3']
    mean_ground = ((data['ground1'] + data['ground2'] + data['ground3'])/3)
    time_vector = np.arange(start=0, stop= (len(data)))
    for chan in channels:
        data[chan] = data[chan] - mean_ground
    if plot == True:
        fig, ax = plt.subplots(nrows=len(channels), ncols=1, figsize=(24,24), sharex=True)
        for chan in channels:
            i = channels.index(chan)
            ax[i].plot(time_vector, data[chan])
            ax[i].set_title(chan)
        plt.suptitle('Ground Corrected Data', fontsize=48,y=0.93)
        plt.show()
    return data
"""def power_spectral_density(data, fs=1024, title='Data'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    recording_channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    plt.figure(figsize=(24, 24))  # Create a larger figure for clarity

    # Define a list of colors, one for each channel
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

    # Choose a window size for smoothing
    window_size = 10  # Adjust this based on your data

    # Create a moving average (smoothing) window
    window = np.ones(window_size) / window_size

    for i, chan in enumerate(recording_channels):
        f, S = signal.periodogram(data[chan], fs, scaling='density', nfft=1024)

        # Apply the moving average filter for smoothing
        S_smoothed = np.convolve(S, window, mode='same')

        plt.semilogy(f, S_smoothed, label=chan, color=colors[i % len(colors)], linewidth=4)

    plt.xlabel('Frequency [Hz]', fontsize=36)
    plt.ylabel('PSD [V**2/Hz]', fontsize=36)
    plt.xlim(1, 40)  # Focus on the 1-40 Hz range
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.ylim(0.0001, None)  # Adjust as needed for your data
    plt.grid(visible=True, which='major')
    plt.title(title, fontsize=48)
    plt.legend(fontsize=20)  # Add a legend to distinguish channels, with readable font size

    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Save the figure
    plt.close()  # Close the figure
"""

"""def compare_power_between_windows(data1, data2, fs=1024, freq_range=(1, 40)):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    results = {}

    for chan in channels:
        f1, Pxx1 = signal.welch(data1[chan], fs, nperseg=1024)
        f2, Pxx2 = signal.welch(data2[chan], fs, nperseg=1024)

        idx1 = np.logical_and(f1 >= freq_range[0], f1 <= freq_range[1])
        idx2 = np.logical_and(f2 >= freq_range[0], f2 <= freq_range[1])

        # Perform the Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(Pxx1[idx1], Pxx2[idx2])

        results[chan] = {'avg_power_window1': np.mean(Pxx1[idx1]), 'avg_power_window2': np.mean(Pxx2[idx2]), 'p_value': p_value}

    return results"""


# Example usage
def power_spectral_density(data, fs=1024, title='Data'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    recording_channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    plt.figure(figsize=(24, 24))  # Create a figure

    # Define a list of colors, one for each channel
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



# Import data
rat83_dmn = proper_format('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 83/Baseline Recording Data 20230928/09282023-124713_16g.csv')
rat84_dmn = proper_format("/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 84/Baseline 20230928/09282023-145933_16g.csv")

# Visualize Raw Data
#rat83_dmn_plot = plot_data(rat83_dmn, "Rat83 Raw Plot")
#rat84_dmn_plot = plot_data(rat84_dmn, "Rat84 Raw Plot")

# Bandpass data
rat83_bandpass = bandpass_of_interest(rat83_dmn, 1, 40)
rat84_bandpass = bandpass_of_interest(rat84_dmn, 1, 40)

# Ground correct data
rat83_corrected = mean_ground_correction(rat83_bandpass)
rat84_corrected = mean_ground_correction(rat84_bandpass)

# Plot corrected data
#rat83_cleaned_plot = plot_data(rat83_corrected, "Rat83 Corrected Data")
#rat84_cleaned_plot = plot_data(rat84_corrected, "Rat84 Corrected Data")

# Splitting into sections

## Calculating length
rat83_total_time = len(rat83_corrected)/1024
print(f'Rat83 recording is {rat83_total_time} seconds long')
rat84_total_time = len(rat84_corrected)/1024
print(f'Rat84 recording is {rat84_total_time} seconds long')

## Time indicies
three_minute = 3 * 60 * 1024  # Start of 3 minutes
five_minute = 5 * 60 * 1024    # End of 5 minutes
ten_minute = 10 * 60 * 1024
twelve_minute = 12 * 60 * 1024

## Creating  windows
rat83_first_window = rat83_corrected.iloc[three_minute:five_minute]
rat84_first_window = rat84_corrected.iloc[three_minute:five_minute]

rat83_second_window = rat83_corrected.iloc[ten_minute:twelve_minute]
rat84_second_window = rat84_corrected.iloc[ten_minute:twelve_minute]

# Power spectrums
rat83_corrected_psd =power_spectral_density(rat83_corrected,title="Rat83 Whole Session PSD")
rat84_corrected_psd =power_spectral_density(rat84_corrected,title="Rat84 Whole Session PSD")
rat83_first_window_psd = power_spectral_density(rat83_first_window,title="Rat83 PSD 3-5 minutes")
rat84_first_window_psd = power_spectral_density(rat84_first_window,title="Rat84 PSD 3-5 minutes")
rat83_second_window_psd = power_spectral_density(rat83_second_window,title="Rat83 PSD 10-12 minutes")
rat84_second_window_psd = power_spectral_density(rat84_second_window,title="Rat84 PSD 10-12 minutes")





"""def compare_power_between_windows(data1, data2, fs=1024, freq_range=(1, 40), alpha=0.05):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    results_summary = {}

    for chan in channels:
        f1, Pxx1 = signal.welch(data1[chan], fs, nperseg=1024)
        f2, Pxx2 = signal.welch(data2[chan], fs, nperseg=1024)

        idx1 = np.logical_and(f1 >= freq_range[0], f1 <= freq_range[1])
        idx2 = np.logical_and(f2 >= freq_range[0], f2 <= freq_range[1])

        # Perform the Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(Pxx1[idx1], Pxx2[idx2])

        # Determine if the result is statistically significant
        significant = "Yes" if p_value < alpha else "No"

        # Prepare a summary of results
        results_summary[chan] = {
            'Avg Power Window 1': np.mean(Pxx1[idx1]),
            'Avg Power Window 2': np.mean(Pxx2[idx2]),
            'P-Value': p_value,
            'Significant Difference': significant
        }

    return results_summary

# Example usage:
results_rat83 = compare_power_between_windows(rat83_first_window, rat83_second_window)
results_rat84 = compare_power_between_windows(rat84_first_window, rat84_second_window)

# Print results in a readable format
print("Rat 83 Comparison Results:")
for channel, results in results_rat83.items():
    print(f"\nChannel: {channel}")
    for key, value in results.items():
        print(f"{key}: {value:.2e}" if isinstance(value, float) else f"{key}: {value}")

print("\nRat 84 Comparison Results:")
for channel, results in results_rat84.items():
    print(f"\nChannel: {channel}")
    for key, value in results.items():
        print(f"{key}: {value:.2e}" if isinstance(value, float) else f"{key}: {value}")

# Comparing powers
results_rat83 = compare_power_between_windows(rat83_first_window, rat83_second_window)
results_rat84 = compare_power_between_windows(rat84_first_window, rat84_second_window)

print(results_rat83)
print(results_rat84)"""


def compare_power_between_windows_and_save_table_periodogram(data1, data2, fs=1024, freq_range=(1, 40), alpha=0.05, output_dir='output/tables/', title="Test"):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
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

        # Append results to the summary list
        results_summary.append([chan, np.mean(Pxx1[idx1]), np.mean(Pxx2[idx2]), p_value, significant])

    # Create a table and save as PNG
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table_data = [["Channel", "Avg Power Window 1", "Avg Power Window 2", "P-Value", "Significant Difference"]] + results_summary
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Adjust table style as needed
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{title}comparison_results_periodogram.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return results_summary

# Example usage:
os.makedirs('output/tables/', exist_ok=True)  # Ensure output directory exists
results_rat83 = compare_power_between_windows_and_save_table_periodogram(rat83_first_window, rat83_second_window, output_dir='output/tables/', title="Rat83")
results_rat84 = compare_power_between_windows_and_save_table_periodogram(rat84_first_window, rat84_second_window, output_dir='output/tables/', title="Rat84")




"""
def compare_power_between_windows_wavelet(data1, data2, fs=1024, freq_range=(1, 40), wavelet='cmor', alpha=0.05, output_dir='output/tables/', title="Test"):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    results_summary = []

    for chan in channels:
        # Perform Continuous Wavelet Transform
        coef1, freqs1 = pywt.cwt(data1[chan], scales=np.arange(1, 100), wavelet=wavelet, sampling_period=1/fs)
        coef2, freqs2 = pywt.cwt(data2[chan], scales=np.arange(1, 100), wavelet=wavelet, sampling_period=1/fs)

        # Filter frequencies within the specified range
        idx1 = np.logical_and(freqs1 >= freq_range[0], freqs1 <= freq_range[1])
        idx2 = np.logical_and(freqs2 >= freq_range[0], freqs2 <= freq_range[1])

        # Calculate average power
        power1 = np.mean(np.abs(coef1[idx1])**2, axis=0)
        power2 = np.mean(np.abs(coef2[idx2])**2, axis=0)

        # Perform the Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(power1, power2)

        # Determine if the result is statistically significant
        significant = "Yes" if p_value < alpha else "No"

        # Append results to the summary list
        results_summary.append([chan, np.mean(power1), np.mean(power2), p_value, significant])

    # Create a table and save as PNG
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table_data = [["Channel", "Avg Power Window 1", "Avg Power Window 2", "P-Value", "Significant Difference"]] + results_summary
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Adjust table style as needed
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{title}wavelet_comparison_results.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return results_summary

# Example usage:
os.makedirs('output/tables/', exist_ok=True)  # Ensure output directory exists
results_rat83 = compare_power_between_windows_wavelet(rat83_first_window, rat83_second_window, output_dir='output/tables/', title="83")
results_rat84 = compare_power_between_windows_wavelet(rat84_first_window, rat84_second_window, output_dir='output/tables/', title = "84")"""