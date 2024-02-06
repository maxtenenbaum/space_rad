"""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
import seaborn as sns
import scipy.signal as signal
from scipy import stats"""
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

#%% Functions

def load_and_process(file_path, lo, hi, fs=1024):
    # Load data
    headers = ['time', 'PFC1', 'PFC2', 'PPC1', 'PPC2', 'striatum1',
               'striatum2', 'ground1', 'ground2', 'ground3', 'ground4',
               'x_accel', 'y_accel', 'z_accel']
    df = pd.read_csv(file_path, skiprows=4, names=headers, on_bad_lines='skip')
    # Ground correction
    channels = [ 'PFC1', 'PFC2', 'PPC1', 'PPC2', 'striatum1', 'striatum2', 'ground1', 'ground2', 'ground3', 'ground4']
    mean_ground = ((df['ground1'] + df['ground2'] + df['ground3'] + df['ground4'])/4)
    for chan in channels:
        df[chan] = df[chan] - mean_ground
    # Bandpass
    N = 8
    sos = signal.butter(N, (lo, hi), 'bp', fs=fs, output='sos')
    time_vector = np.arange(start=0, stop=(len(df)))
    for chan in channels:
        df[chan] = signal.sosfiltfilt(sos, df[chan])
   
    return df

def plot_data(data, title):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)
    channels = ['PFC1', 'PFC2','PPC1', 'PPC2', 'striatum1', 'striatum2']
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

def power_spectral_density(data, fs=1024, title='Data'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    recording_channels = ['PFC1', 'PFC2','PPC1', 'PPC2', 'striatum1', 'striatum2']
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
    plt.close()  # Close the figure

def combine_channels(data):
    data['PFC'] = (data['PFC1'] + data['PFC2'])/2
    data['striatum'] = (data['striatum1'] + data['striatum2']/2)
    data['PPC'] = (data['PPC1'] + data['PPC2']/2)
    return data

def plot_combined(data, title):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)
    channels = ['PFC','PPC','striatum']
    time_vector = np.arange(start=0, stop= (len(data)))
    fig, ax = plt.subplots(nrows=len(channels), ncols=1, figsize=(48,24), sharex=True)
    for chan in channels:
        i = channels.index(chan)
        ax[i].plot(time_vector/1024, data[chan])
        ax[i].set_title(chan, fontsize=24)
        #ax[i].set_ylim(-25,25)
    plt.suptitle(f'{title}', fontsize=48,y=0.93)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid
    plt.close()

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

def compare_power_between_windows_and_save_table_periodogram(data1, data2, fs=1024, freq_range=(1, 40), alpha=0.05, output_dir='output/tables/', title="Test"):
    channels = ['PFC', 'PPC','striatum']
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
    plt.savefig(os.path.join(output_dir, f'{title}'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return results_summary

def minute_index(minute):
    index = minute * 60 * 1024
    return index


def spectrograms_of_windows(df1, df2, title, recording_channels, fs=1024, freq_limit=40):
    num_channels = len(recording_channels)
    plt.figure(figsize=(72, 36))
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)

    for i, channel in enumerate(recording_channels):
        data1 = df1[channel]
        data2 = df2[channel]

        # Compute spectrograms
        f1, t1_spec, Sxx1 = spectrogram(data1, fs=fs, nperseg=512)
        f2, t2_spec, Sxx2 = spectrogram(data2, fs=fs, nperseg=512)

        # Find the indices corresponding to the frequency limit
        freq_indices = np.where(f1 <= freq_limit)[0]

        # Determine vmin and vmax based on the limited frequency range
        vmin = 10 * np.log10(np.min(Sxx1[freq_indices, :]))
        vmax = 10 * np.log10(np.max(Sxx1[freq_indices, :]))

        # Update vmax if df2 has higher values in the limited range
        vmax = max(vmax, 10 * np.log10(np.max(Sxx2[freq_indices, :])))

        # Plotting for df1
        plt.subplot(num_channels, 2, 2*i + 1)
        plt.pcolormesh(t1_spec, f1, 10 * np.log10(Sxx1), vmin=vmin, vmax=vmax, shading='gouraud', cmap='magma')
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0, freq_limit)
        plt.xlabel('Time [sec]', fontsize=32)
        plt.title(f'Spectrogram for {channel} in first window', fontsize=32)

        # Plotting for df2
        plt.subplot(num_channels, 2, 2*i + 2)
        plt.pcolormesh(t2_spec, f2, 10 * np.log10(Sxx2), vmin=vmin, vmax=vmax, shading='gouraud', cmap='magma')
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0, freq_limit)
        plt.xlabel('Time [sec]', fontsize=32)
        plt.title(f'Spectrogram for {channel} in second window', fontsize=32)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f'{title}', fontsize=48, y=0.98)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))



#%% Pipeline

# Load and process
rat83 = load_and_process('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 83/Baseline Recording Data 20230928/09282023-124713_16g.csv', 0.5, 40)
rat84 = load_and_process('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 84/Baseline 20230928/09282023-145933_16g.csv', 0.5, 40)

# Visualization
rat83_plot = plot_data(rat83, 'DMN Data - Rat83')
rat84_plot = plot_data(rat84, 'DMN Data - Rat84')

rat83_psd = power_spectral_density(rat83, title='DMN Power Spectral Density - Rat83')
rat84_psd = power_spectral_density(rat84, title='DMN Power Spectral Density -  Rat84')

# Combine and plot 
rat83_combined = combine_channels(rat83)
rat84_combined = combine_channels(rat84)

plot_83_combined = plot_combined(rat83_combined, 'Rat83 Combined Channels')
plot_84_combined = plot_combined(rat84_combined, 'Rat84 Combined Channels')

# Halving windows 
rat83_first_half = create_windows(rat83_combined, first=True)
rat83_second_half = create_windows(rat83_combined, second = True)
rat84_first_half = create_windows(rat84_combined, first = True)
rat84_second_half  = create_windows(rat84_combined, second = True)

# Significance
os.makedirs('output/tables/', exist_ok=True)  # Ensure output directory exists
results_rat83 = compare_power_between_windows_and_save_table_periodogram(rat83_first_half, rat83_second_half, output_dir='output/tables/', title="Rat83 Halved")
results_rat84 = compare_power_between_windows_and_save_table_periodogram(rat84_first_half, rat84_second_half, output_dir='output/tables/', title="Rat84 Halved")

## Creating  windows
rat83_first_window = rat83_combined.iloc[minute_index(5):minute_index(7)]
rat84_first_window = rat84_combined.iloc[minute_index(9):minute_index(11)]
rat83_second_window = rat83_combined.iloc[minute_index(12):minute_index(14)]
rat84_second_window = rat84_combined.iloc[minute_index(16):minute_index(18)]
### Significance between windows
results_rat83_windowed = compare_power_between_windows_and_save_table_periodogram(rat83_first_window, rat83_second_window, output_dir='output/tables', title="Rat83 5:7 and 12:14 Windows")
results_rat84_windowed = compare_power_between_windows_and_save_table_periodogram(rat84_first_window, rat84_second_window, output_dir='output/tables', title="Rat84 9:11 and 16:18 Windows")

# Plot Spectrograms of windows
recording_channels = ['PFC','PPC','striatum']
rat83_graphs = spectrograms_of_windows(rat83_first_window, rat83_second_window, 'Rat83 5-7min and 12-14min Spectrogram', recording_channels=recording_channels)
rat84_graphs = spectrograms_of_windows(rat84_first_window, rat84_second_window, 'Rat84 9-11min and 16-18min Spectrogram', recording_channels=recording_channels)

#%%

fixed84_first_half = power_spectral_density(rat84_first_half, title="Rat84 - First Half")
fixed84_second_half = power_spectral_density(rat84_second_half, title="Rat84 - Second Half")