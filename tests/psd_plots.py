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

def combine_channels(data):
    data['PFC'] = (data['PFC1'] + data['PFC2'])/2
    data['Striatum'] = (data['striatum1'] + data['striatum2']/2)
    data['PPC'] = (data['PPC1'] + data['PPC2']/2)
    return data


def power_spectral_density(data, fs=1024, title='Data'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    recording_channels = ['PFC', 'PPC', 'Striatum']
    plt.figure(figsize=(24, 24), facecolor='w')  # Create a figure with a white background
    colors = ['blue', 'darkorange', 'green']  # Use distinct colors for visibility

    for i, chan in enumerate(recording_channels):
        f, S = signal.periodogram(data[chan], fs, scaling='density', nfft=1024)
        plt.semilogy(f, S, label=chan, color=colors[i % len(colors)], linewidth=5)
    
    plt.xlabel('Frequency [Hz]', fontsize=40)
    plt.ylabel('PSD [V**2/Hz]', fontsize=40)
    plt.xlim(1, 40)  # Set x-axis limits
    plt.ylim(0.0001, None)  # Set y-axis limits
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.grid(visible=True, which='major', color='grey', linestyle='-', linewidth=0.5)
    plt.title(title, fontsize=50, pad=20)

    plt.legend(fontsize=30, loc='upper right')  # Increase legend font size and position

    plt.savefig(os.path.join(output_dir, f"{title}.png"), bbox_inches='tight')  # Save the figure with tight bounding box
    plt.close()  # Close the figure



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

    #%%
# Load and process
rat83 = load_and_process('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 83/Baseline Recording Data 20230928/09282023-124713_16g.csv', 0.5, 40)
rat84 = load_and_process('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 84/Baseline 20230928/09282023-145933_16g.csv', 0.5, 40)

rat83_combined = combine_channels(rat83)
rat84_combined = combine_channels(rat84)

# Halving windows 
rat83_first_half = create_windows(rat83_combined, first=True)
rat83_second_half = create_windows(rat83_combined, second = True)
rat84_first_half = create_windows(rat84_combined, first = True)
rat84_second_half  = create_windows(rat84_combined, second = True)

fixed84_first_half = power_spectral_density(rat84_first_half, title="Rat84 - First Half")
fixed84_second_half = power_spectral_density(rat84_second_half, title="Rat84 - Second Half")
