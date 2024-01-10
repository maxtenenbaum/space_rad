import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import welch
import seaborn as sns
from scipy.signal import butter, filtfilt

def proper_format(file_path):
    headers = ['time', 'PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2',
               'striatum3', 'ground1', 'ground2', 'ground3', 'ground4',
               'x_accel', 'y_accel', 'z_accel']

    # Read CSV, skip top 4 rows, and set headers
    df = pd.read_csv(file_path, skiprows=4, names=headers, on_bad_lines='skip')

    return df


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
    
"""def combine_channels(data):
    hippocampus_channels = ['hippocampus1', 'hippocampus2']
    data['hippocampus_combined'] = data[hippocampus_channels].mean(axis=1)

    striatum_channels = ['striatum1', 'striatum2', 'striatum3']
    data['striatum_combined'] = data[striatum_channels].mean(axis=1)

    data.drop(columns=hippocampus_channels + striatum_channels, inplace=True, errors='ignore')

    return data
"""

def combine_channels(data):
    data['hippocampus'] = (data['hippocampus1'] + data['hippocampus2'])/2
    data['striatum'] = (data['striatum1'] + data['striatum2'] + data['striatum3']/3)
    return data

def create_windows(data, start_time, stop_time, fs=1024):
    duration = len(data)/fs
    if duration < stop_time or duration < start_time:
        return f'Invalid indices: Data is {duration}s long'
    else:
        start_index = start_time * fs
        stop_index = stop_time * fs
        window = data.iloc[start_index:stop_index]
    return window


def signal_to_noise(data, recording_channels):
    # Calculate mean squared value for the noise channel
    noise_power = np.mean(data['ground1']**2)
    # Check for zero noise power to avoid division by zero
    if noise_power == 0:
        print(f"Noise power for ground is zero, cannot compute SNR.")
        return
    # Calculate and print SNR for each signal channel
    for channel in recording_channels:
        # Check if the signal channel exists in the dataframe
        if channel not in data.columns:
            print(f"Signal channel '{channel}' not found in the dataframe.")
            continue
        # Calculate mean squared value for the signal channel
        signal_power = np.mean(data[channel]**2)

        # Calculate SNR
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"SNR for '{channel}': {snr} dB")

# Example usage:
# calculate_snr_for_channels(your_dataframe, ['channel1', 'channel2', ...], 'ground2')



