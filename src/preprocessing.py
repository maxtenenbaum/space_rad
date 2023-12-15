import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
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


def mean_ground_correction(data, plot=True):
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3', 'ground1', 'ground2', 'ground3']
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


def bandpass_of_interest(data, hi, lo, fs=1024, plot=True):
    recording_channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    sos = signal.butter(lo,hi,'lp', fs, output = 'sos')
    time_vector = np.arange(start=0, stop= (len(data)))
    for chan in channels:
        data[chan] = signal.sosfiltfilt(sos, data[chan])
    if plot == True:
        fig, ax = plt.subplots(nrows=len(recording_channels), ncols=1, figsize=(24,24), sharex=True)
        for chan in recording_channels:
            i = channels.index(chan)
            ax[i].plot(time_vector, data[chan])
            ax[i].set_title(chan, fontsize=24)
        plt.suptitle('Bandpassed Data', fontsize=48,y=0.93)
        plt.show()
    else:
        return





