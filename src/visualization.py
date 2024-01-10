import numpy as np
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pywt
from scipy.signal import spectrogram

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
    plt.xlim(2, 50)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.ylim(0.0001,None)
    plt.grid(visible=True, which='major')
    plt.title(title, fontsize=48)
    plt.legend()  # Add a legend to distinguish channels

    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Save the figure
    plt.close()  # Close the figure

def plot_power(data, title, reference_power=1):
    output_dir = "output/plots/"
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3']
    time_vector = np.arange(len(data)) / 1024  # Assuming a sample rate of 1024 Hz for time axis

    fig, ax = plt.subplots(figsize=(24, 24))

    # Define a list of colors, one for each channel
    colors = ['b', 'g', 'r', 'c', 'm','k']  # Add more colors if you have more channels

    for i, chan in enumerate(channels):
        # Calculate power and convert to decibels
        power = np.square(data[chan])  # Power of the signal
        decibels = 10 * np.log10(power / reference_power)

        ax.plot(time_vector, decibels, label=chan, color=colors[i % len(colors)])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Power (dB)")
    #ax.set_xlim(500,505)
    ax.set_title(title)
    ax.legend()  # Add a legend to distinguish channels

    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid




def spectrogram_and_data(data, recording_channels, fs=1024, title='Test'):
    output_dir = "output/plots/"
    os.makedirs(output_dir, exist_ok=True)
    n_channels = len(recording_channels)
    fig, axes = plt.subplots(2 * n_channels, 1, figsize=(10, 4 * n_channels))

    for i, channel in enumerate(recording_channels):
        # Plot the raw data
        axes[2*i].plot(data[channel])
        axes[2*i].set_title(f'{title} - {channel} - Raw Data')
        axes[2*i].set_xlabel('Time [samples]')
        axes[2*i].set_ylabel('Amplitude')

        # Compute and plot the spectrogram
        f, t, Sxx = spectrogram(data[channel], fs=fs)
        axes[2*i + 1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        axes[2*i + 1].set_ylabel('Frequency [Hz]')
        axes[2*i + 1].set_xlabel('Time [sec]')
        axes[2*i + 1].set_title(f'{title} - {channel} - Spectrogram')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Save the figure
    plt.close() 
