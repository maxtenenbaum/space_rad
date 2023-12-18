import numpy as np
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def plot_data(data, title):
    output_dir = "output/plots"
    #os.makedirs(output_dir, exist_ok=True)
    channels = ['PFC', 'hippocampus1', 'hippocampus2', 'striatum1', 'striatum2', 'striatum3', 'ground1', 'ground2', 'ground3']
    time_vector = np.arange(start=0, stop= (len(data)))
    fig, ax = plt.subplots(nrows=len(channels), ncols=1, figsize=(24,24), sharex=True)
    for chan in channels:
        i = channels.index(chan)
        ax[i].plot(time_vector/1024, data[chan])
        ax[i].set_title(chan, fontsize=24)
    plt.suptitle(f'{title}', fontsize=48,y=0.93)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))  # Ensure the filename is valid
    plt.close()

