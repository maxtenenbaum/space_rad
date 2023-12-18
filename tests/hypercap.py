import os
import pandas as pd
from src import preprocessing, visualization
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import numpy as np
import tkinter as tk
from tkinter import filedialog

def load_and_process(filepath, name):
    # Load data
    raw_data = preprocessing.proper_format(str(filepath))
    # Ground correction
    preprocessing.mean_ground_correction(raw_data,plot='False')
    # Bandpass data
    theta = preprocessing.bandpass_of_interest(raw_data, 4, 8)
    alpha = preprocessing.bandpass_of_interest(raw_data, 8, 13)
    beta = preprocessing.bandpass_of_interest(raw_data, 13, 40)
    return raw_data, theta, alpha, beta, name

def plot_data_and_psd(raw_data, theta, alpha, beta, name):
    # Plot processed data and PSD
    visualization.plot_data(raw_data, f"{name} Raw Data")
    visualization.plot_data(theta, f"{name} Theta Band Data")
    visualization.plot_data(alpha, f"{name} Alpha Band Data")
    visualization.plot_data(beta, f"{name} Beta Band Data")

    visualization.power_spectral_density(raw_data, title= f"{name} Raw PSD")
    visualization.power_spectral_density(alpha, title= f"{name} Alpha PSD")
    visualization.power_spectral_density(beta, title= f"{name} Beta PSD")
    visualization.power_spectral_density(theta, title= f"{name} Theta PSD")

def main(filepath, name):
    raw_data, theta, alpha, beta, name = load_and_process(filepath, name)
    plot_data_and_psd(raw_data, theta, alpha, beta, name)

main('/Volumes/Elements/Space Radiation Project/evms_hypercapnia_male1/Rat 55/Normoxic - Hypercapnic/06052023 Normoxic cond Standard housing/rat55_H06052023-105935_16g.csv', 'Rat55 Baseline')






"""
def open_file_and_process():
    filepath = filedialog.askopenfilename(title="Select file", filetypes=[("CSV files", "*.csv")])
    if filepath:
        name = os.path.basename(filepath)  # Extract a name from the filepath
        main(filepath, name)

def create_main_window():
    root = tk.Tk()
    root.title("Data Processing and Visualization")

    open_button = tk.Button(root, text="Open File", command=open_file_and_process)
    open_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_main_window()"""