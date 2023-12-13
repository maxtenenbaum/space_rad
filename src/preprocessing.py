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