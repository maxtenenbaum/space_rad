import os
import pandas as pd
from src import preprocessing, visualization
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import numpy as np

# Import data
rat83_dmn = preprocessing.proper_format('/Volumes/Elements/Space Radiation Project/Default Mode Network Study/Rat 83/Baseline Recording Data 20230928/09282023-124713_16g.csv')

# Visualize Raw Data
rat83_dmn_plot = visualization.plot_data(rat83_dmn, "Rat83 Raw Plot")

