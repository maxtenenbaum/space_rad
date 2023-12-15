import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import seaborn as sns

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmaps(dataframes, recording_channels):
    """
    Plot correlation heatmaps for a list of dataframes.

    Args:
    dataframes (list): List of pandas DataFrames.
    recording_channels (list): List of column names to calculate correlations.
    """

    # Compute correlation matrices
    correlation_matrices = [df[recording_channels].corr() for df in dataframes]

    # Process each correlation matrix
    for i, corr_matrix in enumerate(correlation_matrices):
        # Mask the diagonal elements
        mask = np.eye(corr_matrix.shape[0], dtype=bool)
        corr_matrix_no_diagonal = corr_matrix.copy()
        corr_matrix_no_diagonal[mask] = np.nan

        # Find the max and min non-diagonal values
        vmax = np.nanmax(corr_matrix_no_diagonal)
        vmin = np.nanmin(corr_matrix_no_diagonal)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.title(f'Pearson Correlation for {dataframes[i].name}')
        plt.show()

# Example usage
# plot_correlation_heatmaps([dmn_84_first_theta, dmn_84_last10_theta], recording_channels)
