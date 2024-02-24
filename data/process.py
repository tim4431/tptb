import numpy as np


def smooth_1d_data(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Smooth 1D data using a window size.
    - data: 1D data
    - window_size: size of the window
    """
    kernel = np.ones(window_size) / window_size
    y_smooth = np.convolve(data, kernel, mode="same")
    #
    for i in range(window_size // 2):
        y_smooth[i] = data[i]  # fix the first i elements
    for i in range(-window_size // 2, 0):
        y_smooth[i] = data[i]  # fix the last i elements
    #
    return y_smooth
