"""Signal processing utilities."""

import numpy as np

from scipy import signal

from ..config import FPS


def smooth_signal(x, mode="savgol", window_len=15, polyorder=7, **kwargs):
    """Smooth a signal with different options.

    Available smoothing techniques:
    * Savitzky-Golay filter
    * Median filter
    * Boxcar/Bartlett-windowed moving averages

    Might add Hampel filtering in a future version

    Source:
    # https://www.jneurosci.org/content/jneuro/early/2022/02/22/JNEUROSCI.0938-21.2022.full.pdf

    Parameters
    ----------
    x : array-like
        Input signal
    mode : str, optional
        Smoothing mode, by default 'savgol' (Savitzky-Golay filter)
    window_len : int, optional
        Window length, by default 7
    polyorder : int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    **kwargs : kwargs
        Other arguments passed to the smoothing function

    Returns
    -------
    numpy.ndarray
        Smoothed signal
    """

    if mode == "boxcar":
        w = signal.boxcar(M=window_len)
        return np.convolve(x, w / w.sum(), mode="same", **kwargs)
    if mode == "bartlett":
        w = signal.bartlett(M=window_len)
        return np.convolve(x, w / w.sum(), mode="same", **kwargs)
    if mode == "medfilt":
        return signal.medfilt(x, kernel_size=(window_len,), **kwargs)
    if mode == "savgol":
        return signal.savgol_filter(
            x,
            window_length=window_len,
            polyorder=kwargs.pop("polyorder", polyorder),
            **kwargs,
        )

    raise ValueError(f"Unknown smoothing mode: {mode}.")


def derive(x, order=1, **kwargs):
    """Compute the n-th derivative of a signal using a Savitzky-Golay filter

    If order = 0, simply applies smoothing

    Parameters
    ----------
    x : array-like
        Input array
    order : int, optional
        Derivative order, by default 1
    **kwargs : kwargs
        All other arguments are passed to smooth_signal

    Returns
    -------
    array-like
        n-th derivative of x
    """
    return smooth_signal(x, mode="savgol", deriv=order, delta=FPS, **kwargs)


def filter_outliers(df, max_zscore=3):
    """Replace z-score outliers by NaN.

    True if abs(z-score) > max_zscore, False otherwise

    Parameters
    ----------
    df : pandas.DataFrame
        Input 2D data, columns = features

    Returns
    -------
    pandas.DataFrame
        Filtered data, where outliers are replaced by NaNs
    """
    df_scaled = df.sub(df.mean()).div(df.std())

    outlier_mask = df_scaled.applymap(lambda x: abs(x) > max_zscore)

    return df.mask(outlier_mask)
