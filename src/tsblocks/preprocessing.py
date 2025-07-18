import numpy as np
import pandas as pd



def uniform_time_index(
    times : np.datetime64 | pd.DatetimeIndex,
    freq  : str = "1min"
) -> tuple[
        pd.DatetimeIndex, 
        np.ndarray
]:
    """
    Given a numpy datetime64 array or a pandas DatetimeIndex, returns:
      - a new uniform DatetimeIndex at the specified frequency
      - a boolean mask indicating which times were present in the original input.

    Parameters
    ----------
    times : np.ndarray[np.datetime64] or pd.DatetimeIndex
    freq : str
        Frequency string like '1min'.

    Returns
    -------
    uniform_times : pd.DatetimeIndex
    original_mask : np.ndarray of bool
        True where the time was in the original input, False where inserted.
    """
    # ensure times is handled as pandas datetimeindex
    times = pd.to_datetime(times)
    uniform_times = pd.date_range(start=times.min(), end=times.max(), freq=freq)
    # Boolean mask: True if this time was in the original times
    original_mask = uniform_times.isin(times)
    return uniform_times, original_mask