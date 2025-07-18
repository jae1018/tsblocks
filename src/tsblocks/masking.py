import numpy as np
from tqdm import tqdm
from typing import Tuple



def interp_nans_within_limit(
        
        arr             : np.ndarray,
        max_consec_nans : int = 1,
        progress_bar    : bool = True
        
) -> Tuple[ np.ndarray,
            np.ndarray ]:
    
    """
    Given a 1D numpy array with NaNs, interpolate only short runs
    of NaNs (length ≤ max_consec_nans). Longer runs stay as NaNs.

    Visually, this sequence:
        1 nan 3 4 nan nan 7 8
    with max_consec_nans = 1 becomes
        1 2 3 4 nan nan 7 8
    ... however with max_consec_nans = 2, it becomes
        1 2 3 4 5 6 7 8

    Parameters
    ---------
    arr : 1d numpy array
        array of SINGLE time series
    max_consec_nans : int (optional, default 1)
        Max number of consecutive nans allowed to interpolate over
    progress_var : bool (optional, default True)
        If True, uses tqdm bar to print progress

    Returns
    -------
    arr_out : np.ndarray
        The interpolated array.
    interp_mask : np.ndarray of bool
        Mask indicating where interpolation happened.
    """
    
    arr = arr.copy()
    isnan = np.isnan(arr)

    # Find start and end of NaN runs
    diff = np.diff(isnan.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle if starts or ends with NaNs
    if isnan[0]:
        starts = np.insert(starts, 0, 0)
    if isnan[-1]:
        ends = np.append(ends, len(arr))

    iterable = tqdm(zip(starts, ends), descp='Interpolating NaNs') \
               if progress_bar else zip(starts, ends)
    for start, end in iterable:
        run_length = end - start
        
        # linearly interpolate only this run
        if run_length <= max_consec_nans:
            
            # If run is at beginning or end, we SKIP
            if start == 0 or end == len(arr):
                continue
            else:
                x = np.array([start - 1, end])
                arr[start:end] = np.interp(np.arange(start, end), x, arr[x])

    interp_mask = isnan & (~np.isnan(arr))
    return arr, interp_mask