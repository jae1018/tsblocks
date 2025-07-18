import numpy as np
import pandas as pd
from tqdm import tqdm



def index_blocks_by_duration(
            times           : np.ndarray[np.datetime64] | pd.DatetimeIndex,
            block_duration  : str = '30d', 
            buffer_duration : str = None,
            progress_bar    : bool = True
) -> dict[int, np.ndarray]:
    """
    Given a Pandas Series of datetime values, returns a dict mapping
    block numbers to arrays of row indices corresponding to each time block
    of arbitrary duration (e.g. '35d'), with an optional buffer trimmed from
    the end of each block. Note that this cutoff is imposed with > ! If its
    exactly equal to buffer_duration, the point is kept.

    Parameters:
        datetime_series (pd.Series or array-like): Series of datetime64[ns]
        block_duration (str): duration of each block, e.g. '35d', '6h'
        buffer_duration (str): duration to exclude from end of each block.
        progress_var (bool): If True, uses tqdm bar to print progress

    Returns:
        dict[int, np.ndarray]: keys are block numbers, values are index arrays
    """
    times = pd.Series( pd.to_datetime(times) )
    
    # Sort times (keep original indices to map back)
    sorted_idx = np.argsort(times.values)
    sorted_times = times.iloc[sorted_idx].reset_index(drop=True)

    block_to_indices = {}
    block_num = 0
    n = len(sorted_times)

    block_timedelta = pd.Timedelta(block_duration)

    # Precompute block start times (so that we can use for loop later instead)
    block_starts = []
    idx = 0
    while idx < n:
        block_starts.append(sorted_times.iloc[idx])
        next_time = sorted_times.iloc[idx] + block_timedelta
        idx = np.searchsorted(sorted_times, next_time, side='left')
    
    # Now iterate over with for loop (and we can monitor progress with tqdm
    # if specifie3d)
    iterable = tqdm(block_starts, desc='Breaking time series into blocks') \
               if progress_bar else block_starts
    for block_num, start_time in enumerate( iterable ):
        end_time = start_time + block_timedelta
    
        # Find all indices in this block
        in_block_mask = (sorted_times >= start_time) & (sorted_times < end_time)
        block_idx_in_sorted = np.where(in_block_mask)[0]
    
        # If buffer_duration not None, need to cutoff some times before the end
        if buffer_duration is not None:
            buffer_cutoff = end_time - pd.Timedelta(buffer_duration)
            keep_mask = sorted_times.iloc[block_idx_in_sorted] <= buffer_cutoff
            block_idx_in_sorted = block_idx_in_sorted[keep_mask.values]
    
        # Map back to original indices
        original_indices = sorted_idx[block_idx_in_sorted]
        block_to_indices[block_num] = original_indices

    return block_to_indices