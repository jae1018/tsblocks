import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# suppress empty mean warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")



def compute_bin_averages(
    # args related to the given dataframe
    df: pd.DataFrame,
    df_times: pd.DatetimeIndex,
    # bins times and idxs (related to df) calculated using
    # compute_time_bin_idxs
    bin_idxs: list[np.ndarray[np.int_]],
    bin_times: pd.DatetimeIndex,
    # kwargs
    columns: list[str] = None,
    periodic_columns: list[str] = None,
    keep_empty_times: bool = False,
    progress_bar: bool = True
    # make kwarg called stat_funcs that calculates stats?
    #stat_funcs = List[functions?]
) -> pd.DataFrame:
    
    """
    Computes averages over bins for a pandas DataFrame.

    Handles periodic columns (radians) by averaging via sin/cos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data to be averaged.
    df_times : pd.DatetimeIndex
        Time values aligned with rows of df.
    bin_idxs : list[np.ndarray]
        Each array contains integer indices into df.
    bin_times : pd.DatetimeIndex
        Times corresponding to each bin (used for resulting index).
    columns : list[str]
        Names of columns to average normally.
    periodic_columns : list[str], optional
        Names of columns to average as periodic angles (in radians).
    keep_empty_bins : bool, default False
        If False, drops bins that were completely empty (all NaNs).
    progress_bar : bool, default True
        If True, shows progress bar.

    Returns
    -------
    pd.DataFrame
        Averaged data indexed by bin_times (non-empty bins if keep_empty_bins=False).
    
    TO DO
    -----
    Make subsume this into general moments computing function? Not just mean?
    """
    
    if columns is None:
        columns = list(df)
    if periodic_columns is None:
        periodic_columns = []
    
    ###
    # The code to follow is broken into steps to be more interpretable
    ###
    
    ### step 1: extract columns to avg down as raw arrays for faster access
    # make arr from just columns (easy, non-periodic)
    arr = np.stack([ df[var].values for var in columns ]).T
    # make arr of sin(angle) cos(angle) for each angle in periodic_columns
    if len(periodic_columns) > 0:
        periodic_arr = np.hstack([
                            np.stack([ np.sin(df[var]), np.cos(df[var]) ]).T 
                            for var in periodic_columns 
        ])
    else:
        periodic_arr = np.empty((arr.shape[0],0), dtype=arr.dtype)
    # combine arr and periodic_arr along columns so that we can use
    # numpy's fast nanmean function with specified axis
    combined_arr = np.hstack([arr, periodic_arr])
    
    
    ### Step 2: avg data over each set of time idxs
    averaged_data = []
    iterable = tqdm(bin_idxs, desc="Averaging Down", disable=not progress_bar)
    for idxs in iterable:

        # take avg of data with nanmean (note that might get mean of
        # empty slices where no data was found!)
        averaged_data.append(
            np.nanmean( combined_arr[idxs], axis=0 )
        )
    
    
    ### Step 3: Convert list of averages into array
    # convert list of averaged data into array (non-periodic and periodic components)
    averaged_data = np.atleast_2d( np.array(averaged_data) )
    # if only one column, then can end up with (1, N) array - transpose it!
    if averaged_data.shape[0] == 1:
        averaged_data = averaged_data.T
    # extract averaged periodic cpomponents from array and convert back to angle
    averaged_angles = []
    for i in range(len(periodic_columns)):
        col_index = len(columns) + i*2
        averaged_angles.append(
            np.arctan2(
                averaged_data[:,col_index],   # sin part
                averaged_data[:,col_index+1]  # cos part
            )
        )
    # convert from list to array
    # arctans output goes from -pi to pi, so any results that are negative
    # should have 2*pi added to them to make it 0 to 2pi
    averaged_angles = np.atleast_2d( np.array(averaged_angles).T )#+ np.pi )
    neg_mask = averaged_angles < 0
    averaged_angles[neg_mask] = averaged_angles[neg_mask] + 2*np.pi
    # if only one column, then can end up with (1, N) array - transpose it!
    if averaged_angles.shape[0] == 1:
        averaged_angles = averaged_angles.T
    # if averaged_angles is empty array, reshape to get indexing right
    if averaged_angles.shape[0] == 0:
        averaged_angles = averaged_angles.reshape(averaged_data.shape[0], 0)
    # make new array just of averaged non-periodic data and averaged angles
    averaged_data = np.hstack([ averaged_data[:,:len(columns)], averaged_angles ])
    
    
    ### Step 4: Determine if masking out empty points from df
    # first need to remove empty times, if specified
    if not keep_empty_times:
        is_empty = np.isnan(averaged_data).all(axis=1)
        #is_empty = np.array([ len(idxs) == 0 for idxs in range(len(bin_idxs)-1) ])
    else:
        is_empty = np.full(averaged_data.shape[0], False)
    
    
    ### Step 5: Make dataframe with averaged data and bin_times as index
    averaged_df = pd.DataFrame(
                        averaged_data[~is_empty], 
                        index   = bin_times[~is_empty], 
                        columns = [ *columns, *periodic_columns ]
    )
    # set index name to time
    averaged_df.index.name = 'time'
    # reorder labels of df to align with the order of df
    cols = set(columns + periodic_columns)
    ordered_cols = [col for col in df.columns if col in cols]
    averaged_df = averaged_df[ordered_cols]
    
    return averaged_df






def compute_windowed_stats_from_input_idxs(
    input_dicts    : list[ dict[str, np.ndarray] ],
    selected_times : list[np.datetime64],
    df_inputs      : np.ndarray,
    stat_funcs     : dict[str, callable],
    input_windows  : list[str],
    input_columns  : list[str],
    progress_bar   : bool = True
) -> xr.Dataset:
    """
    Computes rolling stats from precomputed index slices.
    
    Parameters
    ----------
    input_dicts : list of dict
        Each dict maps window string -> np.ndarray of indices.
    selected_times : list of np.datetime64
        Times corresponding to end of each window.
    df_inputs : np.ndarray
        Raw input data matrix, shape (time, features).
    stat_funcs : dict
        e.g. {"mean": np.nanmean, "std": np.nanstd}
    input_windows : list of str
        List of window durations.
    input_columns : list of str
        Names of features.
    progress_bar : bool
        Whether to show progress bar.

    Returns
    -------
    xr.Dataset
        With dimension (time, stat, window, input_feature).
    """
    input_data = []
    iterable = zip(input_dicts, selected_times)
    iterable = tqdm(iterable, total=len(input_dicts), desc="Computing stats of input windows", 
                    disable=not progress_bar)

    for in_dict_idx, ts in iterable:
        
        ###
        # Since the indices were converted from numpy arrays of ints to
        # index ranges (e.g. going from [0,1,2,3,4,5] to (0,5)), the memory
        # usage is a LOT more manageable... but that comes at the cost of speed
        # (results are now a lot slower, but still tolerable). In the future,
        # might make a version that chunks the list of dictionaries so that
        # each chunk is converted into full index arrays and then those
        # chunks are processed. But this will do for now.
        ###
        
        ###
        # Compute stat blocks with block shape (stat, window, feature)
        ###
        stat_block = []
        for stat_name, func in stat_funcs.items():
            window_block = []
            
            # for each window (e.g "3min") of data ...
            for w in input_windows:
                
                # ... apply function to go from 2d array with shape
                # (num_idxs, num_input_feats) -> 1d array with 
                # shape (num_input_feats) ...
                #idx_range = in_dict_idx[w]
                stats = func(
                        ##df_inputs[ in_dict_idx[w] ], 
                        ##df_inputs[ idx_range[0] : idx_range[1]+1 ], 
                        df_inputs[ 
                            in_dict_idx[w][0] 
                              : 
                            in_dict_idx[w][1]+1 
                        ], 
                        ##df_inputs[ 
                        ##    np.arange(in_dict_idx[w][0], in_dict_idx[w][1]+1) 
                        ##], 
                        ##df_inputs[ in_dict_idx_arrs[w] ],
                        axis = 0
                )
                
                # ... and save that 1d array to list
                window_block.append(stats)
                
            # save block of same func applied over different windows
            # (list of 1d arrays -> effectively 2d array)
            stat_block.append(window_block)
            
        # shape (n_stats, n_windows, n_features)
        input_data.append(np.array(stat_block))

    # convert into single 4d numpy array
    input_data = np.array(input_data)

    # Build xarray
    ds = xr.Dataset(
        {
            "inputs": (("time", "stat", "window", "input_feature"), input_data)
        },
        coords={
            "time": selected_times,
            "stat": list(stat_funcs.keys()),
            "window": input_windows,
            "input_feature": input_columns
        }
    )
    
    # make attrs just for the 'inputs' dataarray
    ds.inputs.attrs = {
        "created_by": "TSBlocks.compute_windowed_input_stats",
        "input_windows": input_windows,
        #"freq": freq,
        "stat_funcs": list(stat_funcs.keys()),
        "input_columns": input_columns,
        "time_alignment": "Corresponds to end of each backward input window"
    }

    return ds





def compute_output_and_direct_from_idxs(
    output_idxs       : list[np.ndarray],
    direct_input_idxs : list[np.ndarray],
    selected_times    : list[np.datetime64],
    df_outputs        : np.ndarray,
    df_direct_inputs  : np.ndarray,
    capture_current   : bool = True,
    progress_bar      : bool = True
) -> xr.Dataset:
    """
    Extracts forecast output sequences and optionally direct input values
    at each reference time, then returns them as an xarray Dataset.

    For each time T (aligned with the end of input windows / start of forecast),
    this function:
      - collects the forecast output sequence over the specified output indices,
      - optionally collects direct inputs at T (e.g. positional data).

    Any time point is skipped if:
      - the output sequence contains NaNs, or
      - (if capture_current=True) the direct inputs at T contain NaNs.

    Parameters
    ----------
    output_idxs : list of np.ndarray
        List of integer index arrays specifying rows of df_outputs to extract 
        for each output sequence starting at time T.
    direct_input_idxs : list of np.ndarray
        List of integer index arrays specifying rows of df_direct_inputs to extract 
        as direct inputs at time T.
    selected_times : list of np.datetime64
        The reference times T, typically aligned with forecast start time.
    df_outputs : np.ndarray
        Array of shape (time, output_feature) containing all available output data.
    df_direct_inputs : np.ndarray
        Array of shape (time, direct_feature) containing all direct input data.
    capture_current : bool, default True
        Whether to extract direct inputs at time T. If False, only outputs are extracted.
    progress_bar : bool, default True
        Display a progress bar.

    Returns
    -------
    xr.Dataset
        Dataset containing:
          - outputs: (time, output_step, output_feature)
          - direct_inputs: (time, direct_feature), only present if capture_current=True
        Along with coordinates:
          - time: the forecast start times
          - output_step: step index into forecast sequence
          - output_feature: index (or variable) of output features
          - direct_feature: index (or variable) of direct input features, if present

    Notes
    -----
    - Any samples with NaNs in outputs (or direct inputs if applicable) are skipped.
    - Designed for use after slicing index computation (compute_forward_output_window_idxs).
    - Outputs are aligned on the forecast start time T, which typically equals
      the end of the input sequence plus the forecast lag.
    """
    output_data = []
    direct_data = []
    valid_times = []

    iterable = zip(direct_input_idxs, output_idxs, selected_times)
    iterable = tqdm(iterable, total=len(output_idxs), desc="Computing output windows",
                    disable=not progress_bar)

    for direct_idx, out_idx, ts in iterable:
        output_values = df_outputs[out_idx]
        if np.isnan(output_values).any():
            continue

        direct_values = df_direct_inputs[direct_idx] if capture_current else np.array([])
        if capture_current and np.isnan(direct_values).any():
            continue

        output_data.append(output_values)
        direct_data.append(direct_values)
        valid_times.append(ts)

    output_data = np.array(output_data)
    valid_times = np.array(valid_times, dtype='datetime64[ns]')

    # Prepare base dataset with outputs
    ds = xr.Dataset(
        {
            "outputs": (("time", "output_step", "output_feature"), output_data)
        },
        coords={
            "time": valid_times,
            "output_step": np.arange(output_data.shape[1]),
            "output_feature": np.arange(output_data.shape[2])  # adjust if you have names
        }
    )

    # Add metadata to outputs DataArray
    ds.outputs.attrs = {
        "created_by": "TSBlocks.compute_output_and_direct_from_idxs",
        "note": "Outputs computed using provided output indices. NaNs in outputs caused rows to be skipped."
    }

    # If using direct inputs
    if capture_current:
        direct_data = np.array(direct_data).reshape((len(direct_data), -1))

        ds["direct_inputs"] = (("time", "direct_feature"), direct_data)
        ds = ds.assign_coords({
            "direct_feature": np.arange(direct_data.shape[1])  # adjust if you have names
        })
        ds.direct_inputs.attrs = {
            "created_by": "TSBlocks.compute_output_and_direct_from_idxs",
            "note": "Direct inputs extracted at time T. Rows skipped if NaNs present."
        }

    # Add high-level metadata
    ds.attrs = {
        "capture_current": capture_current,
        "time_alignment": "time coordinate corresponds to start of forecast horizon (forecast_start_time).",
    }

    return ds




