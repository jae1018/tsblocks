import numpy as np
import pandas as pd
from tqdm import tqdm




def compute_rolling_backward_window_idxs(
    uniform_times : np.ndarray[np.datetime64] | pd.DatetimeIndex,
    windows       : list[str],
    freq          : str = "1min",
    progress_bar  : bool = True
) -> tuple[
    list[ dict[str, np.ndarray] ], 
    pd.DatetimeIndex
]:
    """
    For each time T in a uniformly spaced time series, compute backward-looking
    indices for multiple windows ending at T. Returns indices aligned with the end
    of each input window.
    
    TO DO
    -----
    Add option to only build rolling window if window is "full" (e.g. not
    partial)... or maybe that should be checked in funciton calling this?

    Parameters
    ----------
    uniform_times : np.ndarray[np.datetime64] or pd.DatetimeIndex
        Uniformly spaced time array.
    windows : list of str
        List of window durations, e.g. ['3min', '6min'].
    freq : str
        Frequency of the time series sampling (e.g. '1min').
    progress_bar : bool
        Whether to show a progress bar.

    Returns
    -------
    input_dicts : list of dict
        Each dict maps window string to np.ndarray *RANGE* of indices 
        covering [T - window, T] using tuples of the form (T - window, T)
    selected_times : pd.DatetimeIndex
        Times aligned with the end of each window sequence.
    """
    uniform_times = pd.to_datetime(uniform_times)
    deltas = np.diff(uniform_times)
    assert np.all(deltas == deltas[0]), \
        f"Time series is not uniformly spaced. Found unique deltas: {np.unique(deltas)}"

    step_td = pd.Timedelta(freq)
    window_steps = { w: int(pd.Timedelta(w) / step_td) for w in windows }

    input_dicts = []
    selected_times = []

    iterable = tqdm(range(len(uniform_times)), desc='Building rolling input indices',
                    disable=not progress_bar)
    for i in iterable:
        input_dict = {}
        #all_full = True
        for w, steps in window_steps.items():
            start_idx = max(0, i - (steps - 1))
            #start_idx = i - (steps - 1)
            #if start_idx < 0:
            #    all_full = False
            #    break
            ##input_dict[w] = np.arange(start_idx, i+1)
            input_dict[w] = (start_idx, i)

        #if not all_full:
        #    continue

        input_dicts.append(input_dict)
        selected_times.append(uniform_times[i])

    return input_dicts, pd.to_datetime(selected_times)




def compute_expanding_input_output_idxs(
    uniform_times   : np.ndarray[np.datetime64] | pd.DatetimeIndex,
    input_windows   : list[str],
    output_window   : str = "1min",
    forecast_delta  : str = "0min",
    freq            : str = "1min",
    capture_current : bool = False,
    progress_bar    : bool = True
) -> tuple[  
        list[ dict[str, np.ndarray] ],    # input sequences for history stats up to time T-1
        list[np.ndarray],                 # direct indices at time T (positional)
        list[np.ndarray],                 # output sequences (starting at time T + forecast_delta)
        pd.DatetimeIndex                  # times actually used
]:
    """
    Need to reogrnaize this - will instead define time T to be at end of
    input sequence
    vvvv
    For each time T in the series, computes:
      - a dict mapping each input_window string to backward indices from
        T - {window_size} - forecast_delta - 1 up to (and including) 
        time T - forecast_delta - 1
      - optionally the index just at T (for e.g. positional / direct
        inputs - list of empty arrays if capture_current = False)
      - indices for the output window starting at time T
      - the times of legitimate points (indexed relative to the start of
        the output sequence)

    Assumes that times is uniformly spaced (and raises error if not the case)
    
    NOTE: FORECAST IS TO BE INTERPRETED HERE AS THE LENGTH OF TIME (BEYOND
            FREQ, I.E. FREQ + 1) BETWEEN THE END OF THE INPUT SEQUENCE
            AND THE START OF THE OUTPUT SEQUENCE. IF CAPTURE_CURRENT=TRUE,
            THE DIRECT INDICES AT TIME T ARE THE SAME INDICES OCCURING AT THE
            START OF THE OUTPUT SEQUENCES!
            
    TO DO
    -----
    Give this func optional arg that mandates only keeping idxs where "full"
    window duration was accounted for? E.g. If given "3min" and "5min"
    and 3min idxs are [1,2,3] and 5min ard [0,1,2,3], then 5min would not be
    considered "full" and thus ignore... although perhaps this should be
    applied at a higher level function?
          

    Parameters
    ----------
    uniform_times : np.ndarray[np.datetime64] or pd.DatetimeIndex
        Uniformly spaced time array.
    input_windows : list of str
        E.g. ['3min', '6min', '12min']
    output_window : str
        Length of forward output window.
    forecast_delta : str
        Delay before starting the output window after end of input sequence
    freq : str
        Sampling frequency of the time series, e.g. '1min'.
    capture_current : bool
        If True, also returns list of np.array([T]) indices for direct inputs.
    progress_bar : bool

    Returns
    -------
    input_dicts : list[dict]
        For each time T, dict mapping input_window -> np.ndarray of indices.
    direct_input_idxs : list[np.ndarray]
        Each is array([T]) if capture_current=True, else None.
    output_idxs : list[np.ndarray]
        For each T, indices for output window.
    selected_times : pd.DatetimeIndex
        Times corresponding to start of each output sequence (some times skipped).

    Raises
    ------
    AssertionError if `uniform_times` is not evenly spaced.
    """
    uniform_times = pd.to_datetime(uniform_times)
    deltas = np.diff(uniform_times)
    assert np.all(deltas == deltas[0]), \
        f"Time series is not uniformly spaced. Found unique deltas: {np.unique(deltas)}"

    step_td = pd.Timedelta(freq)
    window_steps = { w: int(pd.Timedelta(w) / step_td) for w in input_windows }
    delta_steps = int(pd.Timedelta(forecast_delta) / step_td)
    out_steps = int(pd.Timedelta(output_window) / step_td)

    input_dicts = []
    direct_input_idxs = []
    output_idxs = []
    selected_times = []

    iterable = tqdm(range(len(uniform_times)), desc='Building input-output windows', 
                    disable=not progress_bar)
    for i in iterable:
        input_dict = {}
        for w, steps in window_steps.items():
            start_idx = max(0, i - (steps-1))
            input_dict[w] = np.arange(start_idx, i+1)

        start_out_idx = i + delta_steps + 1
        end_out_idx = start_out_idx + out_steps
        if end_out_idx > len(uniform_times):
            continue

        input_dicts.append(input_dict)
        output_idxs.append(np.arange(start_out_idx, end_out_idx))
        direct_input_idxs.append(
            np.array([start_out_idx]) if capture_current else np.array([], dtype=np.int_)
        )
        selected_times.append(uniform_times[start_out_idx])

    return input_dicts, direct_input_idxs, output_idxs, pd.to_datetime(selected_times)





def compute_forward_output_window_idxs(
    uniform_times   : np.ndarray[np.datetime64] | pd.DatetimeIndex,
    output_window   : str = "1min",
    forecast_delta  : str = "0min",
    freq            : str = "1min",
    capture_current : bool = False,
    progress_bar    : bool = True
) -> tuple[
        list[np.ndarray],  # direct indices at time T (positional)
        list[np.ndarray],  # output sequences (starting at time T + forecast_delta)
        pd.DatetimeIndex   # times aligned with end of input / direct input
]:
    """
    For each time T in the series, computes:
      - optionally the index just at T (for direct inputs if capture_current=True),
      - indices for the output window starting at time T + forecast_delta,
      - times corresponding to the *reference time* (aligned with T, end of input).

    Parameters
    ----------
    uniform_times : np.ndarray[np.datetime64] or pd.DatetimeIndex
        Uniformly spaced time array.
    output_window : str
        Length of forward output window.
    forecast_delta : str
        Delay before starting the output window after time T.
    freq : str
        Sampling frequency of the time series, e.g. '1min'.
    capture_current : bool
        If True, also returns list of np.array([T]) indices for direct inputs.
    progress_bar : bool
        Whether to display tqdm progress bar.

    Returns
    -------
    direct_input_idxs : list[np.ndarray]
        Each is array([T]) if capture_current=True, else empty array.
    output_idxs : list[np.ndarray]
        For each T, indices for output window.
    selected_times : pd.DatetimeIndex
        Times corresponding to the *reference time T* (the end of input / point of prediction).

    Raises
    ------
    AssertionError if `uniform_times` is not evenly spaced.
    """
    uniform_times = pd.to_datetime(uniform_times)
    deltas = np.diff(uniform_times)
    assert np.all(deltas == deltas[0]), \
        f"Time series is not uniformly spaced. Found unique deltas: {np.unique(deltas)}"

    step_td = pd.Timedelta(freq)
    delta_steps = int(pd.Timedelta(forecast_delta) / step_td)
    out_steps = int(pd.Timedelta(output_window) / step_td)

    direct_input_idxs = []
    output_idxs = []
    selected_times = []

    iterable = tqdm(range(len(uniform_times)), desc='Building rolling output indices', 
                    disable=not progress_bar)
    for i in iterable:
        start_out_idx = i + delta_steps + 1
        end_out_idx = start_out_idx + out_steps
        if end_out_idx > len(uniform_times):
            continue

        output_idxs.append(np.arange(start_out_idx, end_out_idx))
        direct_input_idxs.append(
            np.array([start_out_idx]) if capture_current else np.array([], dtype=np.int_)
        )
        selected_times.append(uniform_times[i])

    return direct_input_idxs, output_idxs, pd.to_datetime(selected_times)

