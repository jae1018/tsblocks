import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import json
#from .slicing import create_input_output_window_idxs
#from .aggregation import compute_window_moments
#from .splits import index_blocks_by_duration
##from tsblocks import slicing
from . import slicing
from . import aggregation
from . import preprocessing
from . import splits



class TSBlocks:
    
    """
    
    TO DO
    -----
    1) Reintroduce just the computation of backward windows! Could then
       use that to have independent time alingment with a target (e.g.
       save those precomputed vars for later!)
    2) creating time series for time from 0 to T-1 for a time series then
       only time T for some other vars
       
       
    ----BIG to do----
    Maybe I've been approaching this wrong - instead of double checking
    data in case it's not contiguous in time, could just force it to be
    by insterting the missing times with NaNs as fill value? E.g.
      00:00, 00:01, --missing-- 00:04, 00:05
    would become
      00:00, 00:01, 00:02, 00:03, 00:04, 00:05
    """
    
    """
    def __init__(self, time_res='1min', max_gap='1min'):#, default_stats=('mean', 'std')):
        self.time_res = pd.Timedelta(time_res)
        self.max_gap  = pd.Timedelta(max_gap)
        #self.default_stats = default_stats
    """



    def avg_down(
            
        df: pd.DataFrame,
        df_times: pd.DatetimeIndex | np.ndarray[np.datetime64],
        columns: list[str] = None,
        periodic_columns: list[str] = None,
        freq: str = "1min",
        keep_empty_times: bool = False,
        progress_bar: bool = True
        
    ) -> pd.DataFrame:
        
        """
        High-level pipeline to downsample a time series dataframe.
    
        Calls compute_time_bin_idxs to determine bins,
        then compute_bin_averages_df to average data over those bins.
    
        ===!!!=== NEED TO WRITE THE TEST CASES FOR COMPUTE_BIN_AVGS ===!!!===
    
        Parameters
        ----------
        df : pd.DataFrame
            Your dataframe with time column and data.
        columns : list[str], optional
            Names of columns to average normally. If None, all except periodic_columns.
        periodic_columns : list[str], optional
            Names of columns to average using sin/cos (in radians).
        freq : str
            Bin frequency, e.g. "1min" or "5min".
        keep_empty_times : bool
            Whether to keep bins with no data (filled with NaNs).
        progress_bar : bool
            Whether to show tqdm progress bars.
    
        Returns
        -------
        pd.DataFrame
            Downsampled dataframe indexed by bin times.
        """

        # ensure df_times are pandas datetimes (easier to work with)
        df_times = pd.to_datetime(df_times)
        bin_idxs, bin_times = slicing.compute_time_bin_idxs(df_times, 
                                                            freq=freq)
    
        averaged_df = aggregation.compute_bin_averages(
                            df=df,
                            df_times=df_times,
                            bin_idxs=bin_idxs,
                            bin_times=bin_times,
                            columns=columns,
                            periodic_columns=periodic_columns,
                            keep_empty_times=keep_empty_times,
                            progress_bar=progress_bar
        )
    
        return averaged_df
    
    
    
    def compute_windowed_input_stats(
        df: pd.DataFrame,
        df_times: pd.DatetimeIndex,
        input_windows: list[str],
        freq: str = "1min",
        stat_funcs: dict[str, callable] = {"mean": np.nanmean},
        input_columns: list[str] = None,
        progress_bar: bool = True
    ) -> xr.Dataset:
        """
        Computes an xarray Dataset containing:
          - statistical summaries over multiple backward input windows.
    
        This is designed for cases where there is no direct target sequence
        or direct single-time inputs. It builds rolling backward index slices
        from the time series and computes statistics over each window for each feature.
    
        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed dataframe of features (shape: time x features).
        df_times : pd.DatetimeIndex
            Times corresponding to rows of df.
        input_windows : list of str
            Durations for backward input windows (e.g. ['3min', '6min']).
        freq : str, default '1min'
            Sampling frequency string, used to build uniform grid.
        stat_funcs : dict, default {"mean": np.nanmean}
            Mapping of statistic names to numpy functions to compute over input windows.
        input_columns : list of str, optional
            Columns to compute statistics on. Defaults to all columns.
        progress_bar : bool, default True
            Display a progress bar.
    
        Returns
        -------
        xr.Dataset
            With dimensions and variables:
            - inputs: (time, stat, window, input_feature)
            Also includes labeled coordinates like 'time', 'stat', 'window', etc.
    
        Notes
        -----
        - NaNs in the data are handled by the stat_funcs (like np.nanmean), they are not skipped.
        - Useful for preparing data for unsupervised or purely feature-driven ML pipelines.
        """
        if input_columns is None:
            input_columns = list(df.columns)
    
        # build uniform times
        uniform_times, _ = preprocessing.uniform_time_index(
                                df_times, 
                                freq = freq
        )
        # and reindex dataframe to uniform times
        df = df.reindex(uniform_times)
    
        # build backward slices
        input_dicts, selected_times = slicing.compute_rolling_backward_window_idxs(
            uniform_times = uniform_times,
            windows       = input_windows,
            freq          = freq,
            progress_bar  = progress_bar
        )
        
        #import pdb
        #pdb.set_trace()
    
        # Compute statistics over windows
        ds = aggregation.compute_windowed_stats_from_input_idxs(
            input_dicts    = input_dicts,
            selected_times = selected_times,
            df_inputs      = df[input_columns].values,
            stat_funcs     = stat_funcs,
            input_windows  = input_windows,
            input_columns  = input_columns,
            progress_bar   = progress_bar
        )
    
        return ds
    
    
    
    
    """
    def compute_windowed_input_stats_and_output_targets(
        df: pd.DataFrame,
        df_times: np.ndarray[np.datetime64] | pd.DatetimeIndex,
        input_windows: list[str],
        output_window: str = "1min",
        forecast_delta: str = "0min",
        freq: str = "1min",
        stat_funcs: dict[str, callable] = {"mean": np.nanmean},
        input_columns: list[str] = None,
        direct_input_columns: list[str] = None,
        output_columns: list[str] = None,
        capture_current: bool = True,
        progress_bar: bool = True
    ) -> xr.Dataset:
        ###
        Computes an xarray Dataset containing:
          - statistical summaries over multiple backward input windows,
          - optional direct input features at the current time (e.g. geometry or satellite),
          - and output sequences after a forecast delta.
    
        It builds all index slices internally from time series, computes the statistics
        over each window for each feature, organizes the data into an xarray Dataset 
        with labeled dimensions and coordinates.
    
        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed dataframe of features (shape: time x features).
        df_times : pd.DatetimeIndex
            Times corresponding to rows of df (does NOT need to be uniform).
        input_windows : list of str
            Durations for backward input windows (e.g. ['3min', '6min']).
        output_window : str, default '1min'
            Duration of the forward output sequence to predict.
        forecast_delta : str, default '0min'
            Delay after the current time T before output starts.
        freq : str, default '1min'
            Sampling frequency string, used to build uniform grid.
        stat_funcs : dict, default {"mean": np.nanmean}
            Mapping of statistic names to numpy functions to compute over input windows.
        input_columns : list of str, optional
            Columns to compute statistics on. Defaults to all columns.
        direct_input_columns : list of str, optional
            Columns to extract directly at time T (such as positional data).
        output_columns : list of str, optional
            Columns to predict as outputs. Defaults to all columns.
        capture_current : bool, default True
            Whether to include direct input values at T (produces direct_inputs).
        progress_bar : bool, default True
            Display a progress bar.
    
        Returns
        -------
        xr.Dataset
            With dimensions and variables:
            - inputs: (time, stat, window, input_feature)
            - outputs: (time, output_step, output_feature)
            - direct_inputs: (time, direct_feature), only present if direct_input_columns is given
            Also includes labeled coordinates like 'time', 'stat', 'window', etc.
    
        Notes
        -----
        - Rows with NaNs in the outputs (or direct inputs, if used) are skipped.
        - Useful for preparing data for machine learning pipelines that consume 
          rolling stats and future targets.
        ###
        if input_columns is None:
            input_columns = list(df.columns)
        if output_columns is None:
            output_columns = list(df.columns)
        if direct_input_columns is None:
            direct_input_columns = []
            
        # get uniform times here
        uniform_times, uniform_times_mask = preprocessing.uniform_time_index(
                                                            df_times, 
                                                            freq = freq
        )
    
        # ===============================
        # Build slicing indices
        # ===============================
        # note here that if capture_current = False, direct_input_idxs is
        # list of empty arrays!
        (
            input_dicts, 
            direct_input_idxs, 
            output_idxs,
            selected_times
        ) = \
            slicing.compute_expanding_input_output_idxs(
                            uniform_times   = uniform_times,
                            input_windows   = input_windows,
                            output_window   = output_window,
                            forecast_delta  = forecast_delta,
                            freq            = freq,
                            capture_current = capture_current,
                            progress_bar    = progress_bar
            )
            
    
        # extract raw numpy arrays for windowed_inputs (input_columns),
        # outputs (output_columns), and direct_inputs (direct_input_columns)
        df_inputs = df[input_columns].values
        df_outputs = df[output_columns].values
        df_direct_inputs = df[direct_input_columns].values
    
        input_data = []
        direct_data = []
        output_data = []
        valid_times = []
    
        iterable = zip(input_dicts, direct_input_idxs, output_idxs, selected_times)
        iterable = tqdm(iterable, total=len(input_dicts), desc="Computing features", disable=not progress_bar)
    
        for in_dict_idx, direct_idx, out_idx, ts in iterable:
            
            
            ### 
            # outputs must not have nans (make optional later?)
            ###
            output_values = df_outputs[out_idx]
            if np.isnan(output_values).any():
                continue
            
            
            ## require here that if input windows aren't "full" (e.g. 5min only
            ## has array of size < 5 [assuming at 1min freq]) then discard???
            ## TO DO
    
    
            ###
            # Compute stats: (stat, window, feature)
            ###
            stat_block = []
            for stat_name, func in stat_funcs.items():
                window_block = []
                
                # for each window (e.g "3min") of data ...
                for w in input_windows:
                    
                    # ... apply function to go from 2d array with shape
                    # (num_idxs, num_input_feats) -> 1d array with 
                    # shape (num_input_feats) ...
                    stats = func(
                            df_inputs[ in_dict_idx[w] ], 
                            axis=0
                    )
                    
                    # ... and save that 1d array to list
                    window_block.append(stats)
                    
                # save block of same func applied over different windows
                # (list of 1d arrays -> effectively 2d array)
                stat_block.append(window_block)
                
            # shape (n_stats, n_windows, n_features)
            stat_block = np.array(stat_block)
    
    
            ###
            # Direct input checks
            ###
            # calling [0] here b/c direct_idxs are arrays of only 1 int -
            # they're just meant to grab data corresponding to the index at
            # the current time, not multiple times!
            direct_values = df_direct_inputs[direct_idx]#[0]
            # if any nans in direct inputs, skip row
            if np.isnan(direct_values).any():
                continue
    
    
            ### 
            # save values to lists only if no continue conditions met earlier
            ###
            input_data.append(stat_block)
            direct_data.append(direct_values)
            output_data.append(output_values)
            valid_times.append(ts)
    
    
        ## Convert to arrays
        input_data = np.array(input_data)        # (time, stat, window, feature)
        output_data = np.array(output_data)      # (time, output_step, output_feature)
        direct_data = np.array(direct_data)      # (time, direct_feature)
        # direct_data here is special from other 3 arrays - if direct_data
        # is being used, then it has a singleton dimension owing from having
        # direct_idx being 1d singleton arrays (meaning it has a shape like
        # [num_legit_points, 1, num_direct_input_features]). HOWEVER, if 
        # direct_data is NOT used then it will have shape like 
        # (num_legit_points, 0, 0). So turn this 3d array into 2d using 
        # reshape (instead of manual indexing) since that works fine for 
        # dimensions with size 0.
        direct_data = direct_data.reshape( (direct_data.shape[0], direct_data.shape[2]) )
    
        ## Build xarray dataset of just inputs and outputs (for now)
        ds = xr.Dataset(
            {
                "inputs": (("time", "stat", "window", "input_feature"), input_data),
                "outputs": (("time", "output_step", "output_feature"), output_data),
            },
            coords={
                "time": valid_times,
                "stat": list(stat_funcs.keys()),
                "window": input_windows,
                "input_feature": input_columns,
                "output_step": np.arange(output_data.shape[1]),
                "output_feature": output_columns
            }
        )
    
        ## if using direct_inputs, also add that into dataset
        if capture_current:
            ds["direct_inputs"] = (("time", "direct_feature"), direct_data)
            ds = ds.assign_coords({"direct_feature": direct_input_columns})
            
        ## now also add attrs of keyword info used in the calling of this
        ## function to the created dataset
        ds.attrs = {
            "created_by": "TSBlocks.compute_windowed_input_stats_and_output_targets",
            "input_windows": input_windows,
            "output_window": output_window,
            "forecast_delta": forecast_delta,
            "freq": freq,
            "stat_funcs": list(stat_funcs.keys()),
            "capture_current": capture_current,
            "input_columns": input_columns,
            "direct_input_columns": direct_input_columns,
            "output_columns": output_columns,
            "time_alignment": "Corresponds to first element of output sequence (and direct_inputs)"
        }
    
        return ds
    """
    
    
    
    
    
    
    def compute_windowed_input_stats_and_output_targets(
        df: pd.DataFrame,
        df_times: pd.DatetimeIndex,
        input_windows: list[str],
        output_window: str = "1min",
        forecast_delta: str = "0min",
        freq: str = "1min",
        stat_funcs: dict[str, callable] = {"mean": np.nanmean},
        input_columns: list[str] = None,
        direct_input_columns: list[str] = None,
        output_columns: list[str] = None,
        capture_current: bool = True,
        progress_bar: bool = True
    ) -> xr.Dataset:
        """
        Computes an xarray Dataset containing:
          - statistical summaries over backward input windows,
          - direct inputs at T,
          - output sequences after a forecast delta.
          
        TO DO
        ------
        (1) make rejection of data with NaNs in outputs as optional - 
        and also create not-nan mask of inputs and outputs where True
        indices a defined value
        
        (2) also define what to do with nans in the inputs after stats
        are computed - leave them? ffill? what, exactly?
        """
    
        if input_columns is None:
            input_columns = list(df.columns)
        if output_columns is None:
            output_columns = list(df.columns)
        if direct_input_columns is None:
            direct_input_columns = []
    
        # ==================================================
        # (1) Compute rolling input stats
        # ==================================================
        inputs_ds = TSBlocks.compute_windowed_input_stats(
            df            = df,
            df_times      = df_times,
            input_windows = input_windows,
            freq          = freq,
            stat_funcs    = stat_funcs,
            input_columns = input_columns,
            progress_bar  = progress_bar
        )
    
        # ==================================================
        # (2) Compute output indices (+ direct input indices)
        # ==================================================
        # get uniform times
        uniform_times, _ = preprocessing.uniform_time_index(
                                            df_times, 
                                            freq = freq
        )
        # and reindex dataframe to uniform times
        df = df.reindex(uniform_times)
        # get idxs of direct input / outputs
        direct_input_idxs, output_idxs, selected_times_outputs = \
                slicing.compute_forward_output_window_idxs(
                            uniform_times   = uniform_times,
                            output_window   = output_window,
                            forecast_delta  = forecast_delta,
                            freq            = freq,
                            capture_current = capture_current,
                            progress_bar    = progress_bar
                )
    
        # ==================================================
        # (3) Extract outputs & direct inputs
        # ==================================================
        outputs_ds = aggregation.compute_output_and_direct_from_idxs(
            output_idxs       = output_idxs,
            df_outputs        = df[output_columns].values,
            capture_current   = capture_current,
            direct_input_idxs = direct_input_idxs,
            df_direct_inputs  = df[direct_input_columns].values,
            selected_times    = selected_times_outputs,
            progress_bar      = progress_bar
        )
    
        # ==================================================
        # (4) Align on intersection of valid times
        # ==================================================
        # xarray automatically aligns on time when merging
        common_times = np.intersect1d(inputs_ds.time.values, outputs_ds.time.values)
        
        inputs_ds   = inputs_ds.sel(time=common_times)
        outputs_ds  = outputs_ds.sel(time=common_times)
    
        # ==================================================
        # (5) Merge into final dataset
        # ==================================================
        ds = xr.merge([inputs_ds, outputs_ds])
        
        # add global attributes
        ds.attrs = {
            "created_by": "TSBlocks.compute_windowed_input_stats_and_output_targets",
            "input_windows": input_windows,
            "output_window": output_window,
            "forecast_delta": forecast_delta,
            "freq": freq,
            "stat_funcs": list(stat_funcs.keys()),
            "capture_current": str(capture_current),   # bool not supported in netcdf attrs when saving
            "input_columns": input_columns,
            "direct_input_columns": direct_input_columns,
            "output_columns": output_columns,
            "time_alignment": "time corresponds to end of input window (last observation before forecast)"
        }
    
        return ds
    
    
    
    
    
    
    def train_valid_test_split_by_forecast_start(
        dataset        : xr.Dataset,
        forecast_delta : str = None,
        output_window  : str = None,
        block_duration : str = "30d",
        train_frac     : float = 0.7,
        valid_frac     : float = 0.15,
        test_frac      : float = 0.15,
        seed           : int = None,
        shuffle_train  : bool = False,
        progress_bar   : bool = True
    ) -> tuple[
        xr.Dataset, 
        xr.Dataset, 
        xr.Dataset
    ]:
        """
        Splits an already-computed dataset of inputs, outputs, and direct_inputs
        (from compute_windowed_input_stats_and_output_targets) into train, valid, 
        and test sets by creating time blocks on the forecast start time.
    
        The forecast start time is defined as:
            forecast_start_time = time + forecast_delta
        where `time` is the end of the input window (the last observed input point).
    
        To ensure each forecast horizon is fully contained within its block,
        the block creation uses `buffer_duration = output_window`. This trims
        the last portion of each block to avoid forecast sequences from extending 
        into the next block, preventing leakage.
    
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset with dimension 'time' that represents the end of each input window.
        forecast_delta : str, optional
            The forecast delta originally used to build the dataset (e.g. '3h').
            If None, attempts to load from dataset.attrs['forecast_delta'].
        output_window : str, optional
            The length of the forecast horizon (e.g. '1h').
            If None, attempts to load from dataset.attrs['output_window'].
        block_duration : str
            Duration of each block for splitting, e.g. '30d'.
        train_frac, valid_frac, test_frac : float
            Fractions of total blocks assigned to train, valid, and test sets.
        seed : int, optional
            Random seed for reproducibility in shuffling blocks.
        shuffle_train : bool, default False
            Train blocks are already randomly selected. Setting this to True
            shuffle *within* each train block as well (good for FFNNs, but
            bad for RNN variants).
        progress_bar : bool, default True
            Whether to show progress bars.
    
        Returns
        -------
        train_ds, valid_ds, test_ds : xr.Dataset
            The split datasets with the same structure as input, split by forecast start times.
    
        Notes
        -----
        - This function ensures that each forecast start time and its entire output
          sequence belong to a single block, avoiding overlap into other splits.
        - Rolling input windows are allowed to reach backward across block edges,
          which is standard practice and does not cause leakage.
        - Splits are performed on forecast start times, meaning the start of the 
          predicted sequence.
    
        Raises
        ------
        ValueError
            If forecast_delta or output_window is not provided and cannot be inferred
            from dataset.attrs.
        """
        # check that train, valid, test fracs add up to 1
        assert np.abs(np.sum(train_frac + valid_frac + test_frac) - 1) < 1e-6
        
        # ====================================================
        # (1) Infer forecast_delta and output window from attrs
        # ====================================================
        if forecast_delta is None:
            if "forecast_delta" not in dataset.attrs:
                raise ValueError(
                    "forecast_delta not provided and not found in dataset.attrs. "
                    "Please either pass it explicitly or ensure your dataset was "
                    "created by TSBlocks orchestration functions that add this attribute."
                )
            forecast_delta = dataset.attrs["forecast_delta"]
        forecast_delta_td = pd.Timedelta(forecast_delta)
        
        # (2) Infer output_window for buffer
        if output_window is None:
            if "output_window" not in dataset.attrs:
                raise ValueError("output_window not provided and not found in dataset.attrs.")
            output_window = dataset.attrs["output_window"]
        output_window_td = pd.Timedelta(output_window)
    
        # ==================================================
        # (2) Compute forecast start times
        # ==================================================
        forecast_start_times = dataset.time.values + np.timedelta64(forecast_delta_td)
    
        # ==================================================
        # (3) Build blocks
        # ==================================================
        blocks = splits.index_blocks_by_duration(
            times           = forecast_start_times,
            block_duration  = block_duration,
            buffer_duration = output_window_td,
            progress_bar    = progress_bar
        )
        block_keys = list(blocks.keys())
    
        # ==================================================
        # (4) Shuffle and split at block level
        # ==================================================
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(block_keys)
    
        # get number of blocks for train, vlaid, test
        n_blocks = len(block_keys)
        n_train = int(train_frac * n_blocks)
        n_valid = int(valid_frac * n_blocks)
        #n_test  = n_blocks - n_train - n_valid
    
        # assign train, valid, test blocks
        train_blocks = block_keys[:n_train]
        valid_blocks = block_keys[n_train:n_train+n_valid]
        test_blocks  = block_keys[n_train+n_valid:]
        
        # create metadata for blocks
        block_splits = {
            "train": train_blocks,
            "valid": valid_blocks,
            "test":  test_blocks,
        }
        block_metadata_by_split = {}
        for split_name, block_list in block_splits.items():
            metadata_list = []
            for b in block_list:
                block_times = dataset.time.isel(time=blocks[b])
                metadata_list.append({
                    "block": b,
                    "start": pd.to_datetime(block_times.min().item()).isoformat(),
                    "end":   pd.to_datetime(block_times.max().item()).isoformat(),
                })
            block_metadata_by_split[split_name] = metadata_list
    
        # ==================================================
        # (5) Extract datasets by time indexing
        # ==================================================
        def build_subset(block_list):
            indices = np.concatenate([blocks[b] for b in block_list])
            indices.sort()
            # this .copy() could be expensive if the datset is very big...
            return dataset.isel(time=indices).copy()
    
        # build train, valid, test datasets based on blocks
        train_ds = build_subset(train_blocks)
        valid_ds = build_subset(valid_blocks)
        test_ds  = build_subset(test_blocks)
        
        # assign attr detailing if dataset is train, valid, or test
        # as well as block info 
        for ds, split in zip([train_ds, valid_ds, test_ds],['train','valid','test']):
            ds.attrs['data_split'] = split
            ds.attrs["split_blocks"] = json.dumps(block_metadata_by_split[split])
            
        # also shuffle train dataset if requested
        if shuffle_train:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(train_ds.sizes["time"])
            train_ds = train_ds.isel(time=perm)
    
        # ==================================================
        # (6) Print summary
        # ==================================================
        print(f"\n[TSBlocks] Split by forecast start time: "
              f"{len(train_ds.time)} train, {len(valid_ds.time)} valid, {len(test_ds.time)} test samples "
              f"({len(train_blocks)} blocks train, {len(valid_blocks)} valid, {len(test_blocks)} test).")
    
        return train_ds, valid_ds, test_ds
    
