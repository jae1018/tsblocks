import numpy as np
import pandas as pd
import pandas.testing as pdt

from tsblocks.slicing import compute_time_bin_idxs
from tsblocks.aggregation import compute_bin_averages

# some functions use means of empty slices, just ignore those warnings
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")


class TestComputeBinAverages:
    """
    Tests compute_bin_averages from aggregation.py using bins produced by compute_time_bin_idxs.
    """



    def test_noPeriodic_noNans_singleColumn_noEmptyTimes(self):
        df = pd.DataFrame({
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        df_times = pd.date_range("2025-01-01", periods=df.shape[0], freq="1min")
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value"],
            periodic_columns=[],
            keep_empty_times=True,
            progress_bar=False
        )
        
        expected_means = [
            np.mean([1.0, ]),   # 00:00
            np.mean([2.0, 3.0]),   # 00:02
            np.mean([4.0, 5.0])         # 00:04
        ]
        expected_df = pd.DataFrame({'value':expected_means}, index=bin_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
    
    
    
    def test_noPeriodic_Nans_singleColumn_noEmptyTimes(self):
        df = pd.DataFrame({
            "value": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0],
        })
        df_times = pd.date_range("2025-01-01", periods=df.shape[0], freq="1min")
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value"],
            periodic_columns=[],
            keep_empty_times=True,
            progress_bar=False
        )
        
        expected_means = [
            np.mean([1.0, ]),            # 00:00
            np.mean([2.0, 3.0]),         # 00:02
            np.nanmean([np.nan, 4.0]),   # 00:04
            np.mean([5.0])               # 00:06
        ]
        expected_df = pd.DataFrame({'value':expected_means}, index=bin_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
        
        
    
    def test_noPeriodic_Nans_singleColumn_emptyTimes(self):
        df = pd.DataFrame({
            "value": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0],
        })
        df_times_1 = pd.date_range("2025-01-01 00", periods=df.shape[0]//2, freq="1min").tolist()
        df_times_2 = pd.date_range("2025-01-01 01", periods=df.shape[0]//2, freq="1min").tolist()
        df_times = pd.to_datetime(df_times_1 + df_times_2)
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value"],
            periodic_columns=[],
            keep_empty_times=True,
            progress_bar=False
        )
        
        expected_means = [
            np.mean([1.0]),           # 00:00
            np.mean([2.0, 3.0]),      # 00:02
            np.nanmean([np.nan]),     # 00:04
            *np.full(27, np.nan),     # (time in-between)
            np.nanmean([4.0]),        # 01:00
            np.mean([5.0, 6.0]),      # 01:02
            np.mean([7.0])            # 01:04
        ]
        expected_df = pd.DataFrame({'value':expected_means}, index=bin_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
        
    
    
    def test_noPeriodic_Nans_multiColumn_emptyTimes(self):
        df = pd.DataFrame({
            "value1": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0],
            "value2": np.array([1.0, 2.0, 3.0, np.nan, np.nan, 5.0, 6.0, 7.0])+10,
        })
        df_times_1 = pd.date_range("2025-01-01 00", periods=df.shape[0]//2, freq="1min").tolist()
        df_times_2 = pd.date_range("2025-01-01 01", periods=df.shape[0]//2, freq="1min").tolist()
        df_times = pd.to_datetime(df_times_1 + df_times_2)
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value1", "value2"],
            periodic_columns=[],
            keep_empty_times=True,
            progress_bar=False
        )
        
        # value1 means
        expected_value1_means = np.array([
            np.mean([1.0]),           # 00:00
            np.mean([2.0, 3.0]),      # 00:02
            np.nanmean([np.nan]),     # 00:04
            *np.full(27, np.nan),     # (time in-between)
            np.nanmean([4.0]),        # 01:00
            np.mean([5.0, 6.0]),      # 01:02
            np.mean([7.0])            # 01:04
        ])  
        # value2 means
        expected_value2_means = 10+np.array([
            np.mean([1.0]),           # 00:00
            np.mean([2.0, 3.0]),      # 00:02
            np.nanmean([np.nan]),     # 00:04
            *np.full(27, np.nan),     # (time in-between)
            np.nanmean([np.nan]),     # 01:00
            np.mean([5.0, 6.0]),      # 01:02
            np.mean([7.0])            # 01:04
        ])
        expected_df = pd.DataFrame({'value1':expected_value1_means,
                                    'value2':expected_value2_means}, 
                                   index=bin_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
    
    
    
    def test_singlePeriodic_Nans_singleColumn_emptyTimes(self):
        df = pd.DataFrame({
            "periodic": [0.0, np.pi/2, np.pi, np.nan, 0, np.pi/2, np.pi, 3*np.pi/2],
            "value": np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0]),
        })
        df_times_1 = pd.date_range("2025-01-01 00", periods=df.shape[0]//2, freq="1min").tolist()
        df_times_2 = pd.date_range("2025-01-01 01", periods=df.shape[0]//2, freq="1min").tolist()
        df_times = pd.to_datetime(df_times_1 + df_times_2)
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value"],
            periodic_columns=["periodic"],
            keep_empty_times=True,
            progress_bar=False
        )
        
        # periodic means
        expected_periodic_means = np.array([
            np.mean([0.0]),                # 00:00
            np.mean([np.pi/2, np.pi]),     # 00:02
            np.nanmean([np.nan]),          # 00:04
            *np.full(27, np.nan),          # (time in-between)
            np.nanmean([0.0]),             # 01:00
            np.mean([np.pi/2, np.pi]),     # 01:02
            np.mean([3*np.pi/2])           # 01:04
        ])  
        # value means
        expected_value_means = np.array([
            np.mean([1.0]),           # 00:00
            np.mean([2.0, 3.0]),      # 00:02
            np.nanmean([np.nan]),     # 00:04
            *np.full(27, np.nan),     # (time in-between)
            np.nanmean([4.0]),        # 01:00
            np.mean([5.0, 6.0]),      # 01:02
            np.mean([7.0])            # 01:04
        ])
        expected_df = pd.DataFrame({'periodic':expected_periodic_means,
                                    'value':expected_value_means}, 
                                   index=bin_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
    
    
    
    def test_singlePeriodic_Nans_singleColumn_noEmptyTimes(self):
        df = pd.DataFrame({
            "periodic": [0.0, np.pi/2, np.pi, np.nan, 0, np.pi/2, np.pi, 3*np.pi/2],
            "value": np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0]),
        })
        df_times_1 = pd.date_range("2025-01-01 00", periods=df.shape[0]//2, freq="1min").tolist()
        df_times_2 = pd.date_range("2025-01-01 01", periods=df.shape[0]//2, freq="1min").tolist()
        df_times = pd.to_datetime(df_times_1 + df_times_2)
        bins, bin_times = compute_time_bin_idxs(df_times, freq="2min")

        out_df = compute_bin_averages(
            df=df,
            df_times=df_times,
            bin_idxs=bins,
            bin_times=bin_times,
            columns=["value"],
            periodic_columns=["periodic"],
            keep_empty_times=False,
            progress_bar=False
        )
        
        # periodic means
        expected_periodic_means = np.array([
            np.mean([0.0]),                # 00:00
            np.mean([np.pi/2, np.pi]),     # 00:02
            #np.nanmean([np.nan]),          # 00:04
            #*np.full(27, np.nan),          # (time in-between)
            np.nanmean([0.0]),             # 01:00
            np.mean([np.pi/2, np.pi]),     # 01:02
            np.mean([3*np.pi/2])           # 01:04
        ])  
        # value means
        expected_value_means = np.array([
            np.mean([1.0]),           # 00:00
            np.mean([2.0, 3.0]),      # 00:02
            #np.nanmean([np.nan]),     # 00:04
            #*np.full(27, np.nan),     # (time in-between)
            np.nanmean([4.0]),        # 01:00
            np.mean([5.0, 6.0]),      # 01:02
            np.mean([7.0])            # 01:04
        ])
        new_times = pd.to_datetime([
            "2025-01-01 00:00",
            "2025-01-01 00:02",
            "2025-01-01 01:00",
            "2025-01-01 01:02",
            "2025-01-01 01:04",
        ])
        expected_df = pd.DataFrame({'periodic':expected_periodic_means,
                                    'value':expected_value_means},
                                   index=new_times)
        expected_df.index.name = 'time'
        
        pdt.assert_frame_equal(expected_df, out_df)
