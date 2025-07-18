import numpy as np
import pandas as pd
import xarray as xr
import pytest
from tsblocks import TSBlocks



class TestComputeInputStatsAndOutputTargets:
    
    """
    Tests the full pipeline function that computes input window stats,
    direct inputs, and output sequences, returning them in an xarray Dataset.
    """



    def test_basic_structure(self):
        """
        Checks that the function returns an xarray Dataset with the expected 
        dimensions and variables on a tiny dataset with one input and one output feature.
        """
        times = pd.date_range("2025-01-01", periods=10, freq="1min")
        df = pd.DataFrame({
            "feature1": np.arange(10),
            "sat_x": np.linspace(1000, 1010, 10),
            "sat_y": np.linspace(1000, 1010, 10),
            "output": np.linspace(50, 60, 10)
        }, index=times)

        ds = TSBlocks.compute_windowed_input_stats_and_output_targets(
            df=df,
            df_times=times,
            input_windows=["3min"],
            output_window="2min",
            forecast_lag="1min",
            freq="1min",
            stat_funcs={"mean": np.nanmean},
            input_columns=["feature1"],
            direct_input_columns=["sat_x", "sat_y"],
            output_columns=["output"],
            capture_current=True,
            progress_bar=False
        )

        # check dataset contents
        assert isinstance(ds, xr.Dataset)
        assert "inputs" in ds and "outputs" in ds and "direct_inputs" in ds

        # check dimensions are present
        assert "time" in ds.dims
        assert "stat" in ds.dims
        assert "window" in ds.dims
        assert "input_feature" in ds.dims
        assert "output_step" in ds.dims
        assert "output_feature" in ds.dims
        assert "direct_feature" in ds.dims

        # check shapes
        n_time = ds.dims["time"]
        assert ds.inputs.shape == (n_time, 1, 1, 1)
        assert ds.outputs.shape[0] == n_time
        assert ds.direct_inputs.shape[0] == n_time
        #raise ValueError


    def test_multiple_windows_and_stats(self):
        """
        Checks multiple input windows and multiple statistics produce the 
        correct expanded shape.
        """
        times = pd.date_range("2025-01-01", periods=12, freq="1min")
        df = pd.DataFrame({
            "feature1": np.random.normal(0,1,12),
            "feature2": np.random.normal(5,2,12),
            "feature3": np.random.normal(5,2,12),
            "feature4": np.random.normal(5,2,12),
            "output": np.random.normal(10,3,12)
        }, index=times)

        ds = TSBlocks.compute_windowed_input_stats_and_output_targets(
            df=df,
            df_times=times,
            input_windows=["3min", "4min", "5min"],
            output_window="2min",
            forecast_lag="1min",
            freq="1min",
            stat_funcs={"mean": np.nanmean, "std": np.nanstd},
            input_columns=["feature1", "feature2", "feature3", "feature4"],
            direct_input_columns=None,
            output_columns=["output"],
            capture_current=False,
            progress_bar=False
        )

        # Check shape: (time, stat, window, input_feature)
        # 2 stats, 3 windows, 4 input features
        assert ds.inputs.shape[1:] == (2, 3, 4)  



    def test_skips_if_output_has_nan(self):
        """
        Checks that rows with NaNs in the outputs are skipped.
        """
        times = pd.date_range("2025-01-01", periods=8, freq="1min")
        df = pd.DataFrame({
            "feature": np.arange(8),
            "output": np.linspace(10,15,8)
        }, index=times)

        # inject NaN in outputs
        df.loc[times[5],"output"] = np.nan

        ds = TSBlocks.compute_windowed_input_stats_and_output_targets(
            df=df,
            df_times=times,
            input_windows=["3min"],
            output_window="2min",
            forecast_lag="1min",
            freq="1min",
            stat_funcs={"mean": np.nanmean},
            input_columns=["feature"],
            output_columns=["output"],
            capture_current=False,
            progress_bar=False
        )

        # verify that the length is less than full length due to NaN
        assert len(ds.time) < len(times)



    def test_skips_if_direct_has_nan(self):
        """
        Checks that rows with NaNs in direct input columns are skipped.
        """
        times = pd.date_range("2025-01-01", periods=8, freq="1min")
        df = pd.DataFrame({
            "feature": np.arange(8),
            "sat_x": np.linspace(1000,1010,8),
            "output": np.linspace(10,15,8)
        }, index=times)

        # introduce NaN in sat_x
        df.loc[times[2],"sat_x"] = np.nan

        ds = TSBlocks.compute_windowed_input_stats_and_output_targets(
            df=df,
            df_times=times,
            input_windows=["3min"],
            output_window="2min",
            forecast_lag="1min",
            freq="1min",
            stat_funcs={"mean": np.nanmean},
            input_columns=["feature"],
            direct_input_columns=["sat_x"],
            output_columns=["output"],
            capture_current=True,
            progress_bar=False
        )

        # Should skip rows where direct input has NaN
        assert len(ds.time) < len(times)