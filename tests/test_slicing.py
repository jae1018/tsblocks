import numpy as np
import pandas as pd
import pytest

from tsblocks.slicing import create_input_output_window_idxs
from tsblocks.slicing import get_valid_window_starts
from tsblocks.slicing import compute_time_bin_idxs
from tsblocks.slicing import compute_rolling_backward_window_idxs
from tsblocks.slicing import compute_expanding_input_output_idxs



class TestGetValidWindowStarts:
    """
    All tests for get_valid_window_starts function from slicing.py
    """



    def test_finds_valid_windows_basic(self):
        """
        Test that with a perfectly regular time series, all possible start indices
        are accepted, producing expected number of windows.
        """
        # 60 time steps of 1 min spacing
        time_array = pd.date_range("2025-01-01", periods=7, freq="1min")

        windows = get_valid_window_starts(
            time_array=time_array,
            window_duration='5min',
            time_res='1min',
            max_gap='1min',
            progress_bar = False
        )

        # windows should look like
        expected_windows = np.array([
            np.arange(5),
            np.arange(5)+1,
            np.arange(5)+2
        ])
        
        assert np.array_equal(windows, expected_windows)



    def test_skips_windows_over_gap(self):
        """
        Inserts a gap in time series to check that windows spanning the gap are rejected.
        """
        # Build time: 0-5 min, gap, then 10-15 min
        times = list(pd.date_range("2025-01-01 00:00", periods=5, freq="1min"))
        times += list(pd.date_range("2025-01-01 00:10", periods=5, freq="1min"))
        time_array = pd.to_datetime(times)
        
        windows = get_valid_window_starts(
            time_array=time_array,
            window_duration='4min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
        
        # windows should look like
        expected_windows = np.array([
            np.arange(4),
            np.arange(4)+1,
            np.arange(4)+5,
            np.arange(4)+6
        ])

        assert np.array_equal(windows, expected_windows)



    def test_invalid_window_duration_raises(self):
        """
        Test that if window_duration is not a multiple of time_res,
        it raises a ValueError.
        """
        time_array = pd.date_range("2025-01-01", periods=60, freq="1min")

        with pytest.raises(ValueError):
            get_valid_window_starts(
                time_array=time_array,
                window_duration='7min',  # 7 not multiple of 4
                time_res='4min',
                max_gap='4min',
                progress_bar=False
            )



    def test_handles_short_series_gracefully(self):
        """
        Check that if series is too short for even one window, returns empty array.
        """
        time_array = pd.date_range("2025-01-01", periods=3, freq="1min")

        windows = get_valid_window_starts(
            time_array=time_array,
            window_duration='5min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )

        assert windows.shape[0] == 0



    def test_gap_inside_window_rejects(self):
        """
        Builds a window that is long enough but has a gap inside,
        which should be rejected due to max_gap constraint.
        """
        times = list(pd.date_range("2025-01-01 00:00", periods=3, freq="1min"))
        # Insert a jump from minute 2 to minute 10
        times += list(pd.date_range("2025-01-01 00:10", periods=3, freq="1min"))
        time_array = pd.to_datetime(times)

        windows = get_valid_window_starts(
            time_array=time_array,
            window_duration='5min',
            time_res='1min',
            max_gap='1min',
            progress_bar = False
        )
        
        assert windows.shape[0] == 0
        
    
    
    
    
    
class TestCreateInputOutputWindowIdxs:
    
    """
    All tests for create_input_output_window_idxs function from slicing.py
    """



    def test_regular_sequence_noForecastLag_singleOutput_noGap(self):
        """
        Tests that function works where there
          1) is no Forecast lag
          2) only predicting single output
          3) no gap
        """
        time_array = pd.date_range("2025-01-01", periods=6, freq="1min")
        indices = np.arange(len(time_array))

        X_idxs, Y_idxs = create_input_output_window_idxs(
            time_array=time_array,
            indices=indices,
            input_len='3min',
            output_len='1min',
            forecast_lag='1min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
        
        expected_X_idxs = np.array([
            np.arange(3),
            np.arange(3)+1,
            np.arange(3)+2,
        ])
        
        expected_Y_idxs = np.array([ 
            np.arange(1) + row[-1] + 1 for row in expected_X_idxs
        ])

        assert np.array_equal(X_idxs, expected_X_idxs)
        assert np.array_equal(Y_idxs, expected_Y_idxs)
        
        
    
    def test_regular_sequence_noForecastLag_multiOutput_noGap(self):
        """
        Tests that function works where there
          1) is no Forecast lag
          2) predicting mutiple outputs
          3) no gap
        """
        time_array = pd.date_range("2025-01-01", periods=7, freq="1min")
        indices = np.arange(len(time_array))

        X_idxs, Y_idxs = create_input_output_window_idxs(
            time_array=time_array,
            indices=indices,
            input_len='3min',
            output_len='2min',
            forecast_lag='1min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
        
        expected_X_idxs = np.array([
            np.arange(3),
            np.arange(3)+1,
            np.arange(3)+2,
        ])
        
        expected_Y_idxs = np.array([ 
            np.arange(2) + row[-1] + 1 for row in expected_X_idxs
        ])

        assert np.array_equal(X_idxs, expected_X_idxs)
        assert np.array_equal(Y_idxs, expected_Y_idxs)
        
        
        
    def test_regular_sequence_ForecastLag_multiOutput_noGap(self):
        """
        Tests that function works where there
          1) is Forecast lag
          2) predicting mutiple outputs
          3) no gap
        """
        time_array = pd.date_range("2025-01-01", periods=8, freq="1min")
        indices = np.arange(len(time_array))

        X_idxs, Y_idxs = create_input_output_window_idxs(
            time_array=time_array,
            indices=indices,
            input_len='3min',
            output_len='2min',
            forecast_lag='2min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
        
        expected_X_idxs = np.array([
            np.arange(3),
            np.arange(3)+1,
            np.arange(3)+2,
        ])
        
        expected_Y_idxs = np.array([ 
            np.arange(2) + row[-1] + 2 for row in expected_X_idxs
        ])

        assert np.array_equal(X_idxs, expected_X_idxs)
        assert np.array_equal(Y_idxs, expected_Y_idxs)
    
    
    
    def test_regular_sequence_ForecastLag_multiOutput_Gap(self):
        """
        Tests that function works where there
          1) is Forecast lag
          2) predicting mutiple outputs
          3) is gap
        """
        # Build time series: first 5 minutes
        times_first = pd.date_range("2025-01-01 00:00", periods=6, freq="1min").tolist()
        # Gap: skip to 00:10, then another 5 minutes
        times_second = pd.date_range("2025-01-01 00:10", periods=6, freq="1min").tolist()
        time_array = pd.to_datetime(times_first + times_second)
        indices = np.arange(len(time_array))
    
        # Run function with input_len=3, output_len=2, forecast_lag=2min
        X_idxs, Y_idxs = create_input_output_window_idxs(
            time_array=time_array,
            indices=indices,
            input_len='3min',
            output_len='2min',
            forecast_lag='2min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
    
        # Manually compute expected windows
        # In first block: we can have windows starting at 0, 1
        expected_X_idxs = [
            np.array([0,1,2]),
            np.array([6,7,8]),
        ]
        expected_Y_idxs = [
            np.array([4,5]),
            np.array([10,11]),
        ]

        assert np.array_equal(X_idxs, expected_X_idxs)
        assert np.array_equal(Y_idxs, expected_Y_idxs)



    def test_nonzero_start_indices(self):
        """
        Ensures the function handles indices that do not start at 0, 
        such as after slicing a df. Or which indices that may
        have already been split into contiguous chunks.
        """
        time_array = pd.date_range("2025-01-01", periods=6, freq="1min")
        indices = np.arange(len(time_array)) + 1000

        X_idxs, Y_idxs = create_input_output_window_idxs(
            time_array=time_array,
            indices=indices,
            input_len='3min',
            output_len='1min',
            forecast_lag='1min',
            time_res='1min',
            max_gap='1min',
            progress_bar=False
        )
        
        expected_X_idxs = np.array([
            np.arange(3),
            np.arange(3)+1,
            np.arange(3)+2,
        ]) + 1000
        
        expected_Y_idxs = np.array([ 
            np.arange(1) + row[-1] + 1 for row in expected_X_idxs
        ])

        assert np.array_equal(X_idxs, expected_X_idxs)
        assert np.array_equal(Y_idxs, expected_Y_idxs)



    def test_invalid_durations_raise(self):
        """
        Verifies that incompatible input_len / output_len / time_res raises ValueErrors.
        """
        time_array = pd.date_range("2025-01-01", periods=50, freq="1min")
        indices = np.arange(50)

        with pytest.raises(ValueError):
            create_input_output_window_idxs(
                time_array=time_array,
                indices=indices,
                input_len='7min',  # not divisible by 4
                output_len='3min',
                forecast_lag='1min',
                time_res='4min',
                max_gap='1min',
                progress_bar = False
            )





class TestComputeTimeBinIdxs:
    
    """
    Tests for compute_time_bin_idxs from slicing.py
    """



    def test_even_data_points_into_bins(self):
        """
        Perfectly regular time series should produce bins each with same number of points.
        """
        time_vals = pd.date_range("2025-01-01", periods=10, freq="1min").to_numpy()
        bin_indices, bin_times = compute_time_bin_idxs(time_vals, freq="2min")

        # First and last bin should only have one pt
        assert len(bin_indices[0]) == 1
        assert len(bin_indices[-1]) == 1
        # rest should have two
        assert np.all([ len(bin_indices[i]) == 2 for i in [1,2,3,4] ])



    def test_bins_with_empty_intervals(self):
        """
        Should correctly produce empty arrays when there is a gap in the data.
        """
        # Make data with a gap: 0-4 min and then 10-14 min
        
        times_first = pd.date_range("2025-01-01 00:00", periods=5, freq="1min").tolist()
        times_second = pd.date_range("2025-01-01 00:10", periods=5, freq="1min").tolist()
        time_vals = pd.to_datetime(times_first + times_second)

        bin_indices, bin_times = compute_time_bin_idxs(time_vals, freq="2min")

        expected_bin_idxs = [
            np.array([0]),
            np.array([1,2]),
            np.array([3,4]),
            np.array([]),
            np.array([]),
            np.array([5]),
            np.array([6,7]),
            np.array([8,9])
        ]
        
        expected_bin_times = pd.date_range("2025-01-01 00:00", periods=8, freq="2min")
    
        # confirm that lengths match first
        assert len(bin_indices) == len(expected_bin_idxs)
        assert len(bin_times) == len(expected_bin_times)
        
        # now check each element of bin_indices
        assert np.all([ np.array_equal(bin_indices[i], expected_bin_idxs[i]) 
                        for i in range(len(bin_indices)) ])
        
        # and check times
        assert np.all(expected_bin_times == bin_times)



    def test_handles_nonstandard_freq(self):
        """
        Tests with 3min frequency bins.
        """
        time_vals = pd.date_range("2025-01-01", periods=12, freq="1min").to_numpy()
        bin_indices, bin_times = compute_time_bin_idxs(time_vals, freq="2.5min")
        
        expected_bin_idxs = [
            np.array([0,1]),
            np.array([2,3]),
            np.array([4,5,6]),
            np.array([7,8]),
            np.array([9,10,11]),
            np.array([])
        ]
        
        expected_bin_times = pd.date_range("2025-01-01 00:00", periods=6, freq="2.5min")

        # confirm that lengths match first
        assert len(bin_indices) == len(expected_bin_idxs)
        assert len(bin_times) == len(expected_bin_times)
        
        # now check each element of bin_indices
        assert np.all([ np.array_equal(bin_indices[i], expected_bin_idxs[i]) 
                        for i in range(len(bin_indices)) ])
        
        # and check times
        assert np.all(expected_bin_times == bin_times)



    def test_indices_match_original_data(self):
        """
        Ensures indices actually index into original data correctly.
        """
        time_vals = pd.date_range("2025-01-01", periods=6, freq="1min").to_numpy()
        bin_indices, bin_times = compute_time_bin_idxs(time_vals, freq="1min")

        expected_bin_idxs = [ np.array([int_]) 
                              for int_ in np.arange(len(time_vals)) ]

        # confirm that lengths match first
        assert len(bin_indices) == len(expected_bin_idxs)
        assert len(bin_times) == len(time_vals)
        
        # now check each element of bin_indices
        assert np.all([ np.array_equal(bin_indices[i], expected_bin_idxs[i]) 
                        for i in range(len(bin_indices)) ])
        
        # and check times
        assert np.all(bin_times == time_vals)



    def test_returns_empty_bins_if_no_data(self):
        """
        Even if no points fall in a bin (full gap), should return empty arrays for that bin.
        """
        time_vals = pd.date_range("2025-01-01", periods=3, freq="1min").to_numpy()
        bin_indices, bin_times = compute_time_bin_idxs(time_vals, freq="10min")

        # We expect only one bin to actually have data
        non_empty_counts = [len(idxs) for idxs in bin_indices]
        assert any(count > 0 for count in non_empty_counts)
        assert any(count == 0 for count in non_empty_counts)





class TestComputeRollingBackwardWindowIdxs:

    """
    Tests for compute_rolling_backward_window_idxs from slicing.py
    """    



    def test_regular_3min_window(self):
        """
        Tests rolling windows on a simple evenly spaced 1-minute time series
        with a 3-minute window.
        """
        times = pd.date_range("2025-01-01 00:00", periods=6, freq="1min")
        window = "3min"

        idxs_out = compute_rolling_backward_window_idxs(times, 
                                                        window,
                                                        progress_bar=False)

        expected_indices = [
            np.array([0]),
            np.array([0,1]),
            np.array([0,1,2]),
            np.array([1,2,3]),
            np.array([2,3,4]),
            np.array([3,4,5])
        ]

        assert np.all([ np.array_equal(idxs_out[i], expected_indices[i]) 
                        for i in range(len(idxs_out)) ])




    def test_long_window_captures_all(self):
        """
        Tests that a window larger than the entire series always captures all up to T.
        """
        times = pd.date_range("2025-01-01 00:00", periods=5, freq="1min")
        window = "10min"

        idxs_out = compute_rolling_backward_window_idxs(times, 
                                                        window,
                                                        progress_bar=False)

        expected_indices = [
            np.array([0]),
            np.array([0,1]),
            np.array([0,1,2]),
            np.array([0,1,2,3]),
            np.array([0,1,2,3,4]),
        ]

        assert np.all([ np.array_equal(idxs_out[i], expected_indices[i]) 
                        for i in range(len(idxs_out)) ])



    def test_irregular_series(self):
        """
        Tests rolling windows on an irregular time series.
        """
        times = pd.to_datetime([
            "2025-01-01 00:00",
            "2025-01-01 00:02",
            "2025-01-01 00:05",
            "2025-01-01 00:09"
        ])
        window = "3min"

        idxs_out = compute_rolling_backward_window_idxs(times, 
                                                        window,
                                                        progress_bar=False)

        expected_indices = [
            np.array([0]),
            np.array([0,1]),
            np.array([2]),
            np.array([3])
        ]

        assert np.all([ np.array_equal(idxs_out[i], expected_indices[i]) 
                        for i in range(len(idxs_out)) ])







class TestComputeExpandingInputOutputIdxs:
    
    """
    Test suite for `compute_expanding_input_output_idxs` function in slicing.py.

    This function builds indices for backward expanding input windows,
    optionally indices for the direct current time (for geometric / positional inputs),
    and forward output windows, all under the assumption that the input 
    time series is uniformly spaced.
    """



    def test_single_window(self):
        """
        Tests that a single input window duration produces the correct indices for 
        inputs, direct inputs, and outputs over a small evenly spaced time series.
        """
        times = pd.date_range("2025-01-01", periods=10, freq="1min")
        input_dicts, direct_input_idxs, output_idxs, output_times = \
            compute_expanding_input_output_idxs(
                uniform_times=times,
                input_windows=["3min"],
                output_window="2min",
                forecast_lag="1min",
                freq="1min",
                progress_bar=False,
                capture_current=True
            )

        # Expect 7 valid input-output pairs (due to forecast lag + output length)
        assert len(input_dicts) == len(direct_input_idxs) == len(output_idxs) == 7

        # Check that the backward input indices expand correctly
        assert np.array_equal(input_dicts[0]["3min"], np.array([0]))
        assert np.array_equal(input_dicts[1]["3min"], np.array([0,1]))
        assert np.array_equal(input_dicts[2]["3min"], np.array([0,1,2]))
        assert np.array_equal(input_dicts[3]["3min"], np.array([1,2,3]))
        assert np.array_equal(input_dicts[4]["3min"], np.array([2,3,4]))
        assert np.array_equal(input_dicts[5]["3min"], np.array([3,4,5]))
        assert np.array_equal(input_dicts[6]["3min"], np.array([4,5,6]))

        # Check that direct_input_idxs correctly returns the index at T
        assert np.array_equal(direct_input_idxs[0], np.array([2]))
        assert np.array_equal(direct_input_idxs[1], np.array([3]))
        assert np.array_equal(direct_input_idxs[2], np.array([4]))
        assert np.array_equal(direct_input_idxs[3], np.array([5]))
        assert np.array_equal(direct_input_idxs[4], np.array([6]))
        assert np.array_equal(direct_input_idxs[5], np.array([7]))
        assert np.array_equal(direct_input_idxs[6], np.array([8]))

        # Check the output indices start after lag and have the correct length
        assert np.array_equal(output_idxs[0], np.array([2,3]))
        assert np.array_equal(output_idxs[1], np.array([3,4]))
        assert np.array_equal(output_idxs[2], np.array([4,5]))
        assert np.array_equal(output_idxs[3], np.array([5,6]))
        assert np.array_equal(output_idxs[4], np.array([6,7]))
        assert np.array_equal(output_idxs[5], np.array([7,8]))
        assert np.array_equal(output_idxs[6], np.array([8,9]))
    


    def test_multiple_windows(self):
        """
        Tests that multiple input window durations produce separate input sequences
        with each having the expected lengths, and that direct indices are present.
        """
        times = pd.date_range("2025-01-01", periods=10, freq="1min")
        input_dicts, direct_input_idxs, output_idxs, output_times = \
            compute_expanding_input_output_idxs(
                uniform_times=times,
                input_windows=["3min", "5min"],
                output_window="2min",
                forecast_lag="1min",
                freq="1min",
                progress_bar=False,
                capture_current=True
            )

        assert "3min" in input_dicts[0]
        assert "5min" in input_dicts[0]
        
        assert input_dicts[0]['3min'] == input_dicts[0]['5min'] == 0
        assert input_dicts[1]['3min'].tolist() == input_dicts[1]['5min'].tolist() == [0,1]
        assert input_dicts[2]['3min'].tolist() == input_dicts[2]['5min'].tolist() == [0,1,2]
        assert input_dicts[3]['3min'].tolist() == [1,2,3]
        assert input_dicts[3]['5min'].tolist() == [0,1,2,3]
        assert input_dicts[4]['3min'].tolist() == [2,3,4]
        assert input_dicts[4]['5min'].tolist() == [0,1,2,3,4]
        assert input_dicts[5]['3min'].tolist() == [3,4,5]
        assert input_dicts[5]['5min'].tolist() == [1,2,3,4,5]
        
        for i in range(7):
            assert direct_input_idxs[i] == i+2
        
        assert np.array_equal(output_idxs[0], np.array([2,3]))
        assert np.array_equal(output_idxs[1], np.array([3,4]))
        assert np.array_equal(output_idxs[2], np.array([4,5]))
        assert np.array_equal(output_idxs[3], np.array([5,6]))
        assert np.array_equal(output_idxs[4], np.array([6,7]))
        assert np.array_equal(output_idxs[5], np.array([7,8]))
        assert np.array_equal(output_idxs[6], np.array([8,9]))
        


    def test_without_capture_current(self):
        """
        Ensures that if `capture_current` is False, the direct_input_idxs are all None.
        """
        times = pd.date_range("2025-01-01", periods=10, freq="1min")
        input_dicts, direct_input_idxs, output_idxs, output_times = \
            compute_expanding_input_output_idxs(
                uniform_times=times,
                input_windows=["3min"],
                output_window="2min",
                forecast_lag="1min",
                freq="1min",
                progress_bar=False,
                capture_current=False
            )

        assert all(isinstance(x, np.ndarray) and x.size == 0 for x in direct_input_idxs)


    def test_skips_if_output_exceeds(self):
        """
        Tests that when the output window would exceed the end of the time series,
        those indices are correctly skipped and not returned.
        """
        times = pd.date_range("2025-01-01", periods=5, freq="1min")
        input_dicts, direct_input_idxs, output_idxs, output_times = \
            compute_expanding_input_output_idxs(
                uniform_times=times,
                input_windows=["3min"],
                output_window="3min",
                forecast_lag="0min",
                freq="1min",
                progress_bar=False,
                capture_current=True
            )

        # Only the initial indices that can safely produce a full output window
        assert len(input_dicts) == len(output_idxs) == len(direct_input_idxs) == 2

        assert np.array_equal(output_idxs[0], np.array([1,2,3]))
        assert np.array_equal(output_idxs[1], np.array([2,3,4]))


    def test_non_uniform_raises(self):
        """
        Ensures the function raises an AssertionError when the provided 
        time array is not uniformly spaced.
        """
        times = pd.to_datetime([
            "2025-01-01 00:00",
            "2025-01-01 00:01",
            "2025-01-01 00:03"
        ])
        with pytest.raises(AssertionError, match="uniformly spaced"):
            compute_expanding_input_output_idxs(
                uniform_times=times,
                input_windows=["2min"],
                output_window="1min",
                forecast_lag="0min",
                freq="1min",
                progress_bar=False
            )