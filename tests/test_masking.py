import numpy as np
from tsblocks.masking import interp_nans_within_limit



class TestInterpNansWithinLimit:
    """
    All tests for interp_nans_within_limit function from masking.py
    """



    def test_interpolates_short_runs(self):
        """
        Verifies short runs of NaNs are interpolated correctly.
        """
        arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2,
                                               progress_bar=False)

        # Short run of length 2 should be interpolated
        assert np.allclose(fixed, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.all(mask == [False, True, True, False, False])



    def test_leaves_long_runs(self):
        """
        Ensures long runs of NaNs remain as NaNs.
        """
        arr = np.array([1.0, np.nan, np.nan, np.nan, 4.0])
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2,
                                               progress_bar=False)

        # Long run should not be interpolated
        assert np.array_equal(arr, fixed, equal_nan=True)
        assert not mask.any()



    def test_interpolates_start_boundary(self):
        """
        If NaNs at start are shorter than threshold, fills with first known value.
        """
        arr = np.array([np.nan, np.nan, 3.0, 4.0])
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2,
                                               progress_bar=False)

        assert np.array_equal(arr, fixed, equal_nan=True)
        assert not mask.any()



    def test_interpolates_end_boundary(self):
        """
        If NaNs at end are shorter than threshold, fills with last known value.
        """
        arr = np.array([1.0, 2.0, np.nan, np.nan])
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2,
                                               progress_bar=False)

        assert np.array_equal(arr, fixed, equal_nan=True)
        assert not mask.any()
    
    
    
    def test_two_middle_gaps_both_filled(self):
        """
        Tests an array with two short NaN runs in the middle,
        both within max_consec_nans, so both are interpolated.
        """
        arr = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0])
        
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2, 
                                               progress_bar=False)
        
        assert np.allclose(fixed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        # Mask should reflect exactly those locations
        assert np.all(mask == [False, False, True, True, False, False, True, False, False])
    
    
    
    def test_two_gaps_only_one_filled(self):
        """
        Tests an array with two NaN runs, one short enough to interpolate,
        the other exceeding max_consec_nans so it stays as NaNs.
        """
        arr = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 6.0, np.nan, 8.0, 9.0])
        # two runs: [2:5] length=3, and [6:7] length=1
        
        fixed, mask = interp_nans_within_limit(arr, 
                                               max_consec_nans=2, 
                                               progress_bar=False)
        
        # Only the second run (length=1) gets filled, first stays as NaNs
        expected = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0])
        assert np.allclose(fixed, expected, equal_nan=True)
        # Mask reflects only the single interpolated location
        assert np.all(mask == [False, False, False, False, False, False, True, False, False])