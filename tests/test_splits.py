import numpy as np
import pandas as pd

from tsblocks.splits import index_blocks_by_duration



class TestIndexBlocksByDuration:
    
    
    
    """
    All tests for the index_blocks_by_duration function in splits.py.
    """



    def test_regular_blocks(self):
        """
        Simple evenly spaced days to confirm blocks align with duration.
        """
        times = pd.date_range("2025-01-01", periods=30, freq="1D")
        blocks = index_blocks_by_duration(times, 
                                          block_duration='10D',
                                          progress_bar=False)

        assert len(blocks) == 3
        assert all(len(blocks[b]) == 10 for b in blocks)

        # Check actual indices
        expected_first_block = np.arange(0,10)
        assert np.all(np.sort(blocks[0]) == expected_first_block)



    def test_blocks_with_gap(self):
        """
        Inserts a large gap to see blocks respect chronological time not index count.
        """
        times = pd.date_range("2025-01-01", periods=15, freq="1D").tolist()
        times += pd.date_range("2025-03-01", periods=15, freq="1D").tolist()
        times = pd.to_datetime(times)

        blocks = index_blocks_by_duration(times, 
                                          block_duration='10D',
                                          progress_bar=False)
        assert len(blocks) >= 3
        for block_indices in blocks.values():
            assert np.all(block_indices >= 0) and np.all(block_indices < len(times))



    def test_with_buffer_duration(self):
        """
        Checks that buffer_duration trims end of block as expected.
        """
        times = pd.date_range("2025-01-01", periods=30, freq="1D")
        blocks = index_blocks_by_duration(times, 
                                          block_duration='10D', 
                                          buffer_duration='2D',
                                          progress_bar=False)

        assert len(blocks) == 3
        # All blocks should have length 9
        assert np.all([ len(times[ blocks[i] ]) == 9 for i in range(len(blocks)) ])



    def test_respects_sorting(self):
        """
        Input times shuffled should still return same block grouping by datetime, not by input order.
        """
        times = pd.date_range("2025-01-01", periods=30, freq="1D")
        shuffled_times = np.random.permutation(times)
        blocks = index_blocks_by_duration(shuffled_times, 
                                          block_duration='10D',
                                          progress_bar=False)

        # Should still be 3 blocks of ~10
        assert len(blocks) == 3
        sizes = [len(blocks[k]) for k in sorted(blocks)]
        assert all(9 <= s <= 10 for s in sizes)
        
        
        
    def test_skips_blocks_due_to_large_gap(self):
        """
        Ensures that a sufficiently large gap in time skips over what would be 
        enough duration for an otherwise full block. Ensures time-based indexing.
        """
        # Create 10 days from Jan 1 to Jan 13
        first_segment = pd.date_range("2025-01-01", periods=13, freq="1D").tolist()
        # GAP: no data from Jan 14 to Jan 31
        # Create another 10 days from Feb 1 to Feb 10
        second_segment = pd.date_range("2025-02-01", periods=10, freq="1D").tolist()
        # Combine into one datetime series
        times = pd.to_datetime(first_segment + second_segment)

        # Now index with block_duration of 10 days
        blocks = index_blocks_by_duration(times, 
                                          block_duration='10D',
                                          progress_bar=False)

        # Should only produce 3 blocks: 
        assert len(blocks) == 3
        
        # first and last blocks should have length 10, but middle block
        # should only have 3 points
        assert len(blocks[0]) == 10
        assert len(blocks[2]) == 10
        assert len(blocks[1]) == 3
