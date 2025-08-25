import unittest
import numpy as np


class TestRandomSingleton(unittest.TestCase):
    def setUp(self):
        # Print the current RNG states at the start of each test
        print("\nTest:", self._testMethodName)
        print("Initial NP_RNG state:", [NP_RNG.rand() for _ in range(3)])
        print("Initial random state:", [random.random() for _ in range(3)])
        
    def test_fixed_seed(self):
        """Test that random numbers are reproducible within the same run"""
        # Generate first sequence
        np_nums1 = NP_RNG.rand()
        # Generate second number - should be different
        np_nums2 = NP_RNG.rand()
        # These should be different (proving RNG is working)
        self.assertNotEqual(np_nums1, np_nums2)
        
        # Same for Python's random
        py_nums1 = random.random()
        py_nums2 = random.random()
        self.assertNotEqual(py_nums1, py_nums2)
        
        print(f"numpy numbers: {np_nums1}, {np_nums2}")
        print(f"python numbers: {py_nums1}, {py_nums2}")

    def test_seed_disabled(self):
        """Test that reseeding is disabled"""
        # Get initial numbers
        np_initial = NP_RNG.rand()
        py_initial = random.random()
        
        # Try to reseed
        NP_RNG.seed(999)
        random.seed(999)
        
        # Numbers should continue from previous state, not reset
        np_after = NP_RNG.rand()
        py_after = random.random()
        
        print(f"numpy before/after attempted reseed: {np_initial}, {np_after}")
        print(f"python before/after attempted reseed: {py_initial}, {py_after}")
        
        self.assertNotEqual(np_initial, np_after)  # Should get next number in sequence
        self.assertNotEqual(py_initial, py_after)  # Should get next number in sequence

    def test_distributions(self):
        """Test that both RNGs produce valid uniform distributions"""
        # Get multiple numbers from each
        n = 1000
        np_nums = [NP_RNG.rand() for _ in range(n)]
        py_nums = [random.random() for _ in range(n)]
        
        # Calculate mean and std to verify statistical properties
        np_mean = np.mean(np_nums)
        py_mean = np.mean(py_nums)
        
        # Means should be close to 0.5 for uniform [0,1)
        self.assertAlmostEqual(np_mean, 0.5, delta=0.1)
        self.assertAlmostEqual(py_mean, 0.5, delta=0.1)
        
        # Standard deviations should be close to 1/sqrt(12) ≈ 0.289
        np_std = np.std(np_nums)
        py_std = np.std(py_nums)
        expected_std = 1/np.sqrt(12)  # Theoretical std for uniform [0,1)
        
        self.assertAlmostEqual(np_std, expected_std, delta=0.1)
        self.assertAlmostEqual(py_std, expected_std, delta=0.1)

if __name__ == '__main__':
    import sys
    print("\nPython path:", sys.path)
    print("\nChecking if sitecustomize.py is in the correct location...")
    import site
    print("Site packages directories:", site.getsitepackages())
    print("User site directory:", site.getusersitepackages())
    
    try:
        import sitecustomize
        print("\nsitecustomize module found!")
        print("MASTER_SEED value:", sitecustomize.MASTER_SEED)
    except ImportError:
        print("\nERROR: sitecustomize module not found!")
        print("Make sure sitecustomize.py is in one of the Python path directories")
    
    unittest.main()