import numpy as np
import os 
from unittest import TestCase
from HawkesPyLib.inference import (uvhp_approx_powl_logL,
                                    uvhp_approx_powl_cut_logL,
                                    uvhp_expo_logL,
                                    uvhp_expo_logL_grad,
                                    uvhp_sum_expo_logL,
                                    uvhp_sum_expo_logL_grad)

file_path = fpath = os.path.join(os.path.dirname(__file__), "timestamp_fixture.csv")

class TestExpo_logL(TestCase):
    """ 
    Test the negative logL of single exponential Hawkes processes
    """
    def setUp(self):
        self.timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)  
        self.T = 50.0
        self.param_vec = np.array([0.5, 0.7, 0.001])

    def test_nlogL(self):
        
        actual = uvhp_expo_logL(self.param_vec, self.timestamps, self.T)
        desired = -113.6532979778095295841922052204608917236328125000000000000000000000
        self.assertEqual(actual, desired)
    
    def test_nlogL_empty(self):
        actual = uvhp_expo_logL(self.param_vec, np.array([]), self.T)
        desired = self.param_vec[0] * self.T
        self.assertEqual(actual, desired)

    def test_nlogL_grad(self):
        actual_vec = uvhp_expo_logL_grad(self.param_vec, self.timestamps, self.T)
        desired_vec = np.array([1.4612153393174436288859396881889551877975463867187500000000000000,
                              10.2419890433446809652195952367037534713745117187500000000000000000,
                          -12926.1384571849030180601403117179870605468750000000000000000000000000])
        np.testing.assert_equal(actual_vec, desired_vec)

    def test_nlogL_grad_empty(self):
        actual_vec = uvhp_expo_logL_grad(self.param_vec, np.array([]), self.T)
        desired_vec = np.array([self.T, 0, 0])
        np.testing.assert_equal(actual_vec, desired_vec)

class TestSumExpo_logL(TestCase):
    """ Test the negative logL of sum-expo Hawkes process """
    def setUp(self):
        self.timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)  
        self.T = 50.0
        self.param_vec = np.array([0.5, 0.7, 0.001, 1.0, 0.5])

    def test_nlogL(self):
        
        actual = uvhp_sum_expo_logL(self.param_vec, self.timestamps, self.T)
        desired = -88.2822143534869496761530172079801559448242187500000000000000000000
        self.assertEqual(actual, desired)
    
    def test_nlogL_empty(self):
        actual = uvhp_sum_expo_logL(self.param_vec, np.array([]), self.T)
        desired = self.param_vec[0] * self.T
        self.assertEqual(actual, desired)

    def test_nlogL_grad(self):
        actual_vec = uvhp_sum_expo_logL_grad(self.param_vec, self.timestamps, self.T)
        desired_vec = np.array([13.3085405404303127596676858956925570964813232421875000000000000000,
                                    1.7422263440205842943697689406690187752246856689453125000000000000,
                                    -8834.0584846120582369621843099594116210937500000000000000000000000000,
                                    -1.5184645499852693628639599410234950482845306396484375000000000000,
                                    -0.8727071797167826883168118001776747405529022216796875000000000000])
        np.testing.assert_equal(actual_vec, desired_vec)

    def test_nlogL_grad_empty(self):
        actual_vec = uvhp_sum_expo_logL_grad(self.param_vec, np.array([]), self.T)
        desired_vec = np.array([self.T, 0., 0., 0., 0.])
        np.testing.assert_equal(actual_vec, desired_vec)

class TestPowlaw_logL(TestCase):
    """ Test the negative logL of powlaw Hawkes process """
    def setUp(self):
        self.timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)  
        self.T = 50.0
        self.param_vec = np.array([0.5, 0.7, 0.3, 0.01])
        self.m = 5.
        self.M = 4

    def test_nlogL(self):
        
        actual = uvhp_approx_powl_logL(self.param_vec, self.timestamps, self.T, self.m, self.M)
        desired = -68.7895819934511223436857108026742935180664062500000000000000000000
        self.assertEqual(actual, desired)
    
    def test_nlogL_empty(self):
        actual = uvhp_approx_powl_logL(self.param_vec, np.array([]), self.T, self.m, self.M)
        desired = self.param_vec[0] * self.T
        self.assertEqual(actual, desired)

class TestPowlaw_cutoff_logL(TestCase):
    """ Test the negative logL of powlaw-cutoff Hawkes process """
    def setUp(self):
        self.timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)  
        self.T = 50.0
        self.param_vec = np.array([0.5, 0.7, 0.3, 0.01])
        self.m = 5.
        self.M = 4

    def test_nlogL(self):
        
        actual = uvhp_approx_powl_cut_logL(self.param_vec, self.timestamps, self.T, self.m, self.M)
        desired = -41.1549331471721302477817516773939132690429687500000000000000000000
        self.assertEqual(actual, desired)
    
    def test_nlogL_empty(self):
        actual = uvhp_approx_powl_cut_logL(self.param_vec, np.array([]), self.T, self.m, self.M)
        desired = self.param_vec[0] * self.T
        self.assertEqual(actual, desired)

