import numpy as np
from unittest import TestCase
from HawkesPyLib.core.kernel import uvhp_expo_kernel, uvhp_sum_expo_kernel, uvhp_approx_powl_kernel, uvhp_approx_powl_cutoff_kernel

class TestHawkesKernels(TestCase):
    """ 
    Test the the memory kernel functions
    """
    def setUp(self):
        self.times = np.linspace(0, 10, 100)
        self.eta = 0.7

    def test_expo_kernel_zero(self):
        """ expo kernel t=0"""
        theta = 0.1
        actual = uvhp_expo_kernel(0., self.eta, theta)
        desired = self.eta / theta
        self.assertEqual(actual, desired)

    def test_sum_expo_kernel_zero(self):
        """ sum expo t = 0"""
        theta_vec = np.array([0.1, 0.5, 1.5])
        actual = uvhp_sum_expo_kernel(0., self.eta, theta_vec)
        desired = self.eta / len(theta_vec) * np.sum(1 / theta_vec)
        self.assertEqual(actual, desired)

    def test_cross_test_expo(self):
        """ test that sum expo with P=1 equal single expo """
        theta_vec = np.array([0.1])
        sum_expo_values= uvhp_sum_expo_kernel(self.times, self.eta, theta_vec)
        expo_values = uvhp_expo_kernel(self.times, self.eta, theta_vec[0])
        np.testing.assert_almost_equal(sum_expo_values, expo_values, 12)

    def test_approx_powl_zero(self):
        """ test powlaw kernel at t=0 """
        alpha = 0.4
        tau0 = 0.1
        m = 5.
        M = 10
        actual = uvhp_approx_powl_kernel(0., self.eta, alpha, tau0, m, M)

        z = tau0**(-alpha) * (1 - m**(-alpha * M)) / (1 - m**(-alpha))
        ak = tau0 * m**np.arange(0, M, 1)
        desired = (self.eta / z) * np.sum(ak**(-1-alpha))
        self.assertAlmostEqual(actual, desired, 12)

    def test_approx_powl_cutoff_zero(self):
        """ test powlaw cutoff kernel at t=0 """
        alpha = 0.4
        tau0 = 0.1
        m = 5.
        M = 10
        actual = uvhp_approx_powl_cutoff_kernel(0., self.eta, alpha, tau0, m, M)
        desired = 0.
        self.assertEqual(actual, desired)
        


