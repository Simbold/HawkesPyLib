import numpy as np
from unittest import TestCase
from HawkesPyLib.core.intensity import uvhp_expo_intensity, uvhp_sum_expo_intensity, uvhp_approx_powl_cutoff_intensity, uvhp_approx_powl_intensity, generate_eval_grid
from HawkesPyLib.core.kernel import uvhp_approx_powl_cutoff_kernel, uvhp_approx_powl_kernel, uvhp_expo_kernel, uvhp_sum_expo_kernel


class TestExpo_intensity(TestCase):
    """ Tests the intensity evaluation functions """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 15.0
        self.mu = 0.5
        self.eta = 0.7
        self.theta = 0.1
        grid = generate_eval_grid(0.0001, self.T)
        self.intensity = uvhp_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta)

    def test_flat_intensity_before_first_jump(self):
        """ Check if intensity is equal to background intensity until first event arrival. """
        int_test = self.intensity[self.intensity[:, 0] < self.timestamps[0], 1]
        condition = (int_test == self.mu).all()
        self.assertTrue(condition)

    def test_jump_size(self):
        """ Check if all jumpsizes have the desired size."""
        d_intensity = np.diff(self.intensity[:,1])
        idx = np.where(np.isin(self.intensity[:, 0], self.timestamps))[0]
        actual_jump_sizes = d_intensity[idx-1]
        desired_jump_sizes = np.repeat(self.eta / self.theta, len(self.timestamps))

        np.testing.assert_allclose(actual_jump_sizes, desired_jump_sizes, rtol=1e-3, atol=1e-3)

    def test_decay_speed(self):
        """ Check decay speed after first event arrival """
        cond = ((self.intensity[:, 0] >= self.timestamps[0]) & (self.intensity[:, 0] < self.timestamps[1]))
        actual = self.intensity[cond , 1] - self.mu
        times = self.intensity[cond, 0] 

        desired = uvhp_expo_kernel(times-times[0], self.eta, self.theta)
        np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)
        

class TestSumExpo_intensity(TestCase):
    """ Tests the intensity evaluation functions """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 15.0
        self.mu = 0.5
        self.eta = 0.7
        self.theta_vec = np.array([0.1, 0.5, 1.5])
        grid = generate_eval_grid(0.0001, self.T)
        self.intensity = uvhp_sum_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta_vec)

    def test_flat_intensity_before_first_jump(self):
        """ Check if intensity is equal to background intensity until first event arrival. """
        int_test = self.intensity[self.intensity[:, 0] < self.timestamps[0], 1]
        condition = (int_test == self.mu).all()
        self.assertTrue(condition)

    def test_jump_size(self):
        """ Check if all jumpsizes have the desired size."""
        d_intensity = np.diff(self.intensity[:,1])
        idx = np.where(np.isin(self.intensity[:, 0], self.timestamps))[0]
        actual_jump_sizes = d_intensity[idx-1]

        P = len(self.theta_vec)
        etaom = np.float64(self.eta / P)
        jump_size = etaom * np.sum(1/self.theta_vec)
        desired_jump_sizes = np.repeat(jump_size, len(self.timestamps))

        np.testing.assert_allclose(actual_jump_sizes, desired_jump_sizes, rtol=1e-3, atol=1e-3)

    def test_decay_speed(self):
        """ Check decay speed after first event arrival """
        cond = ((self.intensity[:, 0] >= self.timestamps[0]) & (self.intensity[:, 0] < self.timestamps[1]))
        actual = self.intensity[cond , 1] - self.mu
        times = self.intensity[cond, 0] 

        desired = uvhp_sum_expo_kernel(times-times[0], self.eta, self.theta_vec)
        np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)


class TestApproxPowlawCutoff_intensity(TestCase):
    """ Tests the intensity evaluation functions """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 15.0
        self.mu = 0.5
        self.eta = 0.7
        self.alpha = 0.4
        self.tau0 = 0.1
        self.m = 5.0
        self.M = 15

        grid = generate_eval_grid(0.0001, self.T)
        self.intensity = uvhp_approx_powl_cutoff_intensity(self.timestamps, grid, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)

    def test_flat_intensity_before_first_jump(self):
        """ Check if intensity is equal to background intensity until first event arrival. """
        int_test = self.intensity[self.intensity[:, 0] < self.timestamps[0], 1]
        condition = (int_test == self.mu).all()
        self.assertTrue(condition)

    def test_jump_size(self):
        """ Check if all jumpsizes have the desired size."""
        d_intensity = np.diff(self.intensity[:,1])
        idx = np.where(np.isin(self.intensity[:, 0], self.timestamps))[0]
        actual_jump_sizes = d_intensity[idx-1] 
        desired_jump_sizes = np.repeat(0., len(self.timestamps))

        np.testing.assert_allclose(actual_jump_sizes, desired_jump_sizes, atol=1e-2)

    def test_decay_speed(self):
        """ Check decay speed after first event arrival """
        cond = ((self.intensity[:, 0] >= self.timestamps[0]) & (self.intensity[:, 0] < self.timestamps[1]))
        actual = self.intensity[cond , 1] - self.mu
        times = self.intensity[cond, 0] 

        desired = uvhp_approx_powl_cutoff_kernel(times-times[0], self.eta, self.alpha, self.tau0, self.m, self.M)
        np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)


class TestApproxPowlaw_intensity(TestCase):
    """ Tests the intensity evaluation functions """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 15.0
        self.mu = 0.5
        self.eta = 0.7
        self.alpha = 0.4
        self.tau0 = 0.1
        self.m = 5.0
        self.M = 15

        grid = generate_eval_grid(0.0001, self.T)
        self.intensity = uvhp_approx_powl_intensity(self.timestamps, grid, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)

    def test_flat_intensity_before_first_jump(self):
        """ Check if intensity is equal to background intensity until first event arrival. """
        int_test = self.intensity[self.intensity[:, 0] < self.timestamps[0], 1]
        condition = (int_test == self.mu).all()
        self.assertTrue(condition)

    def test_jump_size(self):
        """ Check if all jumpsizes have the desired size."""
        d_intensity = np.diff(self.intensity[:,1])
        idx = np.where(np.isin(self.intensity[:, 0], self.timestamps))[0]
        actual_jump_sizes = d_intensity[idx-1]
        
        jump_size = uvhp_approx_powl_kernel(0., self.eta, self.alpha, self.tau0, self.m, self.M)
        desired_jump_sizes = np.repeat(jump_size, len(self.timestamps))

        np.testing.assert_allclose(actual_jump_sizes, desired_jump_sizes, rtol=1e-3, atol=1e-3)

    def test_decay_speed(self):
        """ Check decay speed after first event arrival """
        cond = ((self.intensity[:, 0] >= self.timestamps[0]) & (self.intensity[:, 0] < self.timestamps[1]))
        actual = self.intensity[cond , 1] - self.mu
        times = self.intensity[cond, 0] 

        desired = uvhp_approx_powl_kernel(times-times[0], self.eta, self.alpha, self.tau0, self.m, self.M)
        np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)




