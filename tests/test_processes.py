import numpy as np
from unittest import TestCase
from HawkesPyLib.processes import UnivariateHawkesProcess
import os 

file_path = fpath = os.path.join(os.path.dirname(__file__), "timestamp_fixture.csv")
timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)

class TestApproxPowlaw_process(TestCase):
    """ 
    Test the ApproxPowerlawProcess Class.
    """
    def setUp(self):
        self.timestamps = timestamps
        self.mu = 0.5
        self.eta = 0.5
        self.alpha = 0.5
        self.tau0 = 0.5
        self.m = 5.
        self.M = 4
        self.T = 50.

    def test_param_setter(self):
        ApproxPowlawHP = UnivariateHawkesProcess("powlaw")
        ApproxPowlawHP.set_params(self.mu, self.eta, alpha=self.alpha, tau0=self.tau0, m=self.m, M=self.M)

        self.assertEqual(self.mu, ApproxPowlawHP.mu)
        self.assertEqual(self.eta, ApproxPowlawHP.eta)
        self.assertEqual(self.alpha, ApproxPowlawHP.alpha)
        self.assertEqual(self.tau0, ApproxPowlawHP.tau0)
        self.assertEqual(self.m, ApproxPowlawHP.m)
        self.assertEqual(self.M, ApproxPowlawHP.M)

        ApproxPowlawHP = UnivariateHawkesProcess("powlaw-cutoff")
        ApproxPowlawHP.set_params(self.mu, self.eta, alpha=self.alpha, tau0=self.tau0, m=self.m, M=self.M)

        self.assertEqual(self.mu, ApproxPowlawHP.mu)
        self.assertEqual(self.eta, ApproxPowlawHP.eta)
        self.assertEqual(self.alpha, ApproxPowlawHP.alpha)
        self.assertEqual(self.tau0, ApproxPowlawHP.tau0)
        self.assertEqual(self.m, ApproxPowlawHP.m)
        self.assertEqual(self.M, ApproxPowlawHP.M)

    def test_param_getter(self):
        ApproxPowlawHP = UnivariateHawkesProcess("powlaw")
        ApproxPowlawHP.set_params(self.mu, self.eta, alpha=self.alpha, tau0=self.tau0, m=self.m, M=self.M)
        mu, eta, alpha, tau0, m, M = ApproxPowlawHP.get_params()
        self.assertEqual(self.mu, mu)
        self.assertEqual(self.eta, eta)
        self.assertEqual(self.mu, alpha)
        self.assertEqual(self.tau0, tau0)
        self.assertEqual(self.m, m)
        self.assertEqual(self.M, M)

        ApproxPowlawHP = UnivariateHawkesProcess("powlaw-cutoff")
        ApproxPowlawHP.set_params(self.mu, self.eta, alpha=self.alpha, tau0=self.tau0, m=self.m, M=self.M)
        mu, eta, alpha, tau0, m, M = ApproxPowlawHP.get_params()
        self.assertEqual(self.mu, mu)
        self.assertEqual(self.eta, eta)
        self.assertEqual(self.mu, alpha)
        self.assertEqual(self.tau0, tau0)
        self.assertEqual(self.m, m)
        self.assertEqual(self.M, M)

    def test_timestamp_setter(self):
        """ arrival time setter """
        ApproxPowlawHP = UnivariateHawkesProcess("powlaw")
        ApproxPowlawHP.set_arrival_times(self.timestamps, self.T)
        self.assertEqual(self.T, ApproxPowlawHP.T)
        np.testing.assert_array_equal(self.timestamps, ApproxPowlawHP.timestamps)

        # check if invalid values are rejected
        with self.assertRaises(ValueError):
            ApproxPowlawHP.set_arrival_times(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5.)

        with self.assertRaises(ValueError):
            ApproxPowlawHP.set_arrival_times(np.array([0., 0.5, 1.0, 2.5]), 5.)
        with self.assertRaises(ValueError):
            ApproxPowlawHP.set_arrival_times(np.array([-0.5, -0.1, 1.0, 2.5]), 5.)

        with self.assertRaises(TypeError):
            ApproxPowlawHP.set_arrival_times(np.array([1, 2, 3, 4]), 5.)

        with self.assertRaises(TypeError):
            ApproxPowlawHP.set_arrival_times([0.5, 1., 1.5, 2., 2.9], 5.)


class TestExpoPowlaw_process(TestCase):
    """ 
    Test the ApproxPowerlawProcess Class.
    """
    def setUp(self):
        self.timestamps = timestamps
        self.mu = 0.5
        self.eta = 0.5
        self.theta = 0.5
        self.T = 50.

    def test_param_setter(self):
        ExpEstimator = UnivariateHawkesProcess("expo")
        ExpEstimator.set_params(self.mu, self.eta, theta=self.theta)

        self.assertEqual(self.mu, ExpEstimator.mu)
        self.assertEqual(self.eta, ExpEstimator.eta)
        self.assertEqual(self.theta, ExpEstimator.theta)

    def test_param_getter(self):
        ExpEstimator = UnivariateHawkesProcess("expo")
        ExpEstimator.set_params(self.mu, self.eta, theta=self.theta)
        mu, eta, theta = ExpEstimator.get_params()
        self.assertEqual(self.mu, mu)
        self.assertEqual(self.eta, eta)
        self.assertEqual(self.theta, theta)


class TestSumExpoPowlaw_process(TestCase):
    """ 
    Test the ApproxPowerlawProcess Class.
    """
    def setUp(self):
        self.timestamps = timestamps
        self.mu = 0.5
        self.eta = 0.5
        self.theta_vec = np.array([0.5, 0.3, 0.5])
        self.T = 50.

    def test_param_setter(self):
        SumExpEstimator = UnivariateHawkesProcess("sum-expo")
        SumExpEstimator.set_params(self.mu, self.eta, theta_vec=self.theta_vec)

        self.assertEqual(self.mu, SumExpEstimator.mu)
        self.assertEqual(self.eta, SumExpEstimator.eta)
        np.testing.assert_array_equal(self.theta_vec, SumExpEstimator.theta_vec)


    def test_param_getter(self):
        SumExpEstimator = UnivariateHawkesProcess("sum-expo")
        SumExpEstimator.set_params(self.mu, self.eta, theta_vec=self.theta_vec)

        mu, eta, theta_vec = SumExpEstimator.get_params()
        self.assertEqual(self.mu, mu)
        self.assertEqual(self.eta, eta)
        np.testing.assert_array_equal(self.theta_vec, theta_vec)

