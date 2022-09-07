import numpy as np
from unittest import TestCase, mock
from HawkesPyLib.inference import ExpHawkesProcessInference
from HawkesPyLib.processes import generate_eval_grid
import os 

file_path = fpath = os.path.join(os.path.dirname(__file__), "timestamp_fixture.csv")
timestamps =  np.loadtxt(file_path, delimiter=",", dtype=float)

class TestExpo_estimate(TestCase):
    """ 
    Test the .estimate method.
    """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 50.0
        self.rng = np.random.default_rng(242)
        self.param_vec0 = np.array([0.6, 0.5, 0.4, 0.9])

    def test_expo_estimate_calls_correct_custom(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_expo_mle") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate(self.timestamps, self.T, custom_param_vec0=True, param_vec0=self.param_vec0)
        patched_function.assert_called_once_with(self.timestamps, self.T, self.param_vec0)

    def test_sum_expo_estimate_calls_correct_random(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_expo_mle") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate(self.timestamps, self.T)
        patched_function.assert_called()


    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(ValueError):
            ExpoEst.estimate(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5.) # unsorted case
        with self.assertRaises(ValueError):
            ExpoEst.estimate(np.array([0., 0.5, 1.0, 2.5]), 5.) # first number zero and sorted
        with self.assertRaises(ValueError):
            ExpoEst.estimate(np.array([-0.5, -0.1, 1.0, 2.5]), 5.) # negative numbers but sorted
        with self.assertRaises(TypeError):
            ExpoEst.estimate(np.array([1, 2, 3, 4]), 5.) # int type array sorted positive
        with self.assertRaises(TypeError):
            ExpoEst.estimate([0.5, 1., 1.5, 2., 2.9], 5.) # not a numpy array

    def test_refuse_invalid_T(self):
        """ Check if invalid T input raises error """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(ValueError):
            ExpoEst.estimate(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 ) # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        ExpoEst = ExpHawkesProcessInference()
        ExpoEst.estimate(self.timestamps, self.T)
        self.assertTrue(hasattr(ExpoEst, "mu"))
        self.assertTrue(hasattr(ExpoEst, "eta"))
        self.assertTrue(hasattr(ExpoEst, "theta"))
        self.assertTrue(hasattr(ExpoEst, "logL"))
        self.assertTrue(hasattr(ExpoEst, "_grid_type"))
        self.assertTrue(hasattr(ExpoEst, "_params_set"))
        self.assertTrue(hasattr(ExpoEst, "_timestamps_set"))


class TestExpo_estimate_grid(TestCase):
    """ Class for testing the estimate_grid method """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.P = 3
        self.T = 50.0
        self.rng = np.random.default_rng(242)
        self.grid_size = 6

    def test_expo_calls_correct_mle(self):
        """ Check if mle function called correct and the correct number of times """
        with mock.patch("HawkesPyLib.inference.uvhp_expo_mle") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate_grid(self.timestamps, self.T, "random", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        with mock.patch("HawkesPyLib.inference.uvhp_expo_mle") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate_grid(self.timestamps, self.T, "equidistant", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        custom_grid = np.array([0.1, 0.2, 0.5, 0.6])
        with mock.patch("HawkesPyLib.inference.uvhp_expo_mle") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate_grid(self.timestamps, self.T, "custom", custom_grid=custom_grid)
        self.assertEqual(patched_function.call_count, len(custom_grid))


    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(ValueError):
            ExpoEst.estimate_grid(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5., "random") # unsorted case
        with self.assertRaises(ValueError):
            ExpoEst.estimate_grid(np.array([0., 0.5, 1.0, 2.5]), 5., "random") # first number zero and sorted
        with self.assertRaises(ValueError):
            ExpoEst.estimate_grid(np.array([-0.5, -0.1, 1.0, 2.5]), 5., "random") # negative numbers but sorted
        with self.assertRaises(TypeError):
            ExpoEst.estimate_grid(np.array([1, 2, 3, 4]), 5., "random") # int type array sorted positive
        with self.assertRaises(TypeError):
            ExpoEst.estimate_grid([0.5, 1., 1.5, 2., 2.9], 5., "random") # not a numpy array

    def test_refuse_invalid_T(self):
        """ Check if invalid T input raises error """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(ValueError):
            ExpoEst.estimate_grid(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 , "random") # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        ExpoEst = ExpHawkesProcessInference(rng=self.rng)
        ExpoEst.estimate_grid(self.timestamps, self.T, "random")
        self.assertTrue(hasattr(ExpoEst, "mu"))
        self.assertTrue(hasattr(ExpoEst, "eta"))
        self.assertTrue(hasattr(ExpoEst, "theta"))
        self.assertTrue(hasattr(ExpoEst, "logL"))
        self.assertTrue(hasattr(ExpoEst, "_grid_type"))
        self.assertTrue(hasattr(ExpoEst, "_params_set"))
        self.assertTrue(hasattr(ExpoEst, "_timestamps_set"))
        self.assertTrue(hasattr(ExpoEst, "_grid_size"))

class TestExpo_compensator(TestCase):
    """ Tests the method compensator """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.P = 3
        self.T = 50.0
        self.rng = np.random.default_rng(242)

    def test_params_set_check(self):
        """ test if method refuses if model paramters not set """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(Exception):
            ExpoEst.compensator()
    
    def test_calls_correct_comp(self):
        """ Check if compensator function called correct and the correct with correct params """
        with mock.patch("HawkesPyLib.processes.uvhp_expo_compensator") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate(self.timestamps, self.T)
            ExpoEst.compensator()
        patched_function.assert_called_once_with(self.timestamps, ExpoEst.mu, ExpoEst.eta, ExpoEst.theta)


class TestExpo_intensity(TestCase):
    """ Tests the method intensity """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 50.0
        self.step_size = 0.1

    def test_params_set_check(self):
        """ test if method refuses if model paramters not set """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(Exception):
            ExpoEst.intensity()
    
    def test_calls_correct_comp(self):
        """ Check if intensity function called correct and the correct with correct params """
        with mock.patch("HawkesPyLib.processes.uvhp_expo_intensity") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate(self.timestamps, self.T)
            ExpoEst.intensity(self.step_size)
        
        grid = generate_eval_grid(self.step_size, self.T)
        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.timestamps, patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(ExpoEst.mu, patched_function.call_args[0][2])
        self.assertEqual(ExpoEst.eta, patched_function.call_args[0][3])
        self.assertEqual(ExpoEst.theta, patched_function.call_args[0][4])

class TestExpo_kernel_values(TestCase):
    """ Tests the kernel_values method"""
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 50.0
        self.times = np.linspace(0, 2, 100)

    def test_params_set_check(self):
        """ test if method refuses if model parameters not set """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(Exception):
            ExpoEst.kernel_values(self.times)

    def test_calls_correct_kernel(self):
        """ Check if kernel values function called correct and the with correct params """
        with mock.patch("HawkesPyLib.processes.uvhp_expo_kernel") as patched_function:
            ExpoEst = ExpHawkesProcessInference()
            ExpoEst.estimate(self.timestamps, self.T)

            ExpoEst.kernel_values(self.times)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.times, patched_function.call_args[0][0])
        self.assertEqual(ExpoEst.eta, patched_function.call_args[0][1])
        self.assertEqual(ExpoEst.theta, patched_function.call_args[0][2])

class TestExpo_compute_logL(TestCase):
    """ Tests the compute logL method"""
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 50.

    def test_params_set_check(self):
        """ test if method refuses if model parameters not set """
        ExpoEst = ExpHawkesProcessInference()
        with self.assertRaises(Exception):
            ExpoEst.compute_logL()

    def test_calls_correct_logL(self):
        """ Check if compute_logL function called correct and the  correct params """
        ExpoEst = ExpHawkesProcessInference()
        ExpoEst.estimate(self.timestamps, self.T)

        logL2 = ExpoEst.compute_logL()
        logL1 = ExpoEst.logL
        self.assertEqual(logL1, logL2)


class TestExpoInference_getter(TestCase):
    """ Tests the getter equals estimate return and attribute"""
    def setUp(self):
        self.timestamps = timestamps
        self.T = timestamps[-1]
        self.rng = np.random.default_rng(242)

    def test_getter(self):
        ExpEstimator = ExpHawkesProcessInference(self.rng)
        mu, eta, theta = ExpEstimator.estimate(self.timestamps, self.T, return_params=True)

        # check attributes equal
        self.assertEqual(mu, ExpEstimator.mu)
        self.assertEqual(eta, ExpEstimator.eta)
        self.assertEqual(theta, ExpEstimator.theta)

        # check getter equal
        mu, eta, theta= ExpEstimator.get_params()
        self.assertEqual(mu, ExpEstimator.mu)
        self.assertEqual(eta, ExpEstimator.eta)
        self.assertEqual(theta, ExpEstimator.theta)
