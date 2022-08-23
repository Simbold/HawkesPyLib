import numpy as np
from unittest import TestCase, mock
from HawkesPyLib.inference import ApproxPowlawHawkesProcessEstimation
import os 

file_path = fpath = os.path.join(os.path.dirname(__file__), "timestamp_fixture.csv")
# np.loadtxt(file_path, delimiter=",", dtype=float)

class TestApproxPowlaw_estimate(TestCase):
    """ 
    Test the ApproxPowerlawEstimation Class.
    """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.m = 5.
        self.M = 4
        self.T = 50.0
        self.rng = np.random.default_rng(242)
        self.param_vec0 = np.array([0.6, 0.5, 0.4, 0.9])

    def test_powlaw_estimate_calls_correct_custom(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T, custom_param_vec0=True, param_vec0=self.param_vec0)
        patched_function.assert_called_once_with(self.timestamps, self.T, self.m, self.M, self.param_vec0)

    def test_powlaw_estimate_calls_correct_random(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T)
        patched_function.assert_called()

    def test_powlaw_cut_estimate_calls_correct_custom(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawCutEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawCutEst.estimate(self.timestamps, self.T, custom_param_vec0=True, param_vec0=self.param_vec0)
        patched_function.assert_called_once_with(self.timestamps, self.T, self.m, self.M, self.param_vec0)

    def test_powlaw_cut_estimate_calls_correct_random(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawCutEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawCutEst.estimate(self.timestamps, self.T)
        patched_function.assert_called()

    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5.) # unsorted case
        with self.assertRaises(ValueError):
            PowlawEst.estimate(np.array([0., 0.5, 1.0, 2.5]), 5.) # first number zero and sorted
        with self.assertRaises(ValueError):
            PowlawEst.estimate(np.array([-0.5, -0.1, 1.0, 2.5]), 5.) # negative numbers but sorted
        with self.assertRaises(TypeError):
            PowlawEst.estimate(np.array([1, 2, 3, 4]), 5.) # int type array sorted positive
        with self.assertRaises(TypeError):
            PowlawEst.estimate([0.5, 1., 1.5, 2., 2.9], 5.) # not a numpy array

    def test_refuse_invalid_T(self):
        """ Check if invalid T input raises error """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 ) # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        PowlawEst.estimate(self.timestamps, self.T)
        self.assertTrue(hasattr(PowlawEst, "mu"))
        self.assertTrue(hasattr(PowlawEst, "eta"))
        self.assertTrue(hasattr(PowlawEst, "alpha"))
        self.assertTrue(hasattr(PowlawEst, "tau0"))
        self.assertTrue(hasattr(PowlawEst, "logL"))
        self.assertTrue(hasattr(PowlawEst, "_grid_type"))
        self.assertTrue(hasattr(PowlawEst, "success_flag"))
        self.assertTrue(hasattr(PowlawEst, "_opt_result"))


class TestApproxPowlaw_estimate_grid(TestCase):
    """ Class for testing the estimate_grid method """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.m = 5.
        self.M = 4
        self.T = 50.0
        self.rng = np.random.default_rng(242)
        self.grid_size = 6

    def test_powlaw_calls_correct_mle(self):
        """ Check if mle function called correct and the correct number of times """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "random", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "equidistant", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        custom_grid = np.array([0.1, 0.2, 0.5, 0.6])
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "custom", custom_grid=custom_grid)
        self.assertEqual(patched_function.call_count, len(custom_grid))
    
    def test_powlaw_cut_calls_correct_mle(self):
        """ Check if mle function called correct and the correct number of times """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "random", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "equidistant", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        custom_grid = np.array([0.1, 0.2, 0.5, 0.6])
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "custom", custom_grid=custom_grid)
        self.assertEqual(patched_function.call_count, len(custom_grid))


    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate_grid(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5., "random") # unsorted case
        with self.assertRaises(ValueError):
            PowlawEst.estimate_grid(np.array([0., 0.5, 1.0, 2.5]), 5., "random") # first number zero and sorted
        with self.assertRaises(ValueError):
            PowlawEst.estimate_grid(np.array([-0.5, -0.1, 1.0, 2.5]), 5., "random") # negative numbers but sorted
        with self.assertRaises(TypeError):
            PowlawEst.estimate_grid(np.array([1, 2, 3, 4]), 5., "random") # int type array sorted positive
        with self.assertRaises(TypeError):
            PowlawEst.estimate_grid([0.5, 1., 1.5, 2., 2.9], 5., "random") # not a numpy array

    def test_refuse_invalid_T(self):
        """ Check if invalid T input raises error """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate_grid(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 , "random") # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        PowlawEst.estimate_grid(self.timestamps, self.T, "random")
        self.assertTrue(hasattr(PowlawEst, "mu"))
        self.assertTrue(hasattr(PowlawEst, "eta"))
        self.assertTrue(hasattr(PowlawEst, "alpha"))
        self.assertTrue(hasattr(PowlawEst, "tau0"))
        self.assertTrue(hasattr(PowlawEst, "logL"))
        self.assertTrue(hasattr(PowlawEst, "_grid_type"))
        self.assertTrue(hasattr(PowlawEst, "success_flag"))
        self.assertTrue(hasattr(PowlawEst, "_opt_result"))
        self.assertTrue(hasattr(PowlawEst, "_grid_size"))

class TestApproxPowlaw_compensator(TestCase):
    """ Tests the method compensator """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.m = 5.
        self.M = 4
        self.T = 50.0
        self.rng = np.random.default_rng(242)

    def test_success_flag_check(self):
        """ test if method refuses if model not yet succesfully estimated """
        PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(Exception):
            PowlawEst.compensator()
    
    def test_calls_correct_comp(self):
        """ Check if compensator function called correct and the correct with correct params """
        with mock.patch("HawkesPyLib.inference.uvhp_approx_powl_compensator") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.compensator()
        patched_function.assert_called_once_with(self.timestamps, PowlawEst.mu, PowlawEst.eta, PowlawEst.alpha, PowlawEst.tau0, self.m, self.M)

        with mock.patch("HawkesPyLib.inference.uvhp_approx_powl_cut_compensator") as patched_function:
            PowlawEst = ApproxPowlawHawkesProcessEstimation("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.compensator()
        patched_function.assert_called_once_with(self.timestamps, PowlawEst.mu, PowlawEst.eta, PowlawEst.alpha, PowlawEst.tau0, self.m, self.M)




