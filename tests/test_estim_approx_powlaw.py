import numpy as np
from unittest import TestCase, mock
from HawkesPyLib.inference import ApproxPowerlawHawkesProcessInference
from HawkesPyLib.processes import generate_eval_grid
import os 

file_path = fpath = os.path.join(os.path.dirname(__file__), "timestamp_fixture.csv")
timestamps = np.loadtxt(file_path, delimiter=",", dtype=float)

class TestApproxPowlaw_estimate(TestCase):
    """ 
    Test the ApproxPowerlawInference Class.
    """
    def setUp(self):
        self.timestamps = timestamps
        self.m = 5.
        self.M = 4
        self.T = 50.0
        self.rng = np.random.default_rng(242)
        self.param_vec0 = np.array([0.6, 0.5, 0.4, 0.9])

    def test_powlaw_estimate_calls_correct_custom(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T, custom_param_vec0=True, param_vec0=self.param_vec0)
        patched_function.assert_called_once_with(self.timestamps, self.T, self.m, self.M, self.param_vec0)

    def test_powlaw_estimate_calls_correct_random(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate(self.timestamps, self.T)
        patched_function.assert_called()

    def test_powlaw_cut_estimate_calls_correct_custom(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawCutEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawCutEst.estimate(self.timestamps, self.T, custom_param_vec0=True, param_vec0=self.param_vec0)
        patched_function.assert_called_once_with(self.timestamps, self.T, self.m, self.M, self.param_vec0)

    def test_powlaw_cut_estimate_calls_correct_random(self):
        """ Test if .estimate() calls correct mle with custom param_vec0 """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawCutEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawCutEst.estimate(self.timestamps, self.T)
        patched_function.assert_called()

    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
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
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 ) # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        PowlawEst.estimate(self.timestamps, self.T)
        self.assertTrue(hasattr(PowlawEst, "mu"))
        self.assertTrue(hasattr(PowlawEst, "eta"))
        self.assertTrue(hasattr(PowlawEst, "alpha"))
        self.assertTrue(hasattr(PowlawEst, "tau0"))
        self.assertTrue(hasattr(PowlawEst, "logL"))
        self.assertTrue(hasattr(PowlawEst, "_grid_type"))
        self.assertTrue(hasattr(PowlawEst, "_params_set"))
        self.assertTrue(hasattr(PowlawEst, "_timestamps_set"))

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
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "random", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "equidistant", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        custom_grid = np.array([0.1, 0.2, 0.5, 0.6])
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "custom", custom_grid=custom_grid)
        self.assertEqual(patched_function.call_count, len(custom_grid))
    
    def test_powlaw_cut_calls_correct_mle(self):
        """ Check if mle function called correct and the correct number of times """
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "random", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "equidistant", self.grid_size)
        self.assertEqual(patched_function.call_count, self.grid_size)

        custom_grid = np.array([0.1, 0.2, 0.5, 0.6])
        with mock.patch("HawkesPyLib.inference.uvhp_powlaw_cut_mle") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M, self.rng)
            PowlawEst.estimate_grid(self.timestamps, self.T, "custom", custom_grid=custom_grid)
        self.assertEqual(patched_function.call_count, len(custom_grid))


    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
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
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(ValueError):
            PowlawEst.estimate_grid(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 , "random") # T smaller than last timestamp

    def test_check_attributes(self):
        """ Check if all attributes set after succeseful estimation """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        PowlawEst.estimate_grid(self.timestamps, self.T, "random")
        self.assertTrue(hasattr(PowlawEst, "mu"))
        self.assertTrue(hasattr(PowlawEst, "eta"))
        self.assertTrue(hasattr(PowlawEst, "alpha"))
        self.assertTrue(hasattr(PowlawEst, "tau0"))
        self.assertTrue(hasattr(PowlawEst, "logL"))
        self.assertTrue(hasattr(PowlawEst, "_grid_type"))
        self.assertTrue(hasattr(PowlawEst, "_params_set"))
        self.assertTrue(hasattr(PowlawEst, "_grid_size"))
        self.assertTrue(hasattr(PowlawEst, "_timestamps_set"))


class TestApproxPowlaw_compensator(TestCase):
    """ Tests the method compensator """
    def setUp(self):
        self.timestamps = timestamps
        self.m = 5.
        self.M = 4
        self.T = 50.0

    def test_success_flag_check(self):
        """ test if method refuses if model not yet succesfully estimated """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M)
        with self.assertRaises(Exception):
            PowlawEst.compensator()
    
    def test_calls_correct_compensator(self):
        """ Check if compensator function called correct and the correct with correct params """
        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_compensator") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.compensator()
        patched_function.assert_called_once_with(self.timestamps, PowlawEst.mu, PowlawEst.eta, PowlawEst.alpha, PowlawEst.tau0, self.m, self.M)

        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_cut_compensator") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.compensator()
        patched_function.assert_called_once_with(self.timestamps, PowlawEst.mu, PowlawEst.eta, PowlawEst.alpha, PowlawEst.tau0, self.m, self.M)


class TestApproxPowlaw_intensity(TestCase):
    """ Tests the method intensity """
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 50.0
        self.step_size = 0.1
        self.m = 5.
        self.M = 4
        

    def test_params_set_check(self):
        """ test if method refuses if model paramters not set """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
        with self.assertRaises(Exception):
            PowlawEst.intensity()
    
    def test_calls_correct_intensity(self):
        """ Check if intensity function called correct and the correct with correct params """
        # cutoff
        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_cutoff_intensity") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.intensity(self.step_size)
        
        grid = generate_eval_grid(self.step_size, self.T)
        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.timestamps, patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(PowlawEst.mu, patched_function.call_args[0][2])
        self.assertEqual(PowlawEst.eta, patched_function.call_args[0][3])
        self.assertEqual(PowlawEst.alpha, patched_function.call_args[0][4])
        self.assertEqual(PowlawEst.tau0, patched_function.call_args[0][5])

        # no cutoff
        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_intensity") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)
            PowlawEst.intensity(self.step_size)
        
        grid = generate_eval_grid(self.step_size, self.T)
        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.timestamps, patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(PowlawEst.mu, patched_function.call_args[0][2])
        self.assertEqual(PowlawEst.eta, patched_function.call_args[0][3])
        self.assertEqual(PowlawEst.alpha, patched_function.call_args[0][4])
        self.assertEqual(PowlawEst.tau0, patched_function.call_args[0][5])


class TestApproxPowl_kernel_values(TestCase):
    """ Tests the kernel_values method"""
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 10.0
        self.step_size = 0.1
        self.m = 5.
        self.M = 4
        self.times = np.linspace(0, 2, 100)

    def test_params_set_check(self):
        """ test if method refuses if model parameters not set """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
        with self.assertRaises(Exception):
            PowlawEst.kernel_values(self.times)

    def test_calls_correct_kernel(self):
        """ Check if kernel values function called correct and the with correct params """
        # cutoff
        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_cutoff_kernel") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)

            PowlawEst.kernel_values(self.times)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.times, patched_function.call_args[0][0])
        self.assertEqual(PowlawEst.eta, patched_function.call_args[0][1])
        self.assertEqual(PowlawEst.alpha, patched_function.call_args[0][2])
        self.assertEqual(PowlawEst.tau0, patched_function.call_args[0][3])
        self.assertEqual(self.m, patched_function.call_args[0][4])
        self.assertEqual(self.M, patched_function.call_args[0][5])

        # no cutoff
        with mock.patch("HawkesPyLib.processes.uvhp_approx_powl_kernel") as patched_function:
            PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M)
            PowlawEst.estimate(self.timestamps, self.T)

            PowlawEst.kernel_values(self.times)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(self.times, patched_function.call_args[0][0])
        self.assertEqual(PowlawEst.eta, patched_function.call_args[0][1])
        self.assertEqual(PowlawEst.alpha, patched_function.call_args[0][2])
        self.assertEqual(PowlawEst.tau0, patched_function.call_args[0][3])
        self.assertEqual(self.m, patched_function.call_args[0][4])
        self.assertEqual(self.M, patched_function.call_args[0][5])


class TestAproxPowl_compute_logL(TestCase):
    """ Tests the compute logL method"""
    def setUp(self):
        self.timestamps = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                                    3.27231931, 9.53121707, 9.56803776, 9.59677089]) 
        self.T = 10.0
        self.step_size = 0.1
        self.m = 5.
        self.M = 4
        self.rng = np.random.default_rng(242)
        self.times = np.linspace(0, 2, 100)

    def test_params_set_check(self):
        """ test if method refuses if model parameters not set """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        with self.assertRaises(Exception):
            PowlawEst.compute_logL()

    def test_calls_correct_logL(self):
        """ Check if compute_logL function called correct and the  correct params """
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        PowlawEst.estimate(self.timestamps, self.T)

        logL2 = PowlawEst.compute_logL()
        logL1 = PowlawEst.logL
        self.assertEqual(logL1, logL2)


        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw-cutoff", self.m, self.M)
        PowlawEst.estimate(self.timestamps, self.T)

        logL2 = PowlawEst.compute_logL()
        logL1 = PowlawEst.logL
        self.assertEqual(logL1, logL2)

class TestApproxPowlaw_getter(TestCase):
    """ Tests the getter equals estimate return and attribute"""
    def setUp(self):
        self.timestamps = timestamps
        self.T = timestamps[-1]
        self.m = 5.
        self.M = 4
        self.rng = np.random.default_rng(242)

    def test_getter(self):
        PowlawEst = ApproxPowerlawHawkesProcessInference("powlaw", self.m, self.M, self.rng)
        mu, eta, alpha, tau0 = PowlawEst.estimate(self.timestamps, self.T, return_params=True)

        # check attributes equal
        self.assertEqual(mu, PowlawEst.mu)
        self.assertEqual(eta, PowlawEst.eta)
        self.assertEqual(alpha, PowlawEst.alpha)
        self.assertEqual(tau0, PowlawEst.tau0)

        # check getter equal
        mu, eta, alpha, tau0, m, M = PowlawEst.get_params()
        self.assertEqual(mu, PowlawEst.mu)
        self.assertEqual(eta, PowlawEst.eta)
        self.assertEqual(alpha, PowlawEst.alpha)
        self.assertEqual(tau0, PowlawEst.tau0)
        self.assertEqual(m, self.m)
        self.assertEqual(M, self.M)

