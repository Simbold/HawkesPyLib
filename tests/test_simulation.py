from unittest import TestCase, mock
import numpy as np
import sys, os
sys.path.insert(0,  os.path.abspath(os.path.join(os.getcwd(), os.pardir, "src/HawkesPyLib")))

from HawkesPyLib.simulation import (ApproxPowerlawHawkesProcessSimulation,
                                        ExpHawkesProcessSimulation,
                                        SumExpHawkesProcessSimulation,
                                        PoissonProcessSimulation)
from HawkesPyLib.core.intensity import generate_eval_grid


class TestHomogenousPoissonSimulation(TestCase):
    def setUp(self):
        self.mu = 10.
        self.T = 10000.
        self.seed = 123

    def test_calls_correct_simu(self):
        """ Test calls correct simulator """
        with mock.patch("HawkesPyLib.simulation.homogenous_poisson_simulator") as patched_function:
            PoissonSimulator = PoissonProcessSimulation(self.mu)
            timestamps = PoissonSimulator.simulate(self.T, self.seed)
        patched_function.assert_called_once_with(self.T, self.mu, seed = self.seed)

    def test_correct_number_samples(self):
        """ Test simulator generates expected number of samples """
        PoissonSimulator = PoissonProcessSimulation(self.mu)
        timestamps = PoissonSimulator.simulate(self.T)
        n_actual1 = PoissonSimulator.n_jumps
        n_actual2 = len(timestamps)

        n_expected = self.mu * self.T
        self.assertEqual(n_actual1, n_actual2)
        self.assertTrue((abs(n_expected - n_actual1) / n_expected) < 0.01) 
    
    def test_invalid_input(self):
        """ Test refueses invalid inputs """
        with self.assertRaises(ValueError):
            PoisSimulator = PoissonProcessSimulation(0)
        with self.assertRaises(ValueError):
            PoisSimulator = PoissonProcessSimulation(-2.5)

        PoisSimulator = PoissonProcessSimulation(self.mu)
        with self.assertRaises(ValueError):
            timestamps = PoisSimulator.simulate(0)
        with self.assertRaises(ValueError):
            timestamps = PoisSimulator.simulate(-4)
        with self.assertRaises(TypeError):
            timestamps = PoisSimulator.simulate("4.5")

        
class TestExpHawkesSimulation(TestCase):
    def setUp(self):
        self.mu = .5
        self.eta = .8
        self.theta = .01
        self.T = 10.
        self.seed = 242

    def test_sample_fixing(self):
        ExpoSimulator = ExpHawkesProcessSimulation(self.mu, self.eta, self.theta)
        actual = ExpoSimulator.simulate(self.T, self.seed)
        desired = np.array([2.3083755,  2.30908092, 2.31595136, 2.32882508,
                             9.52856952, 9.87906768, 9.88070317, 9.88219681])
        np.testing.assert_allclose(actual, desired, rtol=1e-8, atol=1e-8)
    
    def test_sample_fixing_cross_SumExpoSimulator(self):
        theta_vec = np.array([self.theta])
        SumExpoSimulator = SumExpHawkesProcessSimulation(self.mu, self.eta, theta_vec)
        SumExpo_timestamps = SumExpoSimulator.simulate(self.T, self.seed)

        ExpoSimulator = ExpHawkesProcessSimulation(self.mu, self.eta, self.theta)
        Expo_timestamps = ExpoSimulator.simulate(self.T, self.seed)
        np.testing.assert_allclose(SumExpo_timestamps, Expo_timestamps, rtol=1e-15, atol=1e-15)

    def test_calls_correct_intensity(self):
        """ Check if intensity function called correct and with the correct parameters """
        step_size = 0.1
        with mock.patch("HawkesPyLib.simulation.uvhp_expo_intensity") as patched_function:
            ExpoSimulator = ExpHawkesProcessSimulation(self.mu, self.eta, self.theta)
            timestamps = ExpoSimulator.simulate(self.T, self.seed)
            grid = generate_eval_grid(step_size, self.T)
            ExpoSimulator.intensity(step_size)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(timestamps,patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(self.mu, patched_function.call_args[0][2])
        self.assertEqual(self.eta, patched_function.call_args[0][3])
        self.assertEqual(self.theta, patched_function.call_args[0][4])


    def test_refuse_invalid_mu(self):
        """ Check if non positive mu and non float mu raise error """
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(-0.5, self.eta, self.theta)
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(0., self.eta, self.theta)
        with self.assertRaises(TypeError):
            ExpHawkesProcessSimulation("2", self.eta, self.theta)

    def test_refuse_invalid_eta(self):
        """ Check if values for eta outside the interval (0,1) are refused """
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(self.mu, -.5, self.theta)
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(self.mu, 1., self.theta)
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(self.mu, 0., self.theta)
            
    def test_refuse_invalid_theta(self):
        """ Check if non positive theta and non float theta raise error """
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(self.mu, self.eta, 0.)
        with self.assertRaises(ValueError):
            ExpHawkesProcessSimulation(self.mu, self.eta, -0.6)
        with self.assertRaises(TypeError):
            ExpHawkesProcessSimulation(self.mu, self.eta, "2")
            
    def test_refuse_invalid_T(self):
        """ Check if non positive T and non float int raise error """
        ExpoSimulator = ExpHawkesProcessSimulation(self.mu, self.eta, self.theta)
        with self.assertRaises(ValueError):
            ExpoSimulator.simulate(T=0.0)
        with self.assertRaises(ValueError):
            ExpoSimulator.simulate(T=-3.5)
        with self.assertRaises(TypeError):
            ExpoSimulator.simulate(T="3")

class TestSumExpHawkesSimulation(TestCase):
    def setUp(self) -> None:
        self.mu = .5
        self.eta=.8
        self.theta_vec = np.array([.1, .3, .5])
        self.T = 10.
        self.seed = 242

    def test_sample_fixing(self):
        SumExpoSimulator = SumExpHawkesProcessSimulation(self.mu, self.eta, self.theta_vec)
        actual = SumExpoSimulator.simulate(self.T, self.seed)
        desired = np.array([2.3083755,  2.32075025, 2.45105384, 2.70743681, 3.26019467,
                             3.27231931, 9.53121707, 9.56803776, 9.59677089])
        np.testing.assert_allclose(actual, desired, rtol=1e-8, atol=1e-8)

    def test_calls_correct_intensity(self):
        """ Check if intensity function called correct and the correct with correct params """
        step_size = 0.1
        with mock.patch("HawkesPyLib.simulation.uvhp_sum_expo_intensity") as patched_function:
            SumExpoSimulator = SumExpHawkesProcessSimulation(self.mu, self.eta, self.theta_vec)
            timestamps = SumExpoSimulator.simulate(self.T, self.seed)
            grid = generate_eval_grid(step_size, self.T)
            SumExpoSimulator.intensity(step_size)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(timestamps,patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(self.mu, patched_function.call_args[0][2])
        self.assertEqual(self.eta, patched_function.call_args[0][3])
        np.testing.assert_array_equal(self.theta_vec, patched_function.call_args[0][4])
    
    def test_refuse_invalid_mu(self):
        """ Check if non positive mu and non float mu raise error """
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(-0.5, self.eta, self.theta_vec)
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(0., self.eta, self.theta_vec)
        with self.assertRaises(TypeError):
            SumExpHawkesProcessSimulation("2", self.eta, self.theta_vec)

    def test_refuse_invalid_eta(self):
        """ Check if values for eta outside the interval (0,1) are refused """
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(self.mu, -.5, self.theta_vec)
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(self.mu, 1., self.theta_vec)
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(self.mu, 0., self.theta_vec)
            
    def test_refuse_invalid_theta_vec(self):
        """ Check if non positive theta and non float theta raise error """
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(self.mu, self.eta, np.array([0., 0.1, 0.2]))
        with self.assertRaises(ValueError):
            SumExpHawkesProcessSimulation(self.mu, self.eta, np.array([0.3, -0.6, 0.2]))
        with self.assertRaises(TypeError):
            SumExpHawkesProcessSimulation(self.mu, self.eta, np.array([1, 2, 3]))
        with self.assertRaises(TypeError): # refuses non numpy array
            SumExpHawkesProcessSimulation(self.mu, self.eta, [1., 2., 3.])
            
    def test_refuse_invalid_T(self):
        """ Check if non positive T and non float int raise error """
        SumExpoSimulator = SumExpHawkesProcessSimulation(self.mu, self.eta, self.theta_vec)
        with self.assertRaises(ValueError):
            SumExpoSimulator.simulate(T=0.0)
        with self.assertRaises(ValueError):
            SumExpoSimulator.simulate(T=-3.5)
        with self.assertRaises(TypeError):
            SumExpoSimulator.simulate(T="3")
    
            
class TestApproxPowlawHawkesSimulation(TestCase):
    def setUp(self) -> None:
        self.mu = .5
        self.eta = .8
        self.alpha=.3
        self.tau0=.001
        self.m = 5.
        self.M = 6
        self.T = 10.
        self.seed = 242

    def test_sample_fixing_powlaw(self):
        PowlawSimulator = ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)
        actual = PowlawSimulator.simulate(self.T, self.seed)
        desired = np.array([2.3083755,  2.3085288,  2.31024366, 2.40225152, 2.402425,   3.8940458, 3.89440117, 3.89474725,
                             4.40438068, 5.22852123, 7.30805841, 8.02551795, 8.02582305, 8.02907011, 8.03297857])
        np.testing.assert_allclose(actual, desired, rtol=1e-8, atol=1e-8)

    def test_sample_fixing_powlaw_cutoff(self):
        PowlawSimulator = ApproxPowerlawHawkesProcessSimulation("powlaw-cutoff", self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)
        actual = PowlawSimulator.simulate(self.T, self.seed)
        desired = np.array([2.3083755, 2.30851464, 2.31002491, 2.37521746, 6.05446178, 6.05488375, 6.05520548, 6.43494668,
                             7.04272411, 8.86119774, 9.44193708, 9.44885092, 9.45292938])
        np.testing.assert_allclose(actual, desired, rtol=1e-8, atol=1e-8)

    def test_calls_correct_intensity_powlaw(self):
        """ Check if compensator function called correct and the correct with correct params """
        step_size = 0.1
        with mock.patch("HawkesPyLib.simulation.uvhp_approx_powl_intensity") as patched_function:
            PowlawSimulator = ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)
            timestamps = PowlawSimulator.simulate(self.T, self.seed)
            grid = generate_eval_grid(step_size, self.T)
            PowlawSimulator.intensity(step_size)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(timestamps,patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(self.mu, patched_function.call_args[0][2])
        self.assertEqual(self.eta, patched_function.call_args[0][3])
        self.assertEqual(self.alpha, patched_function.call_args[0][4])
        self.assertEqual(self.tau0, patched_function.call_args[0][5])
        self.assertEqual(self.m, patched_function.call_args[0][6])
        self.assertEqual(self.M, patched_function.call_args[0][7])


    def test_calls_correct_intensity_powlaw_cutoff(self):
        """ Check if compensator function called correct and the correct with correct params """
        step_size = 0.1
        with mock.patch("HawkesPyLib.simulation.uvhp_approx_powl_cutoff_intensity") as patched_function:
            PowlawSimulator = ApproxPowerlawHawkesProcessSimulation("powlaw-cutoff", self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)
            timestamps = PowlawSimulator.simulate(self.T, self.seed)
            grid = generate_eval_grid(step_size, self.T)
            PowlawSimulator.intensity(step_size)

        patched_function.assert_called_once()
        np.testing.assert_array_equal(timestamps,patched_function.call_args[0][0])
        np.testing.assert_array_equal(grid, patched_function.call_args[0][1])
        self.assertEqual(self.mu, patched_function.call_args[0][2])
        self.assertEqual(self.eta, patched_function.call_args[0][3])
        self.assertEqual(self.alpha, patched_function.call_args[0][4])
        self.assertEqual(self.tau0, patched_function.call_args[0][5])
        self.assertEqual(self.m, patched_function.call_args[0][6])
        self.assertEqual(self.M, patched_function.call_args[0][7])

    def test_refuse_invalid_mu(self):
        """ Check if non positive mu and non float mu raise error """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", -0.5, self.eta, self.alpha, self.tau0, self.m, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", 0., self.eta, self.alpha, self.tau0, self.m, self.M)
        with self.assertRaises(TypeError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", "2", self.eta, self.alpha, self.tau0, self.m, self.M)

    def test_refuse_invalid_eta(self):
        """ Check if values for eta outside the interval (0,1) are refused """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, -.5, self.alpha, self.tau0, self.m, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, 1., self.alpha, self.tau0, self.m, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, 0., self.alpha, self.tau0, self.m, self.M)
            
    def test_refuse_invalid_alpha(self):
        """ Check if non positive values for alpha are refused """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, -0.5, self.tau0, self.m, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, 0., self.tau0, self.m, self.M)
        with self.assertRaises(TypeError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, "3", self.tau0, self.m, self.M)

    def test_refuse_invalid_tau0(self):
        """ Check if non positive values for tau0 are refused """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, -0.0002, self.m, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, 0., self.m, self.M)
        with self.assertRaises(TypeError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, "2", self.m, self.M)
    
    def test_refuse_invalid_m(self):
        """ Check if non positive values for m are refused """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, -0.5, self.M)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, 0.0, self.M)
        with self.assertRaises(TypeError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, "2", self.M)
    
    def test_refuse_invalid_M(self):
        """ Check if non positive non integer values for M are refused """
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, -4)
        with self.assertRaises(ValueError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, 0)
        with self.assertRaises(TypeError):
            ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, 3.0)

    
    def test_refuse_invalid_T(self):
        """ Check if non positive T and non float int raise error """
        PowlawSimulator = ApproxPowerlawHawkesProcessSimulation("powlaw", self.mu, self.eta, self.alpha, self.tau0, self.m, self.M)
        with self.assertRaises(ValueError):
            PowlawSimulator.simulate(T=0.0)
        with self.assertRaises(ValueError):
            PowlawSimulator.simulate(T=-3.5)
        with self.assertRaises(TypeError):
            PowlawSimulator.simulate(T="2")
   
