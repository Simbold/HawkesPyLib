import numpy as np
from unittest import TestCase
from HawkesPyLib.inference import PoissonProcessInference
from HawkesPyLib.simulation import PoissonProcessSimulation

class TestPoissonInference(TestCase):
    """
    Test the Poisson inference class
    """
    def setUp(self):
        self.mu = 10.
        self.T = 10000
        PoisSimulator = PoissonProcessSimulation(self.mu)
        self.timestamps = PoisSimulator.simulate(self.T)

    def test_refuse_invalid_timestamp(self):
        """ Check if invalid timestamp input raises error """
        PoisEst = PoissonProcessInference()
        with self.assertRaises(ValueError):
            PoisEst.estimate(np.array([0.5, 1.0, 2.5, 0.1, 3.0, 2.0]), 5.) # unsorted case
        with self.assertRaises(ValueError):
            PoisEst.estimate(np.array([0., 0.5, 1.0, 2.5]), 5.) # first number zero and sorted
        with self.assertRaises(ValueError):
            PoisEst.estimate(np.array([-0.5, -0.1, 1.0, 2.5]), 5.) # negative numbers but sorted
        with self.assertRaises(TypeError):
            PoisEst.estimate(np.array([1, 2, 3, 4]), 5.) # int type array sorted positive
        with self.assertRaises(TypeError):
            PoisEst.estimate([0.5, 1., 1.5, 2., 2.9], 5.) # not a numpy array

    def test_refuse_invalid_T(self):
        """ Check if invalid T input raises error """
        PoisEst = PoissonProcessInference()
        with self.assertRaises(ValueError):
            PoisEst.estimate(np.array([0.5, 1., 1.5, 2., 2.9]), 2.8 ) # T smaller than last timestamp

    def test_estimate(self):
        """ test Poisson estimate method """
        PoisEst = PoissonProcessInference()
        mu_actual, logL = PoisEst.estimate(self.timestamps, self.T, return_result=True)
        
        self.assertTrue((abs(mu_actual - self.mu) / self.mu) < 0.01)

        # test attributes set correctly
        self.assertEqual(mu_actual, PoisEst.mu)
        self.assertEqual(logL, PoisEst.logL)
        self.assertTrue(PoisEst._estimated_flag)

