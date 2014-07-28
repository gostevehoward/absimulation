#!/usr/bin/env python

import unittest

import simulation

def state(visitors_per_bucket, baseline_conversions, treatment_conversions):
    return simulation.ExperimentState(
        baseline_conversions,
        visitors_per_bucket - baseline_conversions,
        treatment_conversions,
        visitors_per_bucket - treatment_conversions,
    )

class ChisqDecisionTest(unittest.TestCase):
    def test_sample_size_calculation(self):
        # test values from http://www.stat.ubc.ca/~rollin/stats/ssize/b2.html
        self.assertEqual(
            14751,
            simulation.ChisqDecision(0.95, 0.1).necessary_sample_size_per_bucket(0.1),
        )
        self.assertEqual(
            9780,
            simulation.ChisqDecision(0.85, 0.1).necessary_sample_size_per_bucket(0.1),
        )
        self.assertEqual(
            2507,
            simulation.ChisqDecision(0.95, 0.25).necessary_sample_size_per_bucket(0.1),
        )
        self.assertEqual(
            6510,
            simulation.ChisqDecision(0.95, 0.1).necessary_sample_size_per_bucket(0.2),
        )

    def test_decision(self):
        baseline_rate = 0.5
        chisq_decision = simulation.ChisqDecision(0.95, 0.1)

        # sanity checks
        self.assertEqual('keep running', chisq_decision.decision(state(20, 7, 10), baseline_rate))
        self.assertEqual(
            'baseline',
            chisq_decision.decision(state(10000, 5000, 5000), baseline_rate),
        )
        self.assertEqual(
            'baseline',
            chisq_decision.decision(state(10000, 6000, 4000), baseline_rate),
        )
        self.assertEqual(
            'treatment',
            chisq_decision.decision(state(10000, 4000, 6000), baseline_rate),
        )

        # some close calls, using Chi-squared values from
        # http://www.graphpad.com/quickcalcs/contingency1.cfm
        self.assertEqual(
            'baseline',
            chisq_decision.decision(state(10000, 5000, 5100), baseline_rate),
        )
        self.assertEqual(
            'treatment',
            chisq_decision.decision(state(10000, 5000, 5150), baseline_rate),
        )

class BayesianDecisionTest(unittest.TestCase):
    def setUp(self):
        self.decision = simulation.BayesianDecision(0.01)

    def test_posterior_probability_treatment_is_better(self):
        # sanity checks
        self.assertAlmostEqual(
            1,
            self.decision.posterior_probability_treatment_is_better(state(1000, 1, 999)),
        )
        self.assertAlmostEqual(
            0,
            self.decision.posterior_probability_treatment_is_better(state(1000, 999, 1)),
        )
        self.assertAlmostEqual(
            0.5,
            self.decision.posterior_probability_treatment_is_better(state(100, 50, 50)),
        )
        self.assertGreater(
            self.decision.posterior_probability_treatment_is_better(state(100, 50, 51)),
            0.5,
        )
        self.assertLess(
            self.decision.posterior_probability_treatment_is_better(state(100, 50, 49)),
            0.5,
        )

        # some less obvious ones which might be wrong (generated using my own implementation), but
        # useful for catching unintended changes at least
        self.assertAlmostEqual(
            0.92318343,
            self.decision.posterior_probability_treatment_is_better(state(1000, 100, 120)),
        )
        self.assertAlmostEqual(
            0.22343071,
            self.decision.posterior_probability_treatment_is_better(state(1000, 100, 90)),
        )

    def test_expected_loss_from_choosing_treatment(self):
        # sanity checks
        self.assertAlmostEqual(
            0.9,
            self.decision.expected_loss_from_choosing_treatment(state(1000, 950, 50)),
            places=2,
        )
        self.assertAlmostEqual(
            0,
            self.decision.expected_loss_from_choosing_treatment(state(1000, 1, 999)),
        )

        # some values from Chris Stucchio's numerical integration code
        # https://gist.github.com/stucchio/9090456
        # see stucchio.py in this repository
        self.assertAlmostEqual(
            0.017,
            self.decision.expected_loss_from_choosing_treatment(state(100, 10, 10)),
            places=3,
        )
        self.assertAlmostEqual(
            0.0005,
            self.decision.expected_loss_from_choosing_treatment(state(100, 10, 20)),
            places=4,
        )
        self.assertAlmostEqual(
            0.1,
            self.decision.expected_loss_from_choosing_treatment(state(100, 20, 10)),
            places=1,
        )

if __name__ == '__main__':
    unittest.main()
