#!/usr/bin/env python

import argparse
import collections
import csv
import itertools
import math
import multiprocessing
import random
import sys

import numpy
import scipy.special
import scipy.stats

BASELINE = 'baseline'
TREATMENT = 'treatment'
KEEP_RUNNING = 'keep running'

MEDIAN_ODDS_RATIO = 0.85
ODDS_RATIO_SPREAD = 1.2
DELAY_IN_VISITORS_TO_DESIGN_NEW_EXPERIMENT = 5000

def log_odds(rate):
    return math.log(rate / (1 - rate))

def choose_log_odds_lifts():
    return numpy.random.normal(math.log(MEDIAN_ODDS_RATIO), math.log(ODDS_RATIO_SPREAD), 10000)

def get_treatment_rate(baseline_rate, log_odds_lift):
    baseline_log_odds = log_odds(baseline_rate)
    new_log_odds = baseline_log_odds + log_odds_lift
    return 1 / (1 + math.exp(-new_log_odds))

StateTuple = collections.namedtuple(
    'StateTuple',
    ('baseline_conversions', 'baseline_nonconversions',
     'treatment_conversions', 'treatment_nonconversions'),
)
class ExperimentState(StateTuple):
    @classmethod
    def empty_state(cls):
        return cls(0, 0, 0, 0)

    def add_visitors(self, baseline_conversions=0, baseline_nonconversions=0,
                     treatment_conversions=0, treatment_nonconversions=0):
        return self._replace(
            baseline_conversions=self.baseline_conversions + baseline_conversions,
            baseline_nonconversions=self.baseline_nonconversions + baseline_nonconversions,
            treatment_conversions=self.treatment_conversions + treatment_conversions,
            treatment_nonconversions=self.treatment_nonconversions + treatment_nonconversions,
        )

    def flip_baseline_and_treatment(self):
        return ExperimentState(
            baseline_conversions=self.treatment_conversions,
            baseline_nonconversions=self.treatment_nonconversions,
            treatment_conversions=self.baseline_conversions,
            treatment_nonconversions=self.baseline_nonconversions,
        )

    @property
    def baseline_visitors(self):
        return self.baseline_conversions + self.baseline_nonconversions

    @property
    def treatment_visitors(self):
        return self.treatment_conversions + self.treatment_nonconversions

    @property
    def num_visitors_per_bucket(self):
        assert self.baseline_visitors == self.treatment_visitors, (
            self.baseline_visitors, self.treatment_visitors
        )
        return self.baseline_visitors

    def contingency_table(self):
        return numpy.array([
            [self.baseline_conversions, self.baseline_nonconversions],
            [self.treatment_conversions, self.treatment_nonconversions],
        ])

class ChisqDecision(object):
    def __init__(self, significance_level, relative_lift):
        self._significance_level = significance_level
        assert relative_lift > 0, relative_lift
        self._relative_lift = relative_lift
        self._power = 0.8

    def description(self):
        return 'Chisq, {:.0%} significance, {:.1%} relative lift'.format(
            self._significance_level,
            self._relative_lift,
        )

    def parameters(self):
        return 'Chisq', self._significance_level, self._relative_lift

    def necessary_sample_size_per_bucket(self, baseline_rate):
        # this can be derived using the normal approximation to the binomial
        lift = min(baseline_rate, 1 - baseline_rate) * self._relative_lift
        treatment_rate = baseline_rate + lift
        expected_difference = treatment_rate - baseline_rate
        expected_pooled_rate = (baseline_rate + treatment_rate) / 2
        significance_z_score = scipy.stats.norm.isf((1 - self._significance_level) / 2)
        power_z_score = scipy.stats.norm.isf(1 - self._power)

        square_root_of_sample_size = (
            significance_z_score * math.sqrt(2 * expected_pooled_rate * (1 - expected_pooled_rate))
            + power_z_score * math.sqrt(
                baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
            )
        ) / expected_difference
        return int(math.ceil(square_root_of_sample_size**2))

    def get_step_visitors_per_bucket(self, baseline_rate, current_state):
        if min(current_state.contingency_table().flat) < 5:
            return 4

        sample_size_per_bucket = self.necessary_sample_size_per_bucket(baseline_rate)
        return sample_size_per_bucket - current_state.num_visitors_per_bucket

    def decision(self, state, baseline_rate):
        contingency_table = state.contingency_table()
        if min(contingency_table.flat) < 5:
            return KEEP_RUNNING

        sample_size = self.necessary_sample_size_per_bucket(baseline_rate)
        if state.num_visitors_per_bucket < sample_size:
            return KEEP_RUNNING

        _, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
        is_treatment_winner = (
            state.treatment_conversions > state.baseline_conversions
            and p_value < (1 - self._significance_level)
        )

        return TREATMENT if is_treatment_winner else BASELINE

class BayesianDecision(object):
    log_beta = scipy.special.betaln

    def __init__(self, minimum_relative_lift, verbose=False):
        self._minimum_relative_lift = minimum_relative_lift

    def description(self):
        return 'Bayesian, {:.2%} minimum relative lift'.format(self._minimum_relative_lift)

    def parameters(self):
        return 'Bayesian', self._minimum_relative_lift, 'NA'

    def get_step_visitors_per_bucket(self, baseline_rate, current_state):
        # grow visitors by 5% at a time
        return max(10, round(0.05 * current_state.num_visitors_per_bucket))

    def posterior_probability_treatment_is_better(self, state):
        '''
        http://www.evanmiller.org/bayesian-ab-testing.html
        '''
        treatment_alpha = state.treatment_conversions + 1
        treatment_beta = state.treatment_nonconversions + 1
        baseline_alpha = state.baseline_conversions + 1
        baseline_beta = state.baseline_nonconversions + 1

        i_values = numpy.arange(treatment_alpha)
        sum_result = numpy.sum(
            numpy.exp(
                self.log_beta(baseline_alpha + i_values, treatment_beta + baseline_beta)
                - self.log_beta(1 + i_values, treatment_beta)
                - self.log_beta(baseline_alpha, baseline_beta)
            ) / (treatment_beta + i_values)
        )

        return sum_result

    def expected_loss_from_choosing_treatment(self, state):
        first_term = math.exp(
            self.log_beta(state.baseline_conversions + 2, state.baseline_nonconversions + 1)
            - self.log_beta(state.baseline_conversions + 1, state.baseline_nonconversions + 1)
        ) * self.posterior_probability_treatment_is_better(
            state.add_visitors(baseline_conversions=1).flip_baseline_and_treatment()
        )
        second_term = math.exp(
            self.log_beta(state.treatment_conversions + 2, state.treatment_nonconversions + 1)
            - self.log_beta(state.treatment_conversions + 1, state.treatment_nonconversions + 1)
        ) * self.posterior_probability_treatment_is_better(
            state.add_visitors(treatment_conversions=1).flip_baseline_and_treatment()
        )
        return first_term - second_term

    def decision(self, state, baseline_rate):
        contingency_table = state.contingency_table()
        if min(contingency_table.flat) < 5:
            return KEEP_RUNNING

        choose_treatment_loss = self.expected_loss_from_choosing_treatment(state)
        choose_baseline_loss = self.expected_loss_from_choosing_treatment(
            state.flip_baseline_and_treatment()
        )
        assert 0 <= choose_treatment_loss <= 1, (state, choose_treatment_loss)
        assert 0 <= choose_baseline_loss <= 1, (state, choose_baseline_loss)

        minimum_lift = min(baseline_rate, 1 - baseline_rate) * self._minimum_relative_lift
        if min(abs(choose_treatment_loss), abs(choose_baseline_loss)) > minimum_lift:
            return KEEP_RUNNING
        elif state.treatment_conversions > state.baseline_conversions:
            assert choose_treatment_loss < choose_baseline_loss, state
            return TREATMENT
        else:
            assert choose_baseline_loss <= choose_treatment_loss, state
            return BASELINE

class ExperimentRecord(object):
    def __init__(self, initial_rate):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.loss_from_errors = 0
        self.final_rate = None
        self.path = [(0, initial_rate)] # list of (num_visitors, conversion_rate)

    def add(self, treatment_won, true_baseline_rate, true_treatment_rate, num_visitors_seen):
        self.final_rate = true_treatment_rate if treatment_won else true_baseline_rate
        self.path.append((num_visitors_seen, self.final_rate))

        treatment_is_actually_better = (true_treatment_rate > true_baseline_rate)
        if treatment_is_actually_better and treatment_won:
            self.true_positives += 1
        elif treatment_is_actually_better and not treatment_won:
            self.false_negatives += 1
        elif not treatment_is_actually_better and treatment_won:
            self.false_positives += 1
        elif not treatment_is_actually_better and not treatment_won:
            self.true_negatives += 1

        if treatment_won != treatment_is_actually_better:
            # we made an error
            self.loss_from_errors += abs(
                log_odds(true_baseline_rate) - log_odds(true_treatment_rate)
            )

    def total_experiments_run(self):
        return (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )

def debug_start_experiment(verbose, baseline_rate, treatment_rate, total_visitors_so_far):
    if verbose:
        sys.stdout.write(
            'Baseline {:.2%}, treatment {:.2%}, {} visitors so far\n'.format(
                baseline_rate,
                treatment_rate,
                total_visitors_so_far,
            )
        )

def debug_finish_experiment(verbose, winner, experiment_state):
    if verbose:
        print '    Winner {}, {} samples, {} baseline vs {} treatment'.format(
            winner,
            experiment_state.num_visitors_per_bucket * 2,
            experiment_state.baseline_conversions,
            experiment_state.treatment_conversions,
        )

def run_one_simulation(inital_rate, choose_new_rate_fn, num_visitors, decision_type, verbose=False):
    baseline_rate = inital_rate
    empirical_baseline_rate = baseline_rate
    treatment_rate = choose_new_rate_fn(baseline_rate)
    debug_start_experiment(verbose, baseline_rate, treatment_rate, 0)
    current_state = ExperimentState.empty_state()
    record = ExperimentRecord(inital_rate)

    num_visitors_seen = DELAY_IN_VISITORS_TO_DESIGN_NEW_EXPERIMENT
    while num_visitors_seen < num_visitors:
        step_visitors_per_bucket = decision_type.get_step_visitors_per_bucket(
            empirical_baseline_rate,
            current_state,
        )
        step_visitors_per_bucket = min(step_visitors_per_bucket, num_visitors - num_visitors_seen)

        baseline_conversions = numpy.random.binomial(step_visitors_per_bucket, baseline_rate)
        treatment_conversions = numpy.random.binomial(step_visitors_per_bucket, treatment_rate)
        current_state = current_state.add_visitors(
            baseline_conversions=baseline_conversions,
            baseline_nonconversions=step_visitors_per_bucket - baseline_conversions,
            treatment_conversions=treatment_conversions,
            treatment_nonconversions=step_visitors_per_bucket - treatment_conversions,
        )
        num_visitors_seen += 2 * step_visitors_per_bucket

        winner = decision_type.decision(current_state, empirical_baseline_rate)
        if winner != KEEP_RUNNING:
            debug_finish_experiment(verbose, winner, current_state)
            record.add(winner == TREATMENT, baseline_rate, treatment_rate, num_visitors_seen)
            if winner == TREATMENT:
                baseline_rate = treatment_rate
                empirical_baseline_rate = (
                    1.0 * current_state.treatment_conversions / current_state.treatment_visitors
                )
            treatment_rate = choose_new_rate_fn(baseline_rate)
            num_visitors_seen += DELAY_IN_VISITORS_TO_DESIGN_NEW_EXPERIMENT
            current_state = ExperimentState.empty_state()
            debug_start_experiment(verbose, baseline_rate, treatment_rate, num_visitors_seen)

    return record

class NullCsvWriter(object):
    def writerow(self, row):
        pass

SimulationConfig = collections.namedtuple(
    'SimulationConfig',
    ('decision_types', 'seed', 'verbose', 'path_csv_writer'),
)

def set_random_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)

def run_simulation(simulation_config):
    set_random_seeds(simulation_config.seed)
    log_odds_lifts = choose_log_odds_lifts()
    results_for_each_decision_type = {}

    for decision_type in simulation_config.decision_types:
        log_odds_lift_iter = iter(log_odds_lifts)
        def choose_new_rate(baseline_rate):
            return get_treatment_rate(baseline_rate, log_odds_lift_iter.next())

        set_random_seeds(simulation_config.seed + 1000000)
        record = run_one_simulation(
            0.1,
            choose_new_rate,
            1000000,
            decision_type,
            verbose=simulation_config.verbose,
        )
        for num_visitors_seen, rate in record.path:
            simulation_config.path_csv_writer.writerow(
                [simulation_config.seed, decision_type.description(), num_visitors_seen, rate]
            )
        results_for_each_decision_type[decision_type] = record

    return simulation_config.seed, results_for_each_decision_type

def run_many_simulations(base_config, writer, num_simulations, multiprocess=True):
    writer.writerow(
        [
            'simulation.index',
            'decision.type',
            'test.name',
            'first.parameter',
            'second.parameter',
            'final.rate',
            'total.experiments.run',
            'true.positives',
            'false.positives',
            'true.negatives',
            'false.negatives',
            'loss.from.errors'
        ],
    )

    if multiprocess:
        pool = multiprocessing.Pool(processes=4)
        map_fn = pool.imap_unordered
    else:
        map_fn = itertools.imap

    configs = [base_config._replace(seed=seed) for seed in xrange(num_simulations)]
    results_for_each_simulation = map_fn(run_simulation, configs)
    for seed, results_for_each_decision_type in results_for_each_simulation:
        sys.stderr.write('.')
        for decision_type, record in results_for_each_decision_type.iteritems():
            test_name, first_parameter, second_parameter = decision_type.parameters()
            writer.writerow(
                [
                    seed,
                    decision_type.description(),
                    test_name,
                    first_parameter,
                    second_parameter,
                    record.final_rate,
                    record.total_experiments_run(),
                    record.true_positives,
                    record.false_positives,
                    record.true_negatives,
                    record.false_negatives,
                    record.loss_from_errors,
                ]
            )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--write-path-csv', action='store_true')
    parser.add_argument('--multiprocess', action='store_true')
    parser.add_argument('--num-simulations', type=int, default=1000)
    return parser.parse_args()

def main():
    options = parse_arguments()

    decision_types = [
        ChisqDecision(significance, min_lift)
        for significance in (0.1, 0.25, 0.5, 0.75, 0.9)
        for min_lift in (0.05, 0.1, 0.2, 0.5)
    ]
    decision_types += [
        BayesianDecision(min_lift, verbose=options.verbose)
        for min_lift in (0.0005, 0.002, 0.01, 0.03)
    ]

    writer = csv.writer(sys.stdout)
    if options.write_path_csv:
        writer.writerow(['seed', 'decision.type', 'num.visitors.seen', 'rate'])
        path_writer = writer
        summary_writer = NullCsvWriter()
    else:
        path_writer = NullCsvWriter()
        summary_writer = writer

    base_config = SimulationConfig(
        decision_types=decision_types,
        seed=None,
        verbose=options.verbose,
        path_csv_writer=path_writer,
    )
    run_many_simulations(
        base_config,
        summary_writer,
        options.num_simulations,
        multiprocess=options.multiprocess,
    )

if __name__ == '__main__':
    main()
