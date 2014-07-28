import math
import time

import numpy
import scipy.special

log_beta = scipy.special.betaln

def posterior_probability_treatment_is_better_slow(
    treatment_conversions, treatment_nonconversions, baseline_conversions, baseline_nonconversions
):
    '''
    http://www.evanmiller.org/bayesian-ab-testing.html
    '''
    treatment_alpha = treatment_conversions + 1
    treatment_beta = treatment_nonconversions + 1
    baseline_alpha = baseline_conversions + 1
    baseline_beta = baseline_nonconversions + 1

    sum_result = 0
    for i in xrange(baseline_alpha):
        sum_result += math.exp(
            log_beta(treatment_alpha + i, treatment_beta + baseline_beta)
            - math.log(baseline_beta + i)
            - log_beta(1 + i, baseline_beta)
            - log_beta(treatment_alpha, treatment_beta)
        )

    return 1 - sum_result

def posterior_probability_treatment_is_better_fast(
    treatment_conversions, treatment_nonconversions, baseline_conversions, baseline_nonconversions
):
    '''
    http://www.evanmiller.org/bayesian-ab-testing.html
    '''
    treatment_alpha = treatment_conversions + 1
    treatment_beta = treatment_nonconversions + 1
    baseline_alpha = baseline_conversions + 1
    baseline_beta = baseline_nonconversions + 1

    i_values = numpy.arange(baseline_alpha)
    sum_result = numpy.sum(
        numpy.exp(
            log_beta(baseline_alpha + i_values, treatment_beta + baseline_beta)
            - log_beta(1 + i_values, baseline_beta)
            - log_beta(treatment_alpha, treatment_beta)
        ) / (baseline_beta + i_values)
    )

    return 1 - sum_result

def time_function_us(function, sample_size, repetitions=1000):
    arguments = (
        int(.1 * sample_size),
        int(.9 * sample_size),
        int(.2 * sample_size),
        int(.8 * sample_size),
    )

    # warm up
    for _ in xrange(10):
        function(*arguments)

    total_time = 0
    for _ in xrange(repetitions):
        start = time.time()
        function(*arguments)
        total_time += time.time() - start
    return 1000 * total_time / repetitions

def main():
    FORMAT_STRING = '{:>10} {:>10} {:>10} {:>10}'
    print FORMAT_STRING.format('Samples', 'Slow', 'Fast', 'Ratio')
    for sample_size in (10, 100, 1000, 10000, 100000):
        slow_us = time_function_us(posterior_probability_treatment_is_better_slow, sample_size)
        fast_us = time_function_us(posterior_probability_treatment_is_better_fast, sample_size)
        print FORMAT_STRING.format(
            sample_size,
            '{:.3f}us'.format(slow_us),
            '{:.3f}us'.format(fast_us),
            '{:.1f}x'.format(slow_us / fast_us),
        )

if __name__ == '__main__':
    main()
