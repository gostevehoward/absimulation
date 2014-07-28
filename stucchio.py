from numpy import mgrid, zeros, where, maximum
from scipy.stats import beta

prior_params = [ (1, 1), (1,1) ]

def bayesian_expected_error(N,s, xgrid_size=1024):
    degrees_of_freedom = len(prior_params)
    posteriors = []
    for i in range(degrees_of_freedom):
        posteriors.append(beta(prior_params[i][0] + s[i] - 1, prior_params[i][1] + N[i] - s[i] - 1))
    x = mgrid[0:xgrid_size,0:xgrid_size] / float(xgrid_size)
    # Compute joint posterior, which is a product distribution
    pdf_arr = posteriors[0].pdf(x[1]) * posteriors[1].pdf(x[0])
    pdf_arr /= pdf_arr.sum() # normalization
    expected_error_dist = maximum(x[0]-x[1],0.0) * pdf_arr
    return expected_error_dist.sum()
