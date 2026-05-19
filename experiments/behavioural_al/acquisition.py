# acquisition functions for active learning

import numpy as np
from scipy.stats import norm


def random_acquisition(proba, std, **kwargs):
    """Baseline - uniform random sampling."""
    return np.random.rand(len(proba))


def margin_sampling(proba, std, **kwargs):
    """Uncertainty sampling: score is highest when P(positive) = 0.5."""
    return 1.0 - np.abs(proba - 0.5) * 2


def ucb_classification(proba, std, beta=1.0, **kwargs):
    """UCB: exploit high P(positive) while also exploring uncertain regions."""
    return proba + beta * std


STRATEGIES = {
    "random": random_acquisition,
    "margin": margin_sampling,
    "ucb": ucb_classification,
}


# regression acquisition functions (not used yet)

def ucb_regression(mean, std, beta=2.0):
    return mean + beta * std


def expected_improvement(mean, std, y_best=0.0):
    sigma = std + 1e-9
    z = (mean - y_best) / sigma
    return (mean - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
