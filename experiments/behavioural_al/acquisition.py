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


def core_set(proba, std, X_pool=None, X_labeled=None, **kwargs):
    """Core-set: select genes most distant from the already-labeled set.
    Maximises coverage of feature space — finds surprising genes in unexplored regions.
    Score = minimum euclidean distance to any labeled gene (higher = more distant = priority).
    """
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X_pool, X_labeled)
    return dists.min(axis=1)


def query_by_committee(proba, std, committee_probas=None, **kwargs):
    """Query-by-committee: query genes where an ensemble of models trained on
    bootstrap samples of the labeled set disagrees most.
    Disagreement = std of predicted probabilities across committee members.
    Falls back to model uncertainty if committee predictions are unavailable.
    """
    if committee_probas is None or len(committee_probas) < 2:
        return std
    return np.array(committee_probas).std(axis=0)


def oracle(proba, std, y_pool=None, **kwargs):
    """Oracle upper bound: queries true positives first using ground-truth labels.
    Unrealisable in practice — sets the ceiling for any acquisition strategy.
    Ties among negatives are broken randomly so the curve is smooth across trials.
    """
    scores = y_pool.astype(float) if y_pool is not None else np.zeros(len(proba))
    scores = scores + np.random.rand(len(scores)) * 1e-6
    return scores


STRATEGIES = {
    "random":    random_acquisition,
    "margin":    margin_sampling,
    "ucb":       ucb_classification,
    "core_set":  core_set,
    "qbc":       query_by_committee,
    "oracle":    oracle,
}


# regression acquisition functions (not used yet)

def ucb_regression(mean, std, beta=2.0):
    return mean + beta * std


def expected_improvement(mean, std, y_best=0.0):
    sigma = std + 1e-9
    z = (mean - y_best) / sigma
    return (mean - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
