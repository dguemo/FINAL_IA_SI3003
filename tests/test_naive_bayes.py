import numpy as np
import pytest

from models.baselines import MajorityBaseline, PersistenceBaseline
from models.naive_bayes import GaussianNaiveBayes, logsumexp


def gauss_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def test_parameters_are_mle():
    X = np.array([[1.0, 10], [3.0, 14], [10.0, 0], [12.0, 2], [11.0, 1]])
    y = np.array([0, 0, 1, 1, 1])
    m = GaussianNaiveBayes(n_classes=2, var_smoothing=0.0, alpha=0.0).fit(X, y)
    np.testing.assert_allclose(m.theta_[0], [2.0, 12.0])
    np.testing.assert_allclose(m.var_[1], X[y == 1].var(axis=0))
    np.testing.assert_allclose(m.class_prior_, [0.4, 0.6])


def test_posterior_matches_bayes_rule_by_hand():
    X = np.array([[0.0], [2.0], [4.0], [6.0]])
    y = np.array([0, 0, 1, 1])
    m = GaussianNaiveBayes(n_classes=2, var_smoothing=0.0, alpha=0.0).fit(X, y)
    x = 2.5
    l0 = 0.5 * gauss_pdf(x, 1.0, 1.0)
    l1 = 0.5 * gauss_pdf(x, 5.0, 1.0)
    expected = np.array([l0, l1]) / (l0 + l1)
    np.testing.assert_allclose(m.predict_proba([[x]])[0], expected, rtol=1e-10)


def test_probabilities_sum_to_one_and_predict_is_argmax():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(k * 3, 1, (50, 4)) for k in range(3)])
    y = np.repeat([0, 1, 2], 50)
    m = GaussianNaiveBayes().fit(X, y)
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(m.predict(X), P.argmax(axis=1))
    assert (m.predict(X) == y).mean() > 0.95


def test_numerically_stable_far_from_means():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array([0, 0, 1, 1])
    m = GaussianNaiveBayes(n_classes=2).fit(X, y)
    p = m.predict_proba([[1e4]])
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0)


def test_logsumexp():
    a = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
    np.testing.assert_allclose(logsumexp(a), [1000 + np.log(2), -1000 + np.log(2)])


def test_missing_class_and_priors_modes():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])
    m = GaussianNaiveBayes(n_classes=3).fit(X, y)
    assert m.class_count_[2] == 0 and m.class_prior_[2] > 0
    u = GaussianNaiveBayes(n_classes=3, priors="uniform").fit(X, y)
    np.testing.assert_allclose(u.class_prior_, 1 / 3)
    with pytest.raises(ValueError):
        GaussianNaiveBayes(n_classes=3, priors=np.array([0.5, 0.5, 0.5])).fit(X, y)


def test_prior_power_flattens_priors():
    X = np.arange(10.0)[:, None]
    y = np.array([0] * 8 + [1] * 2)
    p1 = GaussianNaiveBayes(n_classes=2, alpha=0).fit(X, y).class_prior_
    p0 = GaussianNaiveBayes(n_classes=2, alpha=0, prior_power=0.0).fit(X, y).class_prior_
    ph = GaussianNaiveBayes(n_classes=2, alpha=0, prior_power=0.5).fit(X, y).class_prior_
    np.testing.assert_allclose(p1, [0.8, 0.2])
    np.testing.assert_allclose(p0, [0.5, 0.5])
    np.testing.assert_allclose(ph, [2 / 3, 1 / 3])


def test_rejects_nan_and_bad_labels():
    with pytest.raises(ValueError):
        GaussianNaiveBayes().fit(np.array([[np.nan]]), np.array([0]))
    with pytest.raises(ValueError):
        GaussianNaiveBayes(n_classes=2).fit(np.array([[1.0]]), np.array([5]))


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(60, 3)), rng.integers(0, 3, 60)
    m = GaussianNaiveBayes().fit(X, y, feature_names=["a", "b", "c"])
    m.save(tmp_path / "m.npz")
    m2 = GaussianNaiveBayes.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X))
    assert m2.feature_names == ["a", "b", "c"]


def test_baselines():
    assert list(PersistenceBaseline().predict([0, 2, 1])) == [0, 2, 1]
    mb = MajorityBaseline().fit(None, np.array([1, 1, 0, 2, 1]))
    assert list(mb.predict([0, 0])) == [1, 1]
