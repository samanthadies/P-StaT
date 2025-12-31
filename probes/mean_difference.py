"""
mean_difference.py

Robust mean-difference linear classifier for binary probing.

- MeanDifferenceClassifier
    * Computes a linear direction as the difference of class means.
    * Optional Mahalanobis weighting via pooled covariance (Fisher/LDA-style).
    * Supports robust covariance estimation (OAS, Ledoit-Wolf, diagonal, etc.).
    * Returns both normalized and raw weight vectors.

Utility functions:
- normalize: Row-wise L2 normalization with numerical safeguards.
- robust_covariance: Safe covariance estimation with shrinkage and fallbacks.

Adapted from:
@article{marks2310geometry,
  title={The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of {T}rue/{F}alse Datasets},
  year={2024},
  author={Marks, Samuel and Tegmark, Max},
  doi={10.48550/arXiv.2310.06824},
  journal={arXiv preprint arXiv:2310.06824}
}

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import ledoit_wolf, oas
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve, LinAlgError

import logging

log = logging.getLogger(__name__)

VALID_COVARIANCE_METHODS = ["oas", "ledoit", "empirical", "shrunk", "diagonal"]


def normalize(X, tol=1e-9):
    """
    Normalize rows of a vector or matrix to unit L2 norm.

    :param X: Numpy array of shape (d,) or (n_samples, d)
    :param tol: Small constant added to norms for numerical stability
    :return: Row-normalized array with the same shape as X
    """
    if X.ndim == 1:
        return X / (np.linalg.norm(X) + tol)
    return X / (np.linalg.norm(X, axis=1)[:, np.newaxis] + tol)


def robust_covariance(X, method="oas", fallback_scale=1.0):
    """
    Compute a robust covariance estimate with shrinkage and safe fallbacks.

    Supports several shrinkage and diagonal approximations, and falls back
    to a scaled identity matrix if estimation fails or produces non-finite
    values.

    :param X: Data matrix of shape (n_samples, n_features)
    :param method: One of {"oas", "ledoit", "empirical", "shrunk", "diagonal"}
    :param fallback_scale: Scalar used for the identity fallback
    :return: Covariance matrix of shape (n_features, n_features)
    """
    assert method in VALID_COVARIANCE_METHODS, (
        f"Unknown method: {method}, must be one of {VALID_COVARIANCE_METHODS}"
    )
    X = np.asarray(X)
    _, d = X.shape

    try:
        if method == "oas":
            cov, _ = oas(X, assume_centered=True)
        elif method == "ledoit":
            cov, _ = ledoit_wolf(X, assume_centered=True)
        elif method == "empirical":
            cov = np.cov(X, rowvar=False)
        elif method == "diagonal":
            cov = np.diag(np.var(X, axis=0))
        elif method == "shrunk":
            emp_cov = np.cov(X, rowvar=False, bias=False)
            trace = np.trace(emp_cov)
            shrinkage = 0.1
            cov = (1 - shrinkage) * emp_cov + shrinkage * (trace / X.shape[1]) * np.eye(
                X.shape[1]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        if not np.isfinite(X).all():
            log.warning("Non-finite data detected; using scaled identity.")
            return fallback_scale * np.eye(X.shape[1])
        if np.isnan(cov).any():
            raise ValueError("Non-finite entries detected in covariance.")
    except (LinAlgError, ValueError, np.linalg.LinAlgError) as e:
        log.warning(
            f"Covariance estimation failed ({e}); using scaled identity."
        )
        cov = fallback_scale * np.eye(d)

    cov = 0.5 * (cov + cov.T)
    return cov


class MeanDifferenceClassifier(ClassifierMixin, BaseEstimator):
    """
    Binary mean-difference classifier with optional covariance weighting.

    The classifier computes a linear direction as the difference between
    class means and, if requested, applies pooled-covariance weighting
    (Mahalanobis/Fisher-style). The implementation is adapted from
    https://github.com/saprmarks/geometry-of-truth/.
    """

    def __init__(
        self,
        fit_intercept=True,
        with_covariance=False,
        cov_type="oas",
        cov_reg=1e-8,
        tol=1e-8,
        verbose=False,
    ):
        """
        Initialize a mean-difference classifier.

        :param fit_intercept: If True, place the decision boundary at the
                              midpoint between projected class means
        :param with_covariance: If True, use inverse pooled covariance to
                                weight the direction (Fisher/LDA-style)
        :param cov_type: Covariance estimator type in VALID_COVARIANCE_METHODS
        :param cov_reg: Ridge term added to the pooled covariance
        :param tol: Tolerance used when normalizing the weight vector
        :param verbose: If True, log additional information during scoring
        """
        super().__init__()
        assert cov_type in VALID_COVARIANCE_METHODS, (
            f"cov_type must be one of {VALID_COVARIANCE_METHODS}"
        )
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.with_covariance = with_covariance
        self.cov_reg = cov_reg
        self.cov_type = cov_type
        self.tol = tol
        self.M_ = None

        self.intercept_ = None
        self.coef_ = None
        self.coef_raw_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, M=None):
        """
        Fit the model on binary data.

        If with_covariance=True, use either a supplied Mahalanobis matrix M
        or a pooled covariance estimate to whiten the mean-difference
        direction before normalization.

        :param X: Array-like of shape (n_samples, n_features)
        :param y: Binary labels in {0, 1} with shape (n_samples,)
        :param M: Optional Mahalanobis matrix if with_covariance=True
        :return: Self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert type_of_target(y) == "binary", "Labels should be binary."
        if M is not None:
            assert self.with_covariance, "If providing M, must have with_covariance=True"
            assert M.shape[0] == M.shape[1] == X.shape[1], (
                "M must be square and match feature dimension."
            )

        pos_acts = X[y == 1]
        neg_acts = X[y == 0]
        mu_pos = pos_acts.mean(0)
        mu_neg = neg_acts.mean(0)
        delta = mu_pos - mu_neg

        if self.with_covariance:
            if M is not None:
                w = M @ delta
            else:
                S_pos = robust_covariance(
                    pos_acts - mu_pos[None, :],
                    method=self.cov_type,
                    fallback_scale=1.0,
                )
                S_neg = robust_covariance(
                    neg_acts - mu_neg[None, :],
                    method=self.cov_type,
                    fallback_scale=1.0,
                )
                n_pos = pos_acts.shape[0]
                n_neg = neg_acts.shape[0]
                Sp = ((n_pos - 1) * S_pos + (n_neg - 1) * S_neg) / max(
                    1, (n_pos + n_neg - 2)
                )
                Sp = Sp + self.cov_reg * np.eye(Sp.shape[0], dtype=Sp.dtype)
                c, lower = cho_factor(
                    Sp,
                    overwrite_a=False,
                    check_finite=False,
                )
                w = cho_solve((c, lower), delta, check_finite=False)
        else:
            w = delta

        self.coef_raw_ = np.asarray(w, dtype=float).copy()

        w = normalize(w, tol=self.tol)
        self.coef_ = w.reshape(1, -1)

        if self.fit_intercept:
            b_pos = (pos_acts @ self.coef_.T).mean()
            b_neg = (neg_acts @ self.coef_.T).mean()
            self.intercept_ = float(-0.5 * (b_pos + b_neg))
        else:
            self.intercept_ = 0.0

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict binary labels for each sample in X.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of predicted labels in {0, 1}
        """
        return self.predict_proba(X).round()

    def predict_proba(self, X):
        """
        Predict probabilities for label 1 using a logistic link.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of probabilities in [0, 1]
        """
        return expit(self.decision_function(X))

    def decision_function(self, X):
        """
        Compute decision scores for each sample in X.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of real-valued scores with shape (n_samples,)
        """
        check_is_fitted(self)
        X = np.asarray(X)
        return (X @ self.coef_.T).ravel() + self.intercept_

    def score(self, X, y, scorer, sample_weight=None):
        """
        Compute a user-specified score for the model.

        The scorer is typically a callable such as average_precision_score
        or matthews_corrcoef.

        :param X: Array-like of shape (n_samples, n_features)
        :param y: Binary labels in {0, 1} with shape (n_samples,)
        :param scorer: Callable taking (y_true, y_pred_or_proba, sample_weight=None)
        :param sample_weight: Optional sample weights
        :return: Scalar score value
        """
        assert type_of_target(y) == "binary", "Labels should be binary."
        try:
            if sample_weight is not None:
                return scorer(y, self.predict_proba(X), sample_weight=sample_weight)
            return scorer(y, self.predict_proba(X))
        except Exception:
            if self.verbose:
                log.warning("Scorer failed on probabilities; using discrete labels.")
            if sample_weight is not None:
                return scorer(y, self.predict(X), sample_weight=sample_weight)
            return scorer(y, self.predict(X))


__all__ = ["MeanDifferenceClassifier"]
