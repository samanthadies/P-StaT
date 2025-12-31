"""
conformal.py

Script to make classifier compatible with conformal prediction.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import numpy as np
from sklearn.base import BaseEstimator


def reciprocal_nonconformity(y, f, c=0):
    """
    Compute reciprocal/inverse-based nonconformity scores.

    For each sample, if the candidate label matches the predicted label
    induced by the score and threshold c, the score is small; otherwise
    the score is large.

    :param y: Array-like of candidate or true labels (e.g., 0 or 1)
    :param f: Array-like of decision scores (distance to the hyperplane)
    :param c: Threshold for deciding label from score (default 0)
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    scores = []
    for yi, fi in zip(y, f):
        # Predicted label is 1 if fi > c, else 0
        pred = int(fi > c)
        if int(yi) == int(pred):
            # Lower nonconformity when prediction matches
            score = 1 / (1 + abs(fi))
        else:
            # Higher nonconformity when prediction mismatches
            score = 1 + abs(fi)
        scores.append(score)
    return np.array(scores)


def symmetric_nonconformity_with_threshold(y, f, threshold=0.0):
    """
    Compute symmetric nonconformity scores with an adjustable threshold.

    The labels are mapped to {-1, +1} and combined with the shifted scores.
    Larger values indicate more nonconformity.

    :param y: Array-like of candidate or true labels in {0, 1}
    :param f: Array-like of decision scores
    :param threshold: Scalar threshold to shift the decision function
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    y_sym = 2 * np.asarray(y) - 1
    f = np.asarray(f)
    return np.exp(-y_sym * (f - threshold))


def symmetric_nonconformity(y, f):
    """
    Compute symmetric nonconformity scores for binary classification.

    The labels are mapped to {-1, +1} and combined with the raw scores.
    Larger values indicate more nonconformity.

    :param y: Array-like of candidate or true labels in {0, 1}
    :param f: Array-like of decision scores
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    y_sym = 2 * np.asarray(y) - 1
    f = np.asarray(f)
    return np.exp(-y_sym * f)


def inverse_probability_nc(y, probs):
    """
    Compute nonconformity scores from predicted class probabilities.

    For each sample, the nonconformity score is defined as 1 minus the
    predicted probability of the candidate label.

    :param y: Array-like of candidate or true labels (0, 1, 2, ...)
    :param probs: Array-like of predicted probabilities with shape
                  (n_samples, n_classes)
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    y = np.asarray(y)
    probs = np.asarray(probs)

    candidate_probs = probs[np.arange(len(y)), y]
    scores = 1 - candidate_probs
    return scores


def create_symmetric_nc_with_threshold(threshold):
    """
    Create a symmetric nonconformity function with a fixed threshold.

    The returned function has the signature f(y, scores) and internally
    calls symmetric_nonconformity_with_threshold with the given threshold.

    :param threshold: Scalar threshold to shift the decision function
    :return: Nonconformity function taking (y, scores)
    """
    def nc(a, b):
        return symmetric_nonconformity_with_threshold(a, b, threshold=threshold)
    return nc


def margin_nonconformity(y, scores):
    """
    Compute margin-based nonconformity scores.

    For each sample, if the candidate label matches the sign-based prediction
    from scores (score > 0 -> label 1, else 0), the score is small; otherwise
    the score is large.

    :param y: Array-like of candidate or true labels in {0, 1}
    :param scores: Array-like of decision scores
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    preds = scores > 0
    output = []
    for i in range(len(y)):
        if y[i] == preds[i]:
            output.append(1 / (1 + abs(scores[i])))
        else:
            output.append(1 + abs(scores[i]))
    return np.array(output)


def cumulative_softmax_nc(y, probs):
    """
    Compute nonconformity scores from cumulative probability mass.

    For each sample, the score is the sum of probabilities of all classes
    whose probability exceeds that of the candidate label. Smaller scores
    indicate stronger support for the candidate.

    :param y: Array-like of candidate or true labels (0, 1, 2, ...)
    :param probs: Array-like of softmax probabilities with shape
                  (n_samples, n_classes)
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    y = np.asarray(y)
    probs = np.asarray(probs)
    scores = np.zeros_like(y, dtype=float)
    for i in range(y.shape[0]):
        candidate = int(y[i])
        p_candidate = np.array(probs[i, candidate])
        scores[i] = np.sum(probs[i, :][probs[i, :] > p_candidate])
    return scores


def probability_margin_nc(y, probs):
    """
    Compute probability-margin nonconformity scores.

    For each sample, compute the difference between the candidate label
    probability and the largest other-class probability, then map the
    result to [0, 1] as (1 - (py - pz)) / 2.

    :param y: Array-like of candidate or true labels (0, 1, 2, ...)
    :param probs: Array-like of probabilities with shape (n_samples, n_classes)
    :return: Array of nonconformity scores with shape (n_samples,)
    """
    y = np.asarray(y)
    probs = np.atleast_1d(probs)
    scores = np.zeros_like(y, dtype=float)
    for i in range(y.shape[0]):
        candidate = int(y[i])
        py = probs[i, candidate]
        other_probs = np.delete(probs[i], candidate)
        pz = np.max(other_probs)
        scores[i] = (1.0 - (py - pz)) / 2
    return scores


class InductiveConformalPredictor(BaseEstimator):
    """
    Inductive conformal predictor for binary classification.

    This class wraps a nonconformity function and uses calibration scores
    to construct conformal prediction sets and derived predictions.
    """

    def __init__(self, nonconformity_func=margin_nonconformity, alpha=0.1, tie_breaking=True, **nc_kwargs):
        """
        Initialize a binary inductive conformal predictor.

        :param nonconformity_func: Function taking (y, scores, **kwargs) and
                                   returning nonconformity scores
        :param alpha: Significance level (e.g., 0.05 for 95 percent confidence)
        :param tie_breaking: If True, break ties using p-values; otherwise abstain
        :param nc_kwargs: Additional keyword arguments passed to the nonconformity function
        """
        self.alpha = alpha
        self.nc_kwargs = nc_kwargs
        self.nc_func = nonconformity_func
        self.calibration_scores = None
        self.tie_breaking = tie_breaking
        self._is_fitted = False
        super().__init__()

    def fit(self, y, scores):
        """
        Fit the conformal predictor on calibration data.

        This computes and stores the calibration nonconformity scores
        used to form p-values for new samples.

        :param y: Array-like of true labels for the calibration set
        :param scores: Array-like of decision scores for the calibration set
        :return: Self
        """
        self._is_fitted = True
        self.calibration_scores = self.nc_func(y, scores, **self.nc_kwargs)
        return self

    def _predict_set(self, scores):
        """
        Compute conformal prediction sets for new samples.

        For each new score, this evaluates both candidate labels (0 and 1),
        computes their nonconformity scores and p-values, and returns the
        set of labels whose p-values exceed alpha.

        :param scores: Array-like of decision scores for new samples
        :return: Tuple (conformal_sets, p_values) where
                 conformal_sets is a list of label sets per sample,
                 p_values is a list of [p0, p1] per sample
        """
        assert self.calibration_scores is not None, "Fit the model first."
        conformal_sets = []
        p_vals = []
        scores = np.atleast_1d(scores)
        for i in range(len(scores)):
            candidate_set = []
            p_sets = []
            for candidate in [0, 1]:
                candidate_score = self.nc_func(
                    np.array([candidate]), np.array([scores[i]]), **self.nc_kwargs
                )[0]
                p_val = (np.sum(self.calibration_scores >= candidate_score) + 1) / (
                    len(self.calibration_scores) + 1
                )
                if p_val > self.alpha:
                    candidate_set.append(candidate)
                p_sets.append(p_val)
            conformal_sets.append(candidate_set)
            p_vals.append(p_sets)
        return conformal_sets, p_vals

    def evaluate(self, scores):
        """
        Evaluate conformal prediction sets and derive point predictions.

        Predictions are formed as:
        - -1 if the conformal set is empty (abstention)
        - single label if the set is a singleton
        - tie-broken label if the set has both labels and tie_breaking is True
        - -1 if the set has both labels and tie_breaking is False

        :param scores: Array-like of decision scores for new samples
        :return: Dictionary with keys:
                 'predictions' (array of labels or -1),
                 'conformal_sets' (list of label sets),
                 'pvalues' (array of [p0, p1] per sample)
        """
        conformal_sets, p_vals = self._predict_set(scores)
        preds = []
        for i in range(len(conformal_sets)):
            if len(conformal_sets[i]) == 0:
                preds.append(-1)
            elif len(conformal_sets[i]) > 1:
                if self.tie_breaking:
                    if p_vals[i][0] > p_vals[i][1]:
                        preds.append(0)
                    else:
                        preds.append(1)
                else:
                    preds.append(-1)
            else:
                preds.append(conformal_sets[i][0])
        return {
            "predictions": np.array(preds),
            "conformal_sets": conformal_sets,
            "pvalues": np.array(p_vals),
        }

    def acceptance_rate(self, scores):
        """
        Compute the acceptance rate on new samples.

        The acceptance rate is the fraction of samples for which the
        conformal predictor does not abstain.

        :param scores: Array-like of decision scores for new samples
        :return: Acceptance rate in [0, 1]
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval_dict = self.evaluate(scores)
        abstained = eval_dict["predictions"] == -1
        return 1 - np.mean(abstained)

    def coverage(self, scores, y):
        """
        Compute empirical coverage on new samples.

        The coverage is the fraction of true labels that belong to
        their corresponding conformal prediction set.

        :param scores: Array-like of decision scores for new samples
        :param y: Array-like of true labels for new samples
        :return: Coverage in [0, 1]
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval_dict = self.evaluate(scores)
        res = [_y in _set for _y, _set in zip(y, eval_dict["conformal_sets"])]
        return np.mean(res)

    def mask(self, scores):
        """
        Compute a boolean mask of accepted samples.

        Entries are True where the predictor does not abstain and
        False where it returns -1.

        :param scores: Array-like of decision scores for new samples
        :return: Boolean array indicating accepted samples
        """
        assert self.calibration_scores is not None, "Fit the model first."
        eval_dict = self.evaluate(scores)
        return eval_dict["predictions"] != -1

    def predict_set(self, scores):
        """
        Return conformal prediction sets for new samples.

        :param scores: Array-like of decision scores for new samples
        :return: List of label sets per sample
        """
        return self.evaluate(scores)["conformal_sets"]

    def predict(self, scores):
        """
        Return point predictions for new samples.

        :param scores: Array-like of decision scores for new samples
        :return: Array of labels or -1 for abstentions
        """
        return self.evaluate(scores)["predictions"]

    def get_params(self, deep=False):
        """
        Get hyperparameters of the conformal predictor.

        :param deep: Ignored, kept for sklearn API compatibility
        :return: Dictionary of parameter names and values
        """
        params = {
            "alpha": self.alpha,
            "tie_breaking": self.tie_breaking,
            "nc_kwargs": self.nc_kwargs,
            "nonconformity_func": self.nc_func.__name__,
        }
        return params


class MulticlassICP(InductiveConformalPredictor):
    """
    Inductive conformal predictor for multiclass classification.

    This extends the binary ICP to support arbitrary numbers of classes,
    using a multiclass nonconformity function such as probability_margin_nc.
    """

    def __init__(self, nonconformity_func=probability_margin_nc, alpha=0.1, n_classes=3, tie_breaking=True, **nc_kwargs):
        """
        Initialize a multiclass inductive conformal predictor.

        :param nonconformity_func: Function taking (y, scores, **kwargs) and
                                   returning nonconformity scores
        :param alpha: Significance level (e.g., 0.05 for 95 percent confidence)
        :param n_classes: Number of possible class labels
        :param tie_breaking: If True, break ties using p-values; otherwise abstain
        :param nc_kwargs: Additional keyword arguments for the nonconformity function
        """
        super().__init__(nonconformity_func, alpha, tie_breaking, **nc_kwargs)
        self.n_classes = n_classes

    def _predict_set(self, scores):
        """
        Compute conformal prediction sets for multiclass scores.

        For each sample and each candidate class, compute the nonconformity
        score and p-value, then form the set of labels whose p-values exceed
        alpha.

        :param scores: Array-like of scores or probability vectors for new samples
        :return: Tuple (conformal_sets, p_values) where
                 conformal_sets is a list of label sets per sample,
                 p_values is a list of length-n_classes p-value lists per sample
        """
        assert self.calibration_scores is not None, "Fit the model first."
        conformal_sets = []
        p_vals = []
        scores = np.atleast_1d(scores)
        for i in range(len(scores)):
            candidate_set = []
            p_set = [None] * self.n_classes
            for candidate in range(self.n_classes):
                candidate_score = self.nc_func(
                    np.array([candidate]), np.array([scores[i]]), **self.nc_kwargs
                )[0]
                p_val = (np.sum(self.calibration_scores >= candidate_score) + 1) / (
                    len(self.calibration_scores) + 1
                )
                p_set[candidate] = p_val
                if p_val > self.alpha:
                    candidate_set.append(candidate)
            conformal_sets.append(candidate_set)
            p_vals.append(p_set)
        return conformal_sets, p_vals
