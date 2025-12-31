"""
transforms.py

Feature selection, sparsification, and probe utilities for linear probing.

- SparseTransform
    * mRMR-based feature selection with configurable redundancy/relevance.
    * Optionally normalizes data and returns feature scores.
- MeanDifferenceProbe
    * Simple mean-difference linear probe for binary labels.
    * Supports optional data normalization and direction normalization.
- MMProbe
    * PyTorch implementation of a Mahalanobis-style probe using covariance.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import functools
import math


def hellinger_fast(p, q):
    """
    Compute the Hellinger distance between two discrete distributions.

    The distance is defined as:
        H^2(p, q) = sum_i (sqrt(p_i) - sqrt(q_i))^2

    :param p: Iterable of probabilities for the first distribution
    :param q: Iterable of probabilities for the second distribution
    :return: Squared Hellinger distance as a float
    """
    return sum((math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q))


def parallel_df(func, df, series, n_jobs):
    """
    Apply a function to column chunks of a DataFrame in parallel.

    The function is called on sub-dataframes of df, and the results are
    concatenated back into a single Series or DataFrame.

    :param func: Function taking (X_chunk, y_series) and returning a Series/DataFrame
    :param df: Input DataFrame whose columns are to be processed
    :param series: Series or array used as the target variable
    :param n_jobs: Number of parallel jobs (use -1 for all CPUs)
    :return: Concatenated result of applying func to each column chunk
    """
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(df.columns))
    else:
        n_jobs = min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def chatterjee_cc(X, Y, ties=False, random_state=42):
    """
    Compute Chatterjee's concordance correlation coefficient.

    This is a rank-based measure of dependence between two variables.

    :param X: Array-like of shape (n_samples,) for the first variable
    :param Y: Array-like of shape (n_samples,) for the second variable
    :param ties: If True, apply tie-handling with randomization
    :param random_state: Seed for any randomness used in tie-handling
    :return: Chatterjee's concordance correlation coefficient
    """
    np.random.seed(random_state)
    n = len(X)
    order = np.argsort(X)
    ranks = np.argsort(Y[order])
    diff_ranks = np.abs(np.diff(ranks))

    if ties:
        counts = np.bincount(ranks)
        ranks += np.random.uniform(0, counts[ranks] - 1).astype(int)
        l = np.bincount(ranks).astype(float)
        return 1 - (n * np.sum(diff_ranks)) / (2 * np.sum(l * (n - l)))
    else:
        return 1 - (3 * np.sum(diff_ranks)) / (n ** 2 - 1)


def _mi_classif(X, y):
    """
    Compute mutual information between each column of X and y.

    This is a helper to be used with parallel_df and mutual_info_classif.

    :param X: DataFrame whose columns are feature candidates
    :param y: Array-like target labels
    :return: Series of mutual information scores per feature
    """
    def _mi_classif_series(x, y, n_neighbors=25):
        x_not_na = ~x.isna()
        if x_not_na.sum() == 0:
            return 0
        return mutual_info_classif(
            x[x_not_na].to_frame(),
            y[x_not_na],
            n_neighbors=n_neighbors,
        )[0]

    return X.apply(lambda col: _mi_classif_series(col, y)).fillna(0.0)


def mi_classif(X, y, n_jobs):
    """
    Compute mutual information between each column of X and y in parallel.

    :param X: DataFrame whose columns are feature candidates
    :param y: Array-like target labels
    :param n_jobs: Number of parallel jobs (use -1 for all CPUs)
    :return: Series of mutual information scores per feature
    """
    return parallel_df(_mi_classif, X, y, n_jobs=n_jobs)


def random_forest_classif(X, y):
    """
    Compute feature importances using a random forest classifier.

    Missing values are imputed with (global minimum - 1) before fitting.

    :param X: DataFrame of features
    :param y: Array-like target labels
    :return: Series of feature importances indexed by X.columns
    """
    forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        criterion="gini",
        random_state=0,
        n_jobs=-1,
    ).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs, corr_type="spearman"):
    """
    Compute correlation between a target column and multiple feature columns.

    The correlation type can be Spearman, Pearson, Chatterjee, or a placeholder
    for distance covariance (not implemented).

    :param target_column: Name of the target column in X
    :param features: List of feature column names in X
    :param X: DataFrame containing both target and feature columns
    :param n_jobs: Number of parallel jobs (use -1 for all CPUs)
    :param corr_type: One of {"spearman", "pearson", "chatterjee", "distance_covariance"}
    :return: Series of correlation values per feature
    """
    def _correlation(X_chunk, y):
        if corr_type == "spearman":
            return X_chunk.corrwith(y, method="spearman").fillna(0.0)
        elif corr_type == "pearson":
            return X_chunk.corrwith(y, method="pearson").fillna(0.0)
        elif corr_type == "chatterjee":
            return X_chunk.corrwith(y, method=chatterjee_cc).fillna(0.0)
        elif corr_type == "distance_covariance":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown correlation type: {corr_type}")

    return parallel_df(
        _correlation,
        X.loc[:, features],
        X.loc[:, target_column],
        n_jobs=n_jobs,
    )


def normalize(X):
    """
    Normalize rows of a vector or matrix to unit L2 norm.

    :param X: Numpy array of shape (d,) or (n_samples, d)
    :return: Row-normalized array with the same shape as X
    """
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def get_direction(acts, labels):
    """
    Compute a mean-difference direction between positive and negative classes.

    :param acts: Array of activations with shape (n_samples, d)
    :param labels: Binary labels array in {0, 1} with shape (n_samples,)
    :return: Mean-difference direction vector with shape (d,)
    """
    pos_acts = acts[labels == 1]
    neg_acts = acts[labels == 0]
    pos_mean = pos_acts.mean(0)
    neg_mean = neg_acts.mean(0)
    return pos_mean - neg_mean


def sparsify(x, num_features):
    """
    Zero out all but num_features largest-magnitude entries in a vector.

    :param x: Input 1D array
    :param num_features: Number of features to keep non-zero
    :return: Sparsified copy of x
    """
    _x = x.copy()
    x_abs = _x
    idx = np.argsort(x_abs)[:-num_features]
    _x[idx] = 0
    return _x


class SparseTransform(BaseEstimator, TransformerMixin):
    """
    Sparse feature selection transform based on mRMR and configurable criteria.

    This transformer selects up to max_k features using mRMR with specified
    relevance and redundancy functions, optionally normalizing the data.
    """

    def __init__(
        self,
        max_k=25,
        redundancy="spearman",
        relevance="mi",
        normalize=True,
        return_scores=False,
        show_progress=True,
    ):
        """
        Initialize a SparseTransform.

        :param max_k: Maximum number of features to select
        :param redundancy: Redundancy measure or custom function
                           (e.g., "spearman", "pearson", "chatterjee")
        :param relevance: Relevance measure or custom function
                          (e.g., "mi", "rf")
        :param normalize: If True, data will be normalized (not yet implemented)
        :param return_scores: If True, store feature scores from mRMR
        :param show_progress: If True, mrmr_classif may show progress
        """
        super().__init__()
        self.normalize = normalize

        self.max_k = max_k
        self.selected_features = None
        self.scores = None

        self.return_scores = return_scores
        self.show_progress = show_progress
        self.redundancy = redundancy
        self.relevance = relevance

        if self.normalize:
            self.sc = StandardScaler()

        # Set redundancy function
        if isinstance(self.redundancy, str):
            if self.redundancy in ["spearman", "pearson", "chatterjee", "distance_covariance"]:
                self.redundancy_fn = functools.partial(
                    correlation,
                    n_jobs=cpu_count(),
                    corr_type=self.redundancy,
                )
            else:
                raise ValueError(
                    f"Unknown correlation type: {self.redundancy}. "
                    "Choose from ['spearman', 'pearson', 'chatterjee', 'distance_covariance']"
                )
        else:
            self.redundancy_fn = self.redundancy

        # Set relevance function
        if isinstance(self.relevance, str):
            if self.relevance == "mi":
                self.relevance_fn = functools.partial(
                    mi_classif,
                    n_jobs=cpu_count(),
                )
            elif self.relevance in ["ks", "f"]:
                self.relevance_fn = self.relevance
            elif self.relevance == "rf":
                self.relevance_fn = random_forest_classif
            else:
                raise ValueError(
                    f"Unknown relevance type: {self.relevance}. "
                    "Choose from ['mi', 'ks', 'f', 'rf']"
                )
        else:
            self.relevance_fn = self.relevance

        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the SparseTransform by selecting a subset of features.

        mRMR is used to identify up to max_k features with high relevance
        and low redundancy.

        :param X: Array-like of shape (n_samples, n_features)
        :param y: Array-like of shape (n_samples,) with target labels
        :return: Self
        """
        if not self.is_fitted:
            if self.normalize:
                raise NotImplementedError()
                # X = self.sc.fit_transform(X)

            if self.return_scores:
                self.selected_features, self.scores = mrmr_classif(
                    pd.DataFrame(X),
                    pd.Series(y),
                    K=self.max_k,
                    redundancy=self.redundancy_fn,
                    relevance=self.relevance_fn,
                    return_scores=self.return_scores,
                    show_progress=self.show_progress,
                )
            else:
                self.selected_features = mrmr_classif(
                    pd.DataFrame(X),
                    pd.Series(y),
                    K=self.max_k,
                    redundancy=self.redundancy_fn,
                    relevance=self.relevance_fn,
                    show_progress=self.show_progress,
                    return_scores=False,
                )
            self.is_fitted = True

        return self

    def transform(self, X, k=None):
        """
        Transform X by zeroing out all but k selected features.

        :param X: Array-like of shape (n_samples, n_features)
        :param k: Number of features to keep (defaults to max_k)
        :return: Transformed array with the same shape as X
        """
        check_is_fitted(self, "is_fitted")
        if k is None:
            k = self.max_k
        else:
            assert k <= self.max_k, "k should be less than or equal to max_k"
        selected_features = self.selected_features[:k]

        if self.normalize:
            X = self.sc.transform(X)
        _X = np.zeros_like(X)
        _X[:, selected_features] = X[:, selected_features].copy()
        return _X

    def fit_transform(self, X, y, k=None):
        """
        Convenience method to fit and then transform X.

        :param X: Array-like of shape (n_samples, n_features)
        :param y: Array-like of shape (n_samples,) with target labels
        :param k: Number of features to keep (defaults to max_k)
        :return: Transformed array with the same shape as X
        """
        self.fit(X, y)
        return self.transform(X, k=k)