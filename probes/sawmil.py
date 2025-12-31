"""
sawmil.py

Patches for sbMIL / sMIL / SVM classes in misvm.

- Adds L1-style penalty into the SVM QP setup.
- Overrides sbMIL.fit to use patched BagSplitter and two-stage training.
- Adds linearize() utilities to extract a linear direction and bias
  from kernel SVM and MIL models.
- Exposes sort_and_label used to select top positive instances.

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

import logging

import numpy as np

from probes.mil.util import BagSplitter_patched
from misvm.sil import SIL
from misvm.smil import sMIL
from misvm.sbmil import sbMIL
from misvm.svm import SVM, _smart_kernel
from misvm.kernel import by_name as kernel_by_name
from misvm.util import spdiag
from misvm.util import BagSplitter

log = logging.getLogger("sbSVM_patch")

# Use the patched BagSplitter implementation
BagSplitter = BagSplitter_patched


def __init__(self, kernel="linear", C=1.0, p=3, gamma=1.0, scale_C=True,
             verbose=True, sv_cutoff=1e-7, penalty=-0.1):
    """
    Initialize an SVM-like classifier with optional L1 penalty.

    This mirrors the misvm.SVM constructor, with an added penalty term
    that can be used to incorporate L1-like regularization via the QP.

    :param kernel: Kernel name ("linear", "quadratic", "polynomial", "rbf")
    :param C: Loss/regularization tradeoff constant
    :param p: Polynomial degree for the "polynomial" kernel
    :param gamma: RBF scale parameter for the "rbf" kernel
    :param scale_C: If True, scale C by the number of examples
    :param verbose: If True, log optimization progress
    :param sv_cutoff: Threshold for support vector identification
    :param penalty: Nonnegative penalty coefficient added to the QP diagonal
    """
    self.kernel = kernel
    self.C = C
    self.gamma = gamma
    self.p = p
    self.scale_C = scale_C
    self.verbose = verbose
    self.sv_cutoff = sv_cutoff
    self.penalty = penalty

    self._X = None
    self._y = None
    self._objective = None
    self._alphas = None
    self._sv = None
    self._sv_alphas = None
    self._sv_X = None
    self._sv_y = None
    self._b = None
    self._predictions = None


def _setup_svm(self, examples, classes, C):
    """
    Construct kernel matrices and QP ingredients for SVM training.

    This mirrors misvm.SVM._setup_svm but optionally adds an L1-style
    penalty term to the Hessian via self.penalty.

    :param examples: Sequence or array of examples
    :param classes: Column vector of labels in {-1, +1}
    :param C: Scalar or per-example array of upper bounds on alphas
    :return: Tuple (K, H, f, A, b, lb, ub) for QP solver
    """
    kernel = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)
    n = len(examples)
    e = np.matrix(np.ones((n, 1)))

    if kernel is None:
        K = None
        H = None
    else:
        K = _smart_kernel(kernel, examples)
        D = spdiag(classes)
        H = D * K * D

    if self.penalty > 0:
        log.warning("Adding L1-style penalty to Hessian: %s", self.penalty)
        H += self.penalty * np.eye(n)

    f = -e

    A = classes.T.astype(float)
    b = np.matrix([0.0])

    lb = np.matrix(np.zeros((n, 1)))
    if isinstance(C, float):
        ub = C * e
    else:
        ub = C
    return K, H, f, A, b, lb, ub


def fit(self, bags, y, in_bag_labels=None):
    """
    Fit sbMIL with a two-stage procedure and patched BagSplitter.

    Stage 1:
        Train an sMIL classifier to score positive instances.
    Stage 2:
        Select top-scoring positive instances (with intrabag labels),
        then train an SIL classifier on all instances.

    :param bags: Sequence of bags, each an m-by-k array of instances
    :param y: Array-like of bag labels in {-1, +1}
    :param in_bag_labels: Array-like of instance labels within positive bags
    :return: Self
    """
    self._bags = [np.asmatrix(bag) for bag in bags]
    y = np.asmatrix(y).reshape((-1, 1))

    if in_bag_labels is not None:
        self.bs = BagSplitter(self._bags, y, in_bag_labels)
    else:
        raise NotImplementedError("sbMIL.fit requires in_bag_labels")

    bs = self.bs

    if self.verbose:
        log.warning("Training initial sMIL classifier for sbMIL...")
    init_classifier = sMIL(
        kernel=self.kernel,
        C=self.C,
        p=self.p,
        gamma=self.gamma,
        scale_C=self.scale_C,
        verbose=self.verbose,
        sv_cutoff=self.sv_cutoff,
        penalty=self.penalty,
    )
    init_classifier.fit(bags, y)

    if self.verbose:
        log.warning("Training SIL classifier for sbMIL...")
    f_pos = init_classifier.predict(bs.pos_inst_as_bags)

    pos_labels, f_cutoff, _ = sort_and_label(self, bs, f_pos)

    if (pos_labels == 1).sum() < 0.05 * bs.L_p:
        log.warning(
            "Less than 5%% of positives chosen (%s); ignoring intrabag labels.",
            (pos_labels == 1).sum(),
        )
        pos_labels = -np.matrix(np.ones((bs.L_p, 1)))
        pos_labels[np.nonzero(f_pos >= f_cutoff)] = 1.0

    if self.verbose:
        log.warning(
            "Retraining with top %d%% as positive...",
            int(100 * self.eta),
        )

    labels = np.vstack([-np.ones((bs.L_n, 1)), pos_labels])
    self._labels = labels

    if self.verbose:
        log.warning(
            "Number of positive instances: %s out of %s",
            np.sum(pos_labels == 1),
            labels.shape[0],
        )

    super(SIL, self).fit(bs.instances, labels)
    return self


def sort_and_label(self, bs, f_pos):
    """
    Rank positive instances and assign intra-bag labels.

    Instances are sorted by their sMIL scores, and a cutoff is selected
    so that approximately self.eta of positive instances are labeled
    as positive, subject to minimum constraints based on bs.X_p.

    :param self: sbMIL instance with attribute eta
    :param bs: BagSplitter instance with positive-instance metadata
    :param f_pos: Scores for positive instances from the initial sMIL
    :return: Tuple (labels, f_cutoff, mask) where
             labels is a column vector of instance labels in {-1, +1},
             f_cutoff is the score threshold,
             mask is the intrabag positive-label mask
    """
    n = int(round(bs.L_p * self.eta))
    n = min(bs.L_p, n)
    n = max(bs.X_p, n)

    f_cutoff = sorted((float(f) for f in f_pos), reverse=True)[n - 1]

    labels = -np.matrix(np.ones((bs.L_p, 1)))
    mask = bs.instance_intrabag_labels_pos
    labels[(f_pos >= f_cutoff) & (mask == 1)] = 1.0
    return labels, f_cutoff, mask


def linearize(self, normalize=True):
    """
    Extract a linear direction and bias from a trained SVM-like model.

    The direction is computed as a weighted sum over support vectors,
    and the bias is estimated by averaging the margin residuals over
    non-bound support vectors.

    :param self: Trained SVM/sMIL/sbMIL model with support-vector fields
    :param normalize: If True, normalize the direction to unit norm
    :return: Tuple (w, bias) where w is a direction vector and bias a scalar
    """
    X = self._sv_X
    alphas = self._sv_alphas[:, 0]
    y = self._sv_y[:, 0]

    coefs = np.einsum("bs, bs -> b", alphas, y)
    w = np.einsum("b, bh -> bh", coefs, X).sum(0)
    if normalize:
        w /= np.linalg.norm(w)

    non_bound_mask = (alphas > self.sv_cutoff).flatten() & (
        alphas < self.C - self.sv_cutoff
    ).flatten()

    mask_pos = non_bound_mask & (y > 0).flatten()
    mask_neg = non_bound_mask & (y < 0).flatten()

    bias_pos = np.mean(
        y.flatten()[mask_pos] - np.dot(X, w).flatten()[mask_pos]
    )
    bias_neg = np.mean(
        y.flatten()[mask_neg] - np.dot(X, w).flatten()[mask_neg]
    )
    bias = (bias_pos + bias_neg) / 2.0
    return w, bias


# Patch misvm classes with the new behavior
SVM.__init__ = __init__
SVM._setup_svm = _setup_svm
sbMIL.fit = fit
sbMIL.linearize = linearize
sMIL.linearize = linearize
SVM.linearize = linearize
sbMIL.sort_and_label = sort_and_label