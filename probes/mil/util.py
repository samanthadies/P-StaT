"""
mil_util.py

Patched BagSplitter utility for multi-instance learning (MIL).

Adapted from:
@inproceedings{trilemma2025preprint,
  title={The Trilemma of Truth in Large Language Models},
  author={Savcisens, Germans and Eliassi‐Rad, Tina},
  booktitle={arXiv preprint arXiv:2506.23921},
  year={2025}
}

"""

from copy import deepcopy
import numpy as np


class BagSplitter_patched:
    """
    Patched bag splitter for MIL with intra-bag labels.

    This class mirrors the behavior of misvm.util.BagSplitter while
    additionally tracking per-instance labels inside positive bags.

    :param bags: Sequence of bags, where each bag is an array-like of instances
    :param classes: Array-like of bag labels (e.g., -1 or +1)
    :param bag_labels: Sequence of arrays with intra-bag labels for each bag
    """

    def __init__(self, bags, classes, bag_labels=None):
        self._bags = deepcopy(bags)
        self._classes = deepcopy(classes)
        self._bag_labels = deepcopy(bag_labels)

    @property
    def bags(self):
        """
        Return the list of all bags (deep-copied at initialization).

        :return: List of bag arrays
        """
        return self._bags

    @property
    def classes(self):
        """
        Return the array of bag-level classes.

        :return: Array-like of bag labels (e.g., -1 or +1)
        """
        return self._classes

    @property
    def bag_labels(self):
        """
        Return the list of intra-bag label arrays.

        :return: List of arrays, one per bag, with instance-level labels
        """
        return self._bag_labels

    @property
    def pos_bags_with_labels(self):
        """
        Return positive bags paired with their intra-bag labels.

        :return: List of (bag, label_array) tuples for bags with class > 0
        """
        return [
            (bag, label)
            for bag, cls, label in zip(self.bags, self.classes, self.bag_labels)
            if cls > 0.0
        ]

    @property
    def pos_bags(self):
        """
        Return deep copies of all positive bags.

        :return: List of bags whose class label is > 0
        """
        return deepcopy(
            [bag for bag, cls in zip(self.bags, self.classes) if cls > 0.0]
        )

    @property
    def neg_bags(self):
        """
        Return deep copies of all negative bags.

        :return: List of bags whose class label is <= 0
        """
        return deepcopy(
            [bag for bag, cls in zip(self.bags, self.classes) if cls <= 0.0]
        )

    @property
    def neg_instances(self):
        """
        Stack all negative instances into a single array.

        :return: Array of shape (L_n, d) with all negative instances
        """
        return np.vstack(self.neg_bags)

    @property
    def pos_instances(self):
        """
        Stack all positive instances into a single array.

        :return: Array of shape (L_p, d) with all positive instances
        """
        return np.vstack(self.pos_bags)

    @property
    def instances(self):
        """
        Stack all instances (negative first, then positive) into one array.

        :return: Array of shape (L, d) with all instances
        """
        return np.vstack([self.neg_instances, self.pos_instances])

    @property
    def inst_classes(self):
        """
        Return per-instance class labels.

        Negative instances are labeled -1, positive instances +1.

        :return: Column vector of shape (L, 1) with -1/+1 labels
        """
        return np.vstack(
            [-np.ones((self.L_n, 1)), np.ones((self.L_p, 1))]
        )

    @property
    def pos_groups(self):
        """
        Return instance counts per positive bag.

        :return: List of lengths for each positive bag
        """
        return [len(bag) for bag in self.pos_bags]

    @property
    def neg_groups(self):
        """
        Return instance counts per negative bag.

        :return: List of lengths for each negative bag
        """
        return [len(bag) for bag in self.neg_bags]

    @property
    def L_n(self):
        """
        Number of negative instances across all negative bags.

        :return: Integer count of negative instances
        """
        return len(self.neg_instances)

    @property
    def L_p(self):
        """
        Number of positive instances across all positive bags.

        :return: Integer count of positive instances
        """
        return len(self.pos_instances)

    @property
    def L(self):
        """
        Total number of instances (positive + negative).

        :return: Integer count of all instances
        """
        return self.L_p + self.L_n

    @property
    def X_n(self):
        """
        Number of negative bags.

        :return: Integer count of negative bags
        """
        return len(self.neg_bags)

    @property
    def X_p(self):
        """
        Number of positive bags.

        :return: Integer count of positive bags
        """
        return len(self.pos_bags)

    @property
    def X(self):
        """
        Total number of bags (positive + negative).

        :return: Integer count of all bags
        """
        return self.X_p + self.X_n

    @property
    def neg_inst_as_bags(self):
        """
        Flatten negative instances into a list, one instance per "bag".

        :return: List of instance arrays from all negative bags
        """
        return [inst for bag in self.neg_bags for inst in bag]

    @property
    def pos_inst_as_bags(self):
        """
        Flatten positive instances into a list, one instance per "bag".

        :return: List of instance arrays from all positive bags
        """
        return [inst for bag in self.pos_bags for inst in bag]

    @property
    def instance_intrabag_labels_pos(self):
        """
        Flatten intra-bag labels for all positive instances.

        Collects the intra-bag labels from each positive bag and concatenates
        them into a single 1D array.

        :return: Numpy array of intra-bag labels for positive instances
        """
        x = [label for _, label in self.pos_bags_with_labels]
        return np.concatenate(x)
