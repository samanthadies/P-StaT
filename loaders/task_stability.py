"""
task_stability.py

Task helper class for mapping base labels into task-specific
binary targets for probing experiments.

"""

import numpy as np
import logging
from copy import deepcopy

log = logging.getLogger("task")


class Task:
    """
    Handle task definition and label remapping for different probing targets.

    Each task maps raw correctness/realness and dataset-type indicators
    into a binary TRUE-vs-ALL style label.
    """

    def __init__(self, task=0):
        """
        Initialize a Task.

        :param task: integer task id in {0, 1, 2, 3, 4, 5}
        """
        assert task in [0, 1, 2, 3, 4, 5]

        if task == 0:
            log.warning("The TASK is set to: TRUE-vs-ALL")
        elif task == 1:
            log.warning("The TASK is set to: TRUE+s-vs-ALL")
        elif task == 2:
            log.warning("The TASK is set to: TRUE+sf-vs-ALL")
        elif task == 3:
            log.warning("The TASK is set to: TRUE+f-vs-ALL")
        elif task == 4:
            log.warning("The TASK is set to: TRUE+f_t-vs-ALL")
        elif task == 5:
            log.warning("The TASK is set to: TRUE+n-vs-ALL")

        self.task = task

    def _return_labels(
        self,
        correct,
        real,
        fictional=None,
        noise=None,
        synthetic_fic=None,
    ):
        """
        Internal helper to construct task-specific targets and masks.

        :param correct: numpy array of correctness labels (0/1)
        :param real: numpy array of realness labels (0/1)
        :param fictional: optional numpy array of fictional labels (0/1)
        :param noise: optional numpy array of noise labels (0/1)
        :param synthetic_fic: optional numpy array of synthetic_fic labels (0/1)
        :return: dict with keys 'targets' and 'mask'
        """
        assert self.is_binary(correct), "The correct labels must be binary"
        assert self.is_binary(real), "The real labels must be binary"
        assert real.shape == correct.shape, "The shapes must be the same"

        correct = np.copy(correct)
        real = np.copy(real)

        if fictional is None:
            fictional = np.zeros_like(correct)
        if noise is None:
            noise = np.zeros_like(correct)
        if synthetic_fic is None:
            synthetic_fic = np.zeros_like(correct)

        if self.task == 0:
            # TRUE-vs-ALL
            new_targets = np.zeros_like(correct)
            new_targets[(correct == 1) & (real == 1)] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}

        elif self.task == 1:
            # TRUE+s-vs-ALL
            # Important: exclude fictional, noise, and synthetic_fic rows.
            new_targets = np.zeros_like(correct)
            pos_mask = ((correct == 1) & (real == 1)) | (
                (real == 0)
                & (fictional == 0)
                & (noise == 0)
                & (synthetic_fic == 0)
            )
            new_targets[pos_mask] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}

        elif self.task == 2:
            # TRUE+sf-vs-ALL
            new_targets = np.zeros_like(correct)
            pos_mask = ((correct == 1) & (real == 1)) | (synthetic_fic == 1)
            new_targets[pos_mask] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}

        elif self.task == 3:
            # TRUE+f-vs-ALL
            new_targets = np.zeros_like(correct)
            pos_mask = ((correct == 1) & (real == 1)) | (fictional == 1)
            new_targets[pos_mask] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}

        elif self.task == 4:
            # TRUE+f_t-vs-ALL
            new_targets = np.zeros_like(correct)
            pos_mask = ((correct == 1) & (real == 1)) | (
                (correct == 1) & (fictional == 1)
            )
            new_targets[pos_mask] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}

        elif self.task == 5:
            # TRUE+n-vs-ALL
            new_targets = np.zeros_like(correct)
            pos_mask = ((correct == 1) & (real == 1)) | (noise == 1)
            new_targets[pos_mask] = 1
            return {"targets": new_targets, "mask": np.ones_like(real)}


    def return_labels(
        self,
        correct,
        real,
        fictional=None,
        noise=None,
        synthetic_fic=None,
    ):
        """
        Public method to construct detached labels and masks.

        :param correct: numpy array of correctness labels (0/1)
        :param real: numpy array of realness labels (0/1)
        :param fictional: optional numpy array of fictional labels (0/1)
        :param noise: optional numpy array of noise labels (0/1)
        :param synthetic_fic: optional numpy array of synthetic_fic labels (0/1)
        :return: dict with keys 'targets' (np.ndarray) and 'mask' (bool np.ndarray)
        """
        x = self._return_labels(
            correct,
            real,
            fictional=fictional,
            noise=noise,
            synthetic_fic=synthetic_fic,
        )
        return {
            "targets": deepcopy(x["targets"]),
            "mask": deepcopy(x["mask"].astype(bool)),
        }

    def is_binary(self, x):
        """
        Check whether an array is binary (only 0/1 or True/False values).

        :param x: array-like of labels
        :return: True if labels are contained in {0, 1}, else False
        """
        values = set(np.asarray(x).astype(int))
        return values.issubset({0, 1})