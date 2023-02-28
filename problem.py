import os

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import Estimator
from sklearn.model_selection import KFold

problem_title = "Corporal Accidents Severity classification"


# -----------------------------------------------------------------------------
# Workflow element
# -----------------------------------------------------------------------------

workflow = Estimator()

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------

Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1, 2, 3])

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

score_types = [
    rw.score_types.NegativeLogLikelihood(name="loss"),
    rw.score_types.F1Above(name="f1"),
    rw.score_types.BalancedAccuracy(name="balanced_accuracy"),
    rw.score_types.MacroAveragedRecall(name="average_recall"),
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):
    # using 5 folds as default
    k = 5
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
    n_splits = 5
    cv = KFold(n_splits=n_splits)
    splits = list(cv.split(X, y))
    # 5 folds, each point is in test set 4x
    # set k to a lower number if you want less folds
    pattern = [
        ([2, 3, 4], [0, 1]),
        ([0, 1, 4], [2, 3]),
        ([0, 2, 3], [1, 4]),
        ([0, 1, 3], [2, 4]),
        ([1, 2, 4], [0, 3]),
        ([0, 1, 2], [3, 4]),
        ([0, 2, 4], [1, 3]),
        ([1, 2, 3], [0, 4]),
        ([0, 3, 4], [1, 2]),
        ([1, 3, 4], [0, 2]),
    ]
    for ps in pattern[:k]:
        yield (
            np.hstack([splits[p][1] for p in ps[0]]),
            np.hstack([splits[p][1] for p in ps[1]]),
        )


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type_):
    fname = "data_{}.csv".format(type_)
    fp = os.path.join(path, "data", fname)
    data = pd.read_csv(fp, low_memory=False, index_col="Unnamed: 0")

    # Format the columns "pr" and "pr1"
    data.loc[data["pr1"] == "(1)", "pr1"] = "1"
    data.loc[data["pr"] == "(1)", "pr"] = "1"
    data["pr"] = data["pr"].astype(int)
    data["pr1"] = data["pr1"].astype(int)

    fname = "labels_{}.csv".format(type_)
    fp = os.path.join(path, "data", fname)
    y = pd.read_csv(fp) - 1

    # for the "quick-test" mode, use less data
    test = os.getenv("RAMP_TEST_MODE", 0)
    if test:
        N_small = 35000
        data = data[:N_small]
        y = y[:N_small]

    return data, y.astype("int").to_numpy().ravel()


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")
