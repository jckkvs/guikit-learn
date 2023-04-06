
import os
import cloudpickle
import copy

from .base import get_estimator


def run_missing_data(missing_data, X, y, *kwargs):
    print(missing_data)
    if missing_data is None:
        return X, y
    X = missing_data.fit_transform(X)
    y = missing_data.fit_transform(y)

    return X, y

def run_feature_engineerings(feature_engineerings, X, y, *kwargs):
    for feature_engineering in feature_engineerings:
        if feature_engineering is None:
            continue
        else:
            X = feature_engineering.fit_transform(X)

    return X, y


def make_selector(selectors, X, y):
    selectors_cv = copy.deepcopy(selectors)
    [
        i.fit_transform(X, y)
        for i in selectors_cv
        if hasattr(i, "recalculate") == True
    ]
    [
        setattr(i, "recalculate", False)
        for i in selectors_cv
        if hasattr(i, "recalculate") == True
    ]

    selectors_dcv = copy.deepcopy(selectors)
    [
        setattr(i, "recalculate", True)
        for i in selectors_dcv
        if hasattr(i, "recalculate") == True
    ]

    return selectors_cv, selectors_dcv



# calculate vapnik-chervonenkis dimension
def calculate_vc_dimension(estimator):
    return 0

def run_scaler(scaler, X, *kwargs):
    if scaler is None:
        return X, scaler
    scaled = scaler.fit_transform(X)
    return scaled, scaler


def get_n_splits_(splitter, X, y=None, group=None, splitter_args=None):
    splitter.set_params(splitter_args)

    if splitter.__class__.__name__ in ["KFold", "LeaveOneOut", "LeavePOut"]:
        split = splitter.get_n_splits(X, y)

    elif splitter.__class__.__name__ in ["GroupKFold", "LeaveOneGroupOut"]:
        split = splitter.get_n_splits(X, y, group)

    elif splitter.__class__.__name__ == [
        "TimeSeriesSplit",
        "WalkForwardValidation",
    ]:
        split = splitter.get_n_splits(X, y)

    print("splitter", splitter)
    print("split", split)

    if True:
        split_test = copy.deepcopy(split)

        print(split_test)
        for idx, s in enumerate(split_test):
            print(s)
            if idx >= 10:
                break

    return split
