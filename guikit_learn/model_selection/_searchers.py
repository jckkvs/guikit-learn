from sklearn.model_selection import GridSearchCV
from optuna.integration import OptunaSearchCV

from functools import partial

def load_searchcv(n_jobs=None, cv=None, random_state=None, method="optuna"):
    if method == "optuna":
        # OptunaSearchCV
        scv = partial(OptunaSearchCV, cv=cv, n_jobs=n_jobs)
        scv.model_name = "OptunaSearchCV"
        return scv

    elif method == "grid":
        # GridSearchCV
        scv = partial(GridSearchCV, cv=cv, n_jobs=n_jobs)
        scv.model_name = "GridSearchCV"
        return scv

    elif method is None:
        return None

    else:
        return None

