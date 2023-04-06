
# standard
import configparser
from distutils.util import strtobool
from functools import partial
import itertools
from joblib import memory
import os
import sys
import tempfile
from pathlib import Path
import warnings

# third-party
from sklearn.model_selection import GridSearchCV
from optuna.integration import OptunaSearchCV

# this library
from ..scalers._scalers import load_scalers
from ..imputers._imputers import load_imputers
from ..feature_selection._selectors import load_selectors
from ..feature_engineering._engineers import load_engineers
from ..estimators._estimators import load_all_models
from ..model_selection.cv import load_splliter_dict

from ..utils.create_model import make_pipeline, make_pipelines
from ..utils.create_model import make_args, get_model_params, model_fit, wrap_searchcv
from ..utils.base import get_estimator

# from ..utils.base import find_data_file

def _get_file_path(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen
        print("frozen")
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        print(".py")
        datadir = os.path.dirname(__file__)
    return Path(datadir) / filename # os.path.join(datadir, filename)

def apply_given_n_jobs(model):
    # estimatorにn_trialsが指定されている場合はその値を参照する
    each_estimator, _ = get_estimator(
        model, remove_multioutput=False
    )
    if hasattr(each_estimator, "n_trials") == True:
        setattr(
            model,
            "n_trials",
            getattr(each_estimator, "n_trials"),
        )
    return model

def set_n_jobs_1(model, param_name="param_distributions"):
    if len(getattr(model, param_name, {})) == 0:
        setattr(model, "n_trials", 1)
    return model

def load_models(config_path=_get_file_path("default.ini")):

    config = configparser.ConfigParser()
    print(config_path)
    config.read(config_path)
    print(config)

    ml_type = config["ml_type"]["ml_type"]
    imputer_dict = dict(config["imputer"])
    scaler_dict = dict(config["scaler"])
    engineer_dict = dict(config["engineer"])
    selector_dict = dict(config["selector"])
    estimator_dict = dict(config["estimator"])

    def apply_config(models, model_dict):
        """modelのリストに対して、defaultとconfig_dictを適用する

        Parameters
        ----------
        models : list
            list of model
        model_dict : dict
            config dict of model

        Returns
        -------
        new_models : list
            new_models

        Notes
        -----
        configの設定を優先し、configで設定されていない場合はdefaultの設定を用いる
        default==Trueの場合はモデル採用
        default==Falseの場合はモデル非採用
        defaultが設定されていない場合はモデル採用

        """
        new_models = []

        for model in models:
            model_name = getattr(model, "model_name", "")


            if model_name in model_dict.keys():
                bool_ = strtobool(model_dict[model_name])

                if bool_ == True:
                    new_models.append()
                if bool_ == False:
                    continue
            else:
                default = getattr(model, "default", None)

                if default == True:
                    new_models.append(model)
                elif default == False:
                    continue
                elif default is None:
                    new_models.append(model)

        return new_models

    def change_args_type(args_dict):
        int_list = ["cv", "max_iter", "n_jobs", "n_trials", "random_state", "verbose",
                    "n_splits", "n_groups", "p"]
        float_list = ["subsample"]

        new_args_dict = {}
        for k, v in args_dict.items():
            if k in int_list:
                if v is not None:
                    v = int(v)
            elif k in float_list:
                if v is not None:
                    v = float(v)

            new_args_dict[k] = v

        return new_args_dict

    imputers = load_imputers()
    imputers = apply_config(imputers, imputer_dict)

    scalers = load_scalers()
    scalers = apply_config(scalers, scaler_dict)

    engineers = load_engineers()
    engineers = apply_config(engineers, engineer_dict)

    selectors = load_selectors()
    selectors = apply_config(selectors, selector_dict)

    estimators = load_all_models(ml_type=ml_type)
    estimators = apply_config(estimators, estimator_dict)

    cache = memory.Memory(location=tempfile.TemporaryDirectory().name, verbose=0)

    pipelines = make_pipelines(
                            imputers=imputers,
                            scalers=scalers,
                            engineers=engineers,
                            selectors=selectors,
                            estimators=estimators,
                            memory=cache
                        )

    SearchCV_method = config["SearchCV"]["method"]
    SearchCV_args = dict(config["SearchCV_args"])
    SearchCV_args = change_args_type(SearchCV_args)
    print(SearchCV_args)

    SearchCV_cv = config["SearchCV_cv"]["cv"]
    SearchCV_cv_args = dict(config["SearchCV_cv_args"])
    SearchCV_cv_args = change_args_type(SearchCV_cv_args)

    splitter_dict = load_splliter_dict()
    cv = splitter_dict[SearchCV_cv]["instance"]
    for k, v in SearchCV_cv_args:
        if hasattr(cv, k):
            setattr(cv, k , v)

    if SearchCV_method == "optuna":
        param_name = "param_distributions"
        try:
            SearchCV = partial(OptunaSearchCV, cv=cv, **SearchCV_args)
        except:
            warnings.warn("failed to set cv_args")
            print("failed")
            SearchCV = partial(OptunaSearchCV)

    elif SearchCV_method == "gridsearch":
        param_name = "param_grid"
        try:
            SearchCV = partial(GridSearchCV, **SearchCV_args)
        except:
            print("failed")
            warnings.warn("failed to set cv_args")
            SearchCV = partial(GridSearchCV)
    else:
        param_name = ""
        SearchCV = None

    wrapped_models = [wrap_searchcv(i, SearchCV) for i in pipelines]
    print(wrapped_models[0:10])

    # estimatorにn_trialsが指定されている場合はその値を参照する
    """
    for each_pipelines_dcv in wrapped_models:
        each_estimator, _ = get_estimator(
            each_pipelines_dcv, remove_multioutput=False
        )
        if hasattr(each_estimator, "n_trials") == True:
            setattr(
                each_pipelines_dcv,
                "n_trials",
                getattr(each_estimator, "n_trials"),
            )
    """

    wrapped_models = [apply_given_n_jobs(model) for model in wrapped_models]

    # ハイパーパラメータを最適化しない場合はOptunaSearchCVのn_trialsを1とする # 0でも動作はするが、一応1にしておく
    wrapped_models = [set_n_jobs_1(model, param_name) for model in wrapped_models]

    return wrapped_models

if __name__ == "__main__":
    load_models()