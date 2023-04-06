
import itertools
from inspect import signature
import ast

from sklearn.pipeline import Pipeline
import numpy as np

from ..utils.base import get_estimator, recursive_replace

def make_pipeline(steps, memory=None):
    """モデルのリストからPipelineを作成する

    Parameters
    ----------
    steps : list of sklearn instances.
        [StandardScaler(), LinearRegression()]

    Returns
    ----------
    Pipeline : sklearn.pipeline.Pipeline()
        Pipeline([(step_name, step_0), (step_name, step_1)])
        e.g. Pipeline([("scaler",StandardScaler()), ("estimator",LinearRegression())])
    """

    step_list = []
    for idx, step in enumerate(steps):
        each_step_name = getattr(step, "step_name", f"{idx}")
        step_list.append(tuple([each_step_name, step]))
    pipe = Pipeline(step_list, memory=memory)
    return pipe

def make_pipelines(
                        imputers=None,
                        scalers=None,
                        engineers=None,
                        selectors=None,
                        estimators=None,
                        memory=None
):

    if imputers is None:
        imputers = []
    if scalers is None:
        scalers = []
    if engineers is None:
        engineers = []
    if selectors is None:
        selectors = []
    if estimators is None:
        estimators = []

    memory=None

    estimator_product = itertools.product(
        imputers,
        scalers,
        engineers,
        selectors,
        estimators,
    )


    pipelines = [make_pipeline(steps, memory=memory) for steps in estimator_product]


    return pipelines

def make_args(string: str) -> dict:
    print("first string", string)
    if string == "":
        return {}

    string_list = string.split(",")
    print("first string_list", string_list)

    if len(string_list) == 0:
        return {}
    string_list = [i.split(":") for i in string_list]
    args = ['"' + i[0] + '"' + ":" + i[1] + "" for i in string_list]
    args = "{" + ",".join(args) + "}"
    args = recursive_replace(args, " ", "")
    print("args", args)
    args = ast.literal_eval(args)

    return args

def get_model_params(model, search_args="param_distributions"):
    """モデルのリストからハイパーパラメータの最適化範囲を作成する

    Parameters
    ----------
    steps : list of sklearn instances.
        [StandardScaler(), SelectFromModel(estimator=Lasso()), RandomForestRegressor()]
    search_args : name of attribute of param
        param_grid for GridSearchCV
        param_distributions for OptunaSearchCV

    Returns
    ----------
    param : dict
        e.g. {"selector__estimator__alpha":LogUniformDistribution(1e-5, 3e-2),
                "estimator__n_estimators":IntUniformDistribution(5, 200),
                }
    """
    param = {}

    if model.__class__ == Pipeline:
        for idx, step in enumerate(model):
            step_name = getattr(step, "step_name", f"{idx}")
            each_param_dict = getattr(step, search_args, {})
            each_param_dict = {
                f"{step_name}__{k}": v for k, v in each_param_dict.items()
            }
            param = {**param, **each_param_dict}
    else:
        param = getattr(model, search_args, {})

    return param

def model_fit(model, X, y, sample_weight=None, fit_params=None):
    if fit_params is None:
        fit_params = {}

    print("model")
    print(model)

    estimator, estimator_name = get_estimator(model, remove_multioutput=True)
    print(estimator)
    print(estimator_name)
    sig = signature(estimator.fit)
    estimator_fit_params = sig.parameters
    print("estimator_fit_params")
    print(estimator_fit_params)

    if "sample_weight" in estimator_fit_params:
        if model.__class__.__name__ == "Pipeline":
            fit_params[f"estimator__sample_weight"] = sample_weight
        else:
            fit_params["sample_weight"] = sample_weight

    print("fit_params")
    print(fit_params)

    if estimator_name not in ["TabNetRegressor"]:
        X = X
        y = y

    elif estimator_name in ["TabNetRegressor"]:
        X = np.array(X)
        y = np.reshape(np.array(y), (-1))

    model.fit(X, y, **fit_params)
    return

def get_params(model, args_name):
    param_dict = {}
    if model.__class__ == Pipeline:
        for each_model in model:
            params = getattr(each_model, args_name, {})
            param_dict = {**param_dict, **params}
    else:
        params = getattr(model, args_name, {})
        param_dict = {**param_dict, **params}

    return param_dict

def wrap_searchcv(model, SearchCV=None, params=None):
    if SearchCV is None:
        return model

    if SearchCV.__class__.__name__ == 'partial':
        _SearchCV = SearchCV.func
    else:
        _SearchCV = SearchCV

    sig_ = signature(_SearchCV)
    if "param_grid" in  sig_.parameters.keys():
        search_args = "param_grid"
    elif "param_distributions" in  sig_.parameters.keys():
        search_args = "param_distributions"
    else:
        print(f"{_SearchCV} has not param_grid or param_distributions")
        return model

    if params is None:
        params = get_model_params(model, search_args=search_args)

    wrapped_SearchCV = SearchCV(model, params)

    return wrapped_SearchCV

