

from inspect import signature
import time
import traceback

from dcekit.generative_model import GTM
import lightgbm
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import umap
import xgboost


def load_regression_models():
    models_set = {}

    try:
        from sklearn_expansion.auto_modeller import AutoGluonPredictor

        # AutoGluon
        estimator = AutoGluonPredictor(presets='best_quality')
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        model_name = 'AutoGluon'
        model = {'estimator':estimator,
                'default':True,
                'model_name':model_name ,
                'param_range':param_range,
                'param_grid':param_grid}

        models_set[model_name] = model
    except:
        pass

    return models_set


def load_classification_models():
    models_set = {}

    if False:
        models_set = {k:v for idx, (k,v) in enumerate(models_set.items()) if idx<=4}


    return models_set

def load_clustering_models():
    from sklearn.metrics import make_scorer
    from sklearn_expansion.metrics import k3nerror


    if False:
        models_set = {k:v for idx, (k,v) in enumerate(models_set.items()) if idx<=2}

    return models_set


def load_automl(ml_type='regression'):
    models_set = {} 
    
    if ml_type == 'regression':
        models_set = load_regression_models()
    elif ml_type == 'classification':
        models_set = load_classification_models()
    elif ml_type == 'clustering':
        models_set = load_clustering_models()

    return models_set
