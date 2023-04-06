import sklearn
from sklearn.model_selection import GridSearchCV
from optuna.integration import OptunaSearchCV

def load_tuning_models(models=None, searcher_type='OptunaSearchCV', cv=None, n_trials=None, timeout=600, scoring=None):
    tuning_models = []

    for each_model in models:
        if searcher_type == OptunaSearchCV:
            tuning_models.append(OptunaSearchCV(estimator=each_model['estimator'], param_range=each_model['param_range'], scoring=scoring))

        elif searcher_type == GridSearchCV:
            tuning_models.append(GridSearchCV(estimator=each_model['estimator'], param_range=each_model['param_grid'], scoring=scoring))

        elif searcher_type == None:
            tuning_models.append(each_model['estimator'])
        else:
            tuning_models.append(each_model['estimator'])


    return tuning_models
