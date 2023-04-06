import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.distributions import CategoricalDistribution


import sklearn_expansion
from sklearn_expansion.identity_mapping import IdentityMapping
from sklearn_expansion.feature_selection import SelectByModel

from sklearn.preprocessing import FunctionTransformer


def load_regression_selectors():
    selectors = []

    # FunctionTransformer:No selector
    selector = FunctionTransformer()
    estimator_param_range = {}
    estimator_param_grid = {}
    estimator_param_distributions = {}
    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {}
    selector_name = "None"
    param_range = {}
    param_grid = {}
    param_distributions = {}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = True
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # PCA
    selector = sklearn.decomposition.PCA()
    selector_name = "PCA"
    param_range = {}
    param_grid = {}
    param_distributions = {"n_components": IntDistribution(2, 10, log=False)}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # SequentialFeatureSelector(Lasso)
    estimator = Lasso()
    estimator_param_range = {"estimator__alpha": {"suggest_type": "log", "range": [1e-5, 0.1]}}
    estimator_param_distributions = {
        "estimator__alpha": FloatDistribution(1e-5, 0.1, log=True),
    }
    estimator_param_grid = {}

    cv  = KFold(n_splits=5, shuffle=True)
    selector = SequentialFeatureSelector(estimator=estimator, cv=cv, n_jobs=-1)
    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {}
    selector_name = "f-SFS_Lasso"

    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # SequentialFeatureSelector(SVR)
    estimator = sklearn.svm.SVR(kernel="rbf")
    estimator_param_range = {
        "estimator__C": {"suggest_type": "log", "range": [1e-2, 1e3]},
        "estimator__epsilon": {"suggest_type": "log", "range": [1e-1, 1e3]},
        "estimator__gamma": {"suggest_type": "log", "range": [1e-3, 1e2]},
    }
    estimator_param_distributions = {
        "estimator__C": FloatDistribution(1e-1, 1e2, log=True),
        "estimator__epsilon": FloatDistribution(1e-1, 1e3, log=True),
        "estimator__gamma": FloatDistribution(1e-3, 1e2, log=True),
    }
    estimator_param_grid = {}

    cv  = KFold(n_splits=5, shuffle=True)
    selector = SequentialFeatureSelector(estimator=estimator, cv=cv, n_jobs=-1)
    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {}
    selector_name = "f-SFS_SVR"

    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as m_SFS
        try:
            # SequentialFeatureSelector(mlxtend-LGBM)
            import lightgbm

            estiimator_in_selecor = lightgbm.LGBMRegressor()
            selector = m_SFS(estimator=estiimator_in_selecor,
                                k_features=5,
                                forward=True,
                                floating=True,
                                scoring="neg_mean_squared_error",
                                cv=KFold(n_splits=5,shuffle=True),
                                n_jobs=-1)
            estimator_param_range = {}
            estimator_param_grid = {}
            estimator_param_distributions = {}
            selector_param_range = {}
            selector_param_grid = {}
            selector_param_distributions = {"k_features": IntDistribution(2, 10, log=False)}
            selector_name = "SequentialForwardFloating-LGBM"
            param_range = {}
            param_grid = {}
            param_distributions = {}

            selector.model_name = selector_name
            selector.step_name = "selector"
            selector.default = False
            selector.param_range = param_range
            selector.param_grid = param_grid
            selector.param_distributions = param_distributions
            selectors.append(selector)

            # SequentialFeatureSelector(LGBM)
            estimator = lightgbm.LGBMRegressor(n_jobs=-1)
            estimator = MultiOutputRegressor(estimator)
            estimator_param_range = {}
            estimator_param_distributions = {
                "estimator__estimator__num_leaves": IntDistribution(30, 100),
                "estimator__estimator__feature_fraction": FloatDistribution(0.05, 0.95),
                "estimator__estimator__bagging_fraction": FloatDistribution(0.8, 1.0),
                "estimator__estimator__max_depth": IntDistribution(3, 15),
                "estimator__estimator__min_split_gain": FloatDistribution(1e-3, 1e-1, log=True),
                "estimator__estimator__min_child_weight": IntDistribution(5, 20),
            }
            estimator_param_grid = {}

            cv  = KFold(n_splits=5, shuffle=True)
            selector = SequentialFeatureSelector(estimator=estimator, cv=cv, n_jobs=-1)
            selector_param_range = {}
            selector_param_grid = {}
            selector_param_distributions = {}
            selector_name = "f-SFS_LGBM"

            param_distributions = {
                **estimator_param_distributions,
                **selector_param_distributions,
            }
            param_range = {**estimator_param_range, **selector_param_range}
            param_grid = {**estimator_param_grid, **selector_param_grid}

            selector.model_name = selector_name
            selector.step_name = "selector"
            selector.default = False
            selector.param_range = param_range
            selector.param_grid = param_grid
            selector.param_distributions = param_distributions
            selectors.append(selector)
        except:
            pass
    except:
        pass

    # SelectByModel Lasso
    from sklearn_expansion.feature_selection import SelectByModel

    estimator = Lasso()
    estimator_param_range = {
        "estimator__alpha": {"suggest_type": "log", "range": [1e-2, 3e-2]}
    }

    estimator_param_distributions = {
        "estimator__alpha": FloatDistribution(1e-5, 3e-2, log=True)
    }

    estimator_param_grid = {"estimator__alpha": [1e-2, 5e-2, 1e-1, 5e-1]}

    selector = SelectByModel(estimator)
    selector_param_range = {"threshold": {"suggest_type": "log", "range": [1e-5, 0.3]}}
    selector_param_distributions = {"threshold": FloatDistribution(1e-5, 0.3, log=True)}
    selector_param_grid = {"threshold": [1e-5, 1e-3, 1e-1]}
    selector_name = "Lasso"

    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # SelectByModel RandomForest
    estimator = RandomForestRegressor()
    estimator_param_range = {
        "estimator__max_depth": {"suggest_type": "int", "range": [3, 13]},
        "estimator__n_estimators": {"suggest_type": "int", "range": [5, 200]},
    }

    estimator_param_distributions = {
        "estimator__max_depth": IntDistribution(3, 13),
        "estimator__n_estimators": IntDistribution(5, 200, log=True),
    }

    estimator_param_grid = {
        "estimator__max_depth": [5, 7, 11],
        "estimator__n_estimators": [50, 100, 200],
    }

    selector = SelectByModel(estimator)

    selector_param_range = {
        "threshold": {
            "suggest_type": "choice",
            "range": [
                [
                    "median",
                    "mean",
                ]
            ],
        },
    }
    selector_param_distributions = {
        "threshold": CategoricalDistribution(choices=["median", "mean"]),
    }
    selector_param_grid = {
        "threshold": [
            "median",
            "mean",
        ],
    }  # 'magnification'   :[1.25, 1.5]

    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}
    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }

    selector_name = "RandomForest"

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    try:
        from sklearn_expansion.feature_selection import SelectByGACV, SelectByModel
        # SelectByGACV-PLS
        estimator = PLSRegression()
        estimator_param_range = {
            "estimator__n_components": {"suggest_type": "int", "range": [2, 50]}
        }
        estimator_param_grid = {"estimator__n_components": [5, 15, 30, 50]}

        estimator_param_distributions = {
            "estimator__n_components": IntDistribution(2, 50),
        }

        # selector = GeneticSelectionCV(estimator, cv=5)
        selector = SelectByGACV(estimator, cv=5)

        selector_param_range = {}
        selector_param_grid = {}
        selector_param_distributions = {}

        selector_name = "GAPLS"

        param_range = {**estimator_param_range, **selector_param_range}
        param_grid = {**estimator_param_grid, **selector_param_grid}
        param_distributions = {
            **estimator_param_distributions,
            **selector_param_distributions,
        }

        selector.model_name = selector_name
        selector.step_name = "selector"
        selector.default = False
        selector.param_range = param_range
        selector.param_grid = param_grid
        selector.param_distributions = param_distributions
        selectors.append(selector)
    except:
        pass

    try:
        from genetic_selection import GeneticSelectionCV
        # SelectByGACV-PLS
        estimator = PLSRegression()
        estimator_param_range = {
            "estimator__n_components": {"suggest_type": "int", "range": [2, 50]}
        }
        estimator_param_grid = {"estimator__n_components": [5, 15, 30, 50]}

        estimator_param_distributions = {
            "estimator__n_components": IntDistribution(2, 50),
        }

        selector = GeneticSelectionCV(estimator, cv=KFold(5, shuffle=True), n_population=100, n_generations=30)

        selector_param_range = {}
        selector_param_grid = {}
        selector_param_distributions = {}

        selector_name = "GeneticSelectionCV"

        param_range = {**estimator_param_range, **selector_param_range}
        param_grid = {**estimator_param_grid, **selector_param_grid}
        param_distributions = {
            **estimator_param_distributions,
            **selector_param_distributions,
        }

        selector.model_name = selector_name
        selector.step_name = "selector"
        selector.default = False
        selector.param_range = param_range
        selector.param_grid = param_grid
        selector.param_distributions = param_distributions
        selectors.append(selector)
    except:
        pass

    try:
        from boruta import BorutaPy
        # SelectByBoruta
        estimator = RandomForestRegressor()
        estimator_param_range = {
            "estimator__max_depth": {"suggest_type": "int", "range": [3, 13]},
            "estimator__n_estimators": {"suggest_type": "int", "range": [5, 200]},
        }
        estimator_param_grid = {
            "estimator__max_depth": [5, 7, 11],
            "estimator__n_estimators": [50, 100, 200],
        }

        estimator_param_distributions = {
            "estimator__max_depth": IntDistribution(3, 13),
            "estimator__n_estimators": IntDistribution(5, 200, log=True),
        }

        selector = BorutaPy(estimator, n_estimators="auto")

        selector_param_range = {}
        selector_param_grid = {}
        selector_param_distributions = {}

        selector_name = "Boruta"

        param_range = {**estimator_param_range, **selector_param_range}
        param_grid = {**estimator_param_grid, **selector_param_grid}
        param_distributions = {
            **estimator_param_distributions,
            **selector_param_distributions,
        }

        selector.model_name = selector_name
        selector.step_name = "selector"
        selector.default = False
        selector.param_range = param_range
        selector.param_grid = param_grid
        selector.param_distributions = param_distributions
        selectors.append(selector)
    except:
        pass

    try:
        from skrebate import ReliefF, MultiSURFstar

        # ReliefF
        selector = ReliefF(discrete_threshold=15)
        estimator_param_range = {}
        estimator_param_grid = {}
        estimator_param_distributions = {}
        selector_param_range = {}
        selector_param_grid = {}
        selector_param_distributions = {
            "n_features_to_select": IntDistribution(5, 100, log=False),
            "n_neighbors": IntDistribution(3, 30, log=False),
        }

        selector_name = "ReliefF"
        param_range = {**estimator_param_range, **selector_param_range}
        param_grid = {**estimator_param_grid, **selector_param_grid}
        param_distributions = {
            **estimator_param_distributions,
            **selector_param_distributions,
        }
        selector.model_name = selector_name
        selector.step_name = "selector"
        selector.default = False
        selector.param_range = param_range
        selector.param_grid = param_grid
        selector.param_distributions = param_distributions
        selectors.append(selector)


        # MultiSURFstar
        selector = MultiSURFstar(discrete_threshold=15)
        estimator_param_range = {}
        estimator_param_grid = {}
        estimator_param_distributions = {}
        selector_param_range = {}
        selector_param_grid = {}
        selector_param_distributions = {
            "n_features_to_select": IntDistribution(5, 100, log=False),
        }

        selector_name = "MultiSURFstar"
        param_range = {**estimator_param_range, **selector_param_range}
        param_grid = {**estimator_param_grid, **selector_param_grid}
        param_distributions = {
            **estimator_param_distributions,
            **selector_param_distributions,
        }
        selector.model_name = selector_name
        selector.step_name = "selector"
        selector.default = False
        selector.param_range = param_range
        selector.param_grid = param_grid
        selector.param_distributions = param_distributions
        selectors.append(selector)
    except:
        pass

    return selectors


def load_classification_selectors():
    selectors = {}

    # IdentityMapping:None
    selector = FunctionTransformer()
    estimator_param_range = {}
    estimator_param_grid = {}
    selector_param_range = {}
    selector_param_grid = {}
    selector_name = "None"

    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = True
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # SelectByModel RandomForest
    estimator = RandomForestClassifier()
    estimator_param_range = {
        "estimator__max_depth": {"suggest_type": "int", "range": [3, 13]},
        "estimator__n_estimators": {"suggest_type": "int", "range": [5, 200]},
    }
    estimator_param_distributions = {
        "estimator__max_depth": IntDistribution(3, 13),
        "estimator__n_estimators": IntDistribution(5, 200, log=False),
    }
    estimator_param_distributions = {
        "estimator__alpha": FloatDistribution(1e-5, 1e-1, log=False)
    }

    selector = sklearn_expansion.feature_selection.SelectByModel(
        OptunaSearchCV(estimator, estimator_param_range)
    )

    selector_param_range = {
        "threshold": {
            "suggest_type": "choice",
            "range": [
                [
                    "median",
                    "mean",
                ]
            ],
        },
        "magnification": {"suggest_type": "uniform", "range": [0.5, 1.5]},
    }

    selector_param_grid = {
        "threshold": [
            "median",
            "mean",
        ],
        "magnification": [1.25, 1.5],
    }
    selector_param_distributions = {"threshold": FloatDistribution(1e-5, 0.3, log=True)}

    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}
    selector_name = "RandomForest"

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # SelectByBoruta
    estimator = RandomForestClassifier()
    estimator_param_range = {
        "estimator__max_depth": {"suggest_type": "int", "range": [3, 13]},
        "estimator__n_estimators": {"suggest_type": "int", "range": [5, 200]},
    }
    estimator_param_distributions = {
        "estimator__max_depth": IntDistribution(3, 13),
        "estimator__n_estimators": IntDistribution(5, 200, log=True),
    }

    estimator_param_grid = {
        "estimator__max_depth": [5, 7, 11],
        "estimator__n_estimators": [50, 100, 200],
    }

    selector = sklearn_expansion.feature_selection.SelectByBoruta(
        OptunaSearchCV(estimator, estimator_param_range)
    )

    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {}

    param_range = {**estimator_param_range, **selector_param_range}

    param_grid = {**estimator_param_grid, **selector_param_grid}

    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }

    selector_name = "Boruta"

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)


    # ReliefF
    selector = ReliefF()
    estimator_param_range = {}
    estimator_param_grid = {}
    estimator_param_distributions = {}
    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {
        "n_features_to_select": IntDistribution(1, 10),
        "n_neighbors": IntDistribution(1, 10),
        "discrete_limit": IntDistribution(1, 10),
    }

    selector_name = "ReliefF"
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}
    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }
    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    # MultiSURFstar
    selector = MultiSURFstar(discrete_threshold=15)
    estimator_param_range = {}
    estimator_param_grid = {}
    estimator_param_distributions = {}
    selector_param_range = {}
    selector_param_grid = {}
    selector_param_distributions = {
        "n_features_to_select": IntDistribution(1, 100, log=True),
    }

    selector_name = "MultiSURFstar"
    param_range = {**estimator_param_range, **selector_param_range}
    param_grid = {**estimator_param_grid, **selector_param_grid}
    param_distributions = {
        **estimator_param_distributions,
        **selector_param_distributions,
    }

    selector.model_name = selector_name
    selector.step_name = "selector"
    selector.default = False
    selector.param_range = param_range
    selector.param_grid = param_grid
    selector.param_distributions = param_distributions
    selectors.append(selector)

    return selectors


def load_selectors(ml_type="regression"):
    if ml_type == "regression":
        models = load_regression_selectors()
    elif ml_type == "classification":
        models = load_classification_selectors()
    else:
        raise ValueError("Models other than regression have not yet been defined!")
    return models
