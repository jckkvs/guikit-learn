from ensurepip import bootstrap
from inspect import signature
import time
import traceback

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.manifold import TSNE


from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)

from sklearn_expansion.identity_mapping import IdentityMapping

from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    WhiteKernel,
    RBF,
    Matern,
)

def load_regression_models():
    """GUI上にロードしたいsklearn-compatibleなregressionモデルをmodelsに登録する
    Parameters
    ---------

    Returns
    ---------
    models : list
        e.g. [LinearRegression(), RandomForestRegressor(), and so on..]

    Examples
    ---------
        models = []
        model = sklearn.linear_model.LinearRegression()
        models.append(model)

    Notes
    ---------
        modelsにはsklearnのインスタンスを登録するだけでも動作する
            その他全ての項目（model_name, param_range, default等）はなくても最低限の動作は可能
        param_grid, param_distributionsはハイパーパラメータの最適化範囲
            param_gridはGridSearchCV用の範囲
            param_distributionsはOptunaSearchCV用の最適化範囲
        model_nameは解析結果の表示に用いる。
            model_nameを指定しない場合はestimator.__class__.__name__が用いられる
        defaultは解析に使うかどうかの初期設定
            一般的なモデルはTrue、マイナーなモデルはFaultに設定している
        step_nameはsklearn.Pipelineのstepsのnameに用いる
            typeはモデルの大分類：Linear, Kernel, Tree, Linear-Tree, othersなど
            GUI上でモデルを選択する際の表示順に用いる
        shapはSHAPによる解析を行うかどうか
            Trueで解析する、Falseで解析しない
        libraryはそのモデルを実装しているライブラリー名
            例えば[sklearnライブラリのモデルだけ解析に使いたい場合]などの応用のために実装した

    """
    models = []

    # linear
    estimator = sklearn.linear_model.LinearRegression()
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "Linear"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # Theil-Sen
    estimator = sklearn.linear_model.TheilSenRegressor()
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "Theil-Sen"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # Lasso
    estimator = sklearn.linear_model.Lasso()
    param_range = {"alpha": {"suggest_type": "log", "range": [1e-5, 0.1]}}
    param_distributions = {
        "alpha": FloatDistribution(1e-5, 0.1, log=True),
    }

    param_grid = {}
    model_name = "Lasso"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # Ridge
    estimator = sklearn.linear_model.Ridge()
    param_range = {"alpha": {"suggest_type": "log", "range": [1e-2, 1e2]}}
    param_grid = {}
    param_distributions = {
        "alpha": FloatDistribution(1e-2, 1e2, log=True),
    }
    model_name = "Ridge"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # Elastic-Net
    estimator = sklearn.linear_model.ElasticNet()
    param_range = {
        "alpha": {"suggest_type": "log", "range": [1e-2, 1e2]},
        "l1_ratio": {"suggest_type": "uniform", "range": [0, 1]},
    }
    param_grid = {}
    param_distributions = {
        "alpha": FloatDistribution(1e-2, 1e2, log=True),
        "l1_ratio": FloatDistribution(0, 1, log=False),
    }

    model_name = "Elastic-Net"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # ARDRegression
    estimator = sklearn.linear_model.ARDRegression()
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "ARDRegression"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # BayesianRidge
    estimator = sklearn.linear_model.BayesianRidge()
    estimator = MultiOutputRegressor(estimator)

    param_range = {
        "estimator__alpha_1": {"suggest_type": "log", "range": [1e-8, 1e-4]},
        "estimator__alpha_2": {"suggest_type": "log", "range": [1e-8, 1e-4]},
        "estimator__lambda_1": {"suggest_type": "log", "range": [1e-8, 1e-4]},
        "estimator__lambda_2": {"suggest_type": "log", "range": [1e-8, 1e-4]},
    }

    param_distributions = {
        "estimator__alpha_1": FloatDistribution(1e-8, 1e-4, log=True),
        "estimator__alpha_2": FloatDistribution(1e-8, 1e-4, log=True),
        "estimator__lambda_1": FloatDistribution(1e-8, 1e-4, log=True),
        "estimator__lambda_2": FloatDistribution(1e-8, 1e-4, log=True),
    }
    param_grid = {}
    model_name = "BayesianRidge"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    # Stochastic Gradient Descent
    estimator = sklearn.linear_model.SGDRegressor()
    estimator = MultiOutputRegressor(estimator)

    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "SGD"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # KernelRidge
    from sklearn.kernel_ridge import KernelRidge

    estimator = KernelRidge(kernel="rbf")
    param_range = {
        "alpha": {"suggest_type": "log", "range": [1e-4, 1e2]},
        "gamma": {"suggest_type": "log", "range": [1e-3, 1e2]},
    }
    param_distributions = {
        "alpha": FloatDistribution(1e-4, 1e2, log=True),
        "gamma": FloatDistribution(1e-3, 1e2, log=True),
    }
    param_grid = {}
    model_name = "rbf-KernelRidge"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # KernelRidge
    from sklearn.kernel_ridge import KernelRidge

    estimator = KernelRidge(kernel="linear")
    param_range = {
        "alpha": {"suggest_type": "log", "range": [1e-4, 1e2]},
        "gamma": {"suggest_type": "log", "range": [1e-3, 1e2]},
    }
    param_distributions = {
        "alpha": FloatDistribution(1e-4, 1e2, log=True),
        "gamma": FloatDistribution(1e-3, 1e2, log=True),
    }
    param_grid = {}
    model_name = "linear-KernelRidge"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # PLS
    # guikit-learn.pyでn_componentsの最大値をn_rankとしている
    from sklearn.cross_decomposition import PLSRegression

    estimator = PLSRegression()
    param_range = {"n_components": {"suggest_type": "int", "range": [2, 100]}}
    param_distributions = {
        "n_components": IntDistribution(2, 100, log=False),
    }

    param_grid = {}
    model_name = "PLS"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # Linear-SVR
    estimator = sklearn.svm.LinearSVR()
    estimator = MultiOutputRegressor(estimator)

    param_range = {
        "estimator__C": {"suggest_type": "log", "range": [1e-1, 1e2]},
        "estimator__epsilon": {"suggest_type": "log", "range": [1e-2, 1e2]},
    }
    param_grid = {}
    param_distributions = {
        "estimator__C": FloatDistribution(1e-1, 1e2, log=True),
        "estimator__epsilon": FloatDistribution(1e-2, 1e2, log=True),
    }

    model_name = "LSVR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # SVR (rbf)
    estimator = sklearn.svm.SVR(kernel="rbf")
    estimator = MultiOutputRegressor(estimator)

    param_range = {
        "estimator__C": {"suggest_type": "log", "range": [1e-2, 1e3]},
        "estimator__epsilon": {"suggest_type": "log", "range": [1e-1, 1e3]},
        "estimator__gamma": {"suggest_type": "log", "range": [1e-3, 1e2]},
    }
    param_grid = {}

    param_distributions = {
        "estimator__C": FloatDistribution(1e-1, 1e2, log=True),
        "estimator__epsilon": FloatDistribution(1e-1, 1e3, log=True),
        "estimator__gamma": FloatDistribution(1e-3, 1e2, log=True),
    }

    model_name = "rbf-SVR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    # GaussianProcessRegressor (rbf)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn import gaussian_process

    rbf_kernel = gaussian_process.kernels.RBF(
        length_scale=1, length_scale_bounds=(0.1, 10)
    )
    estimator = GaussianProcessRegressor(kernel=rbf_kernel)
    estimator = MultiOutputRegressor(estimator)

    param_range = {"estimator__alpha": {"suggest_type": "log", "range": [1e-2, 1e2]}}
    param_grid = {}

    param_distributions = {
        "estimator__alpha": FloatDistribution(1e-2, 1e2, log=True),
    }
    model_name = "rbf-GPR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # GaussianProcessRegressor (mixed)
    mixed_kernel = (
        gaussian_process.kernels.ConstantKernel(
            constant_value=1, constant_value_bounds=(1e-2, 1e2)
        )
        * gaussian_process.kernels.RBF(length_scale=1, length_scale_bounds=(0.1, 10))
    ) + gaussian_process.kernels.WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-4, 1)
    )
    estimator = sklearn.gaussian_process.GaussianProcessRegressor(kernel=mixed_kernel)
    estimator = MultiOutputRegressor(estimator)

    param_range = {"estimator__alpha": {"suggest_type": "log", "range": [1e-2, 1e2]}}
    param_grid = {}
    param_distributions = {
        "estimator__alpha": FloatDistribution(1e-2, 1e2, log=True),
    }

    model_name = "mixed-kernel-GPR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # GaussianProcessRegressor (tune_kernel)
    kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
    ]
    estimator = sklearn.gaussian_process.GaussianProcessRegressor()
    estimator = MultiOutputRegressor(estimator)

    param_range = {"estimator__alpha": {"suggest_type": "log", "range": [1e-2, 1e2]}}
    param_grid = {}
    param_distributions = {
        "estimator__alpha": FloatDistribution(1e-15, 1e1, log=True),
        "estimator__kernel": CategoricalDistribution(choices=kernels),
    }

    model_name = "tuned-kernel-GPR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Kernel"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    # DecisionTreeRegressor
    estimator = sklearn.tree.DecisionTreeRegressor()
    kernels = []
    param_range = {"max_depth": {"suggest_type": "int", "range": [5, 13]}}
    param_distributions = {
        "max_depth": IntDistribution(5, 13, log=False),
    }
    param_grid = {}
    model_name = "DTR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # RandomForestRegressor
    estimator = sklearn.ensemble.RandomForestRegressor(
        n_jobs=-1, max_depth=5, n_estimators=15
    )
    param_range = {}
    param_grid = {}
    param_distributions = {}

    model_name = "mini-RFR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # RandomForestRegressor
    estimator = sklearn.ensemble.RandomForestRegressor(n_jobs=-1)
    param_range = {}
    param_distributions = {}
    param_grid = {}
    model_name = "default-RFR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # RandomForestRegressor
    estimator = sklearn.ensemble.RandomForestRegressor(n_jobs=-1)
    param_range = {
        "max_depth": {"suggest_type": "int", "range": [3, 13]},
        "n_estimators": {"suggest_type": "int", "range": [5, 200]},
    }
    param_distributions = {
        "max_depth": IntDistribution(3, 13),
        "n_estimators": IntDistribution(5, 200),
    }
    param_grid = {}
    model_name = "RFR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # ExtraTreesRegressor
    estimator = sklearn.ensemble.ExtraTreesRegressor(n_jobs=-1)
    param_range = {}
    param_distributions = {}

    param_grid = {}
    model_name = "default-ETR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 1
    models.append(estimator)

    # ExtraTreesRegressor
    estimator = sklearn.ensemble.ExtraTreesRegressor(n_jobs=-1)
    param_range = {
        "max_depth": {"suggest_type": "int", "range": [3, 13]},
        "n_estimators": {"suggest_type": "int", "range": [5, 200]},
        "min_samples_split": {"suggest_type": "int", "range": [2, 6]},
    }
    param_distributions = {
        "max_depth": IntDistribution(3, 13),
        "n_estimators": IntDistribution(5, 200),
        "min_samples_split": IntDistribution(2, 6),
    }

    param_grid = {}
    model_name = "ETR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    # ExtraTreesRegressor
    estimator = sklearn.ensemble.ExtraTreesRegressor(bootstrap=True, n_jobs=-1)
    param_range = {
        "max_depth": {"suggest_type": "int", "range": [3, 13]},
        "n_estimators": {"suggest_type": "int", "range": [5, 200]},
        "min_samples_split": {"suggest_type": "int", "range": [2, 6]},
    }
    param_distributions = {
        "max_depth": IntDistribution(3, 13),
        "n_estimators": IntDistribution(5, 200),
        "min_samples_split": IntDistribution(2, 6),
    }
    param_grid = {}
    model_name = "bootstrap-ETR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    # AdaBoostRegressor with ExtraTreeRegressor
    estimator = sklearn.ensemble.AdaBoostRegressor(sklearn.tree.ExtraTreeRegressor())
    param_range = {}
    param_distributions = {
        "learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
        "n_estimators": IntDistribution(5, 200),
    }
    param_grid = {}
    model_name = "Ada-ETR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Tree"
    estimator.shap = True
    estimator.library = "sklearn"
    estimator.n_trials = 50
    models.append(estimator)

    try:
        import xgboost
        # XGBoost Regressor
        estimator = xgboost.XGBRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_distributions = {}
        param_grid = {}
        model_name = "default-XGB"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "xgboost"
        estimator.n_trials = 1
        models.append(estimator)

        # XGBoost Regressor
        estimator = xgboost.XGBRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {
            "estimator__min_child_weight": {"suggest_type": "uniform", "range": [0.1, 10]},
            "estimator__subsample": {"suggest_type": "uniform", "range": [0, 1.0]},
            "estimator__learning_rate": {
                "suggest_type": "loguniform",
                "range": [1e-3, 5e-1],
            },
            "estimator__max_depth": {"suggest_type": "int", "range": [3, 15]},
            "estimator__n_estimators": {"suggest_type": "int", "range": [3, 300]},
        }

        param_distributions = {
            "estimator__min_child_weight": FloatDistribution(0.1, 10),
            "estimator__subsample": FloatDistribution(0, 1.0),
            "estimator__learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__n_estimators": IntDistribution(3, 300, log=True),
        }
        param_grid = {}
        model_name = "XGB"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "xgboost"
        estimator.n_trials = 50
        models.append(estimator)

        # XGBoost Regressor - early stopping
        estimator = xgboost.XGBRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {
            "estimator__min_child_weight": {"suggest_type": "uniform", "range": [0.1, 10]},
            "estimator__subsample": {"suggest_type": "uniform", "range": [0, 1.0]},
            "estimator__learning_rate": {
                "suggest_type": "loguniform",
                "range": [1e-3, 5e-1],
            },
            "estimator__max_depth": {"suggest_type": "int", "range": [3, 15]},
            "estimator__n_estimators": {"suggest_type": "int", "range": [3, 300]},
        }

        param_distributions = {
            "estimator__min_child_weight": FloatDistribution(0.1, 10),
            "estimator__subsample": FloatDistribution(0, 1.0),
            "estimator__learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__n_estimators": IntDistribution(3, 300, log=True),
        }
        param_grid = {}
        fit_params = {"early_stopping_rounds": 10}
        model_name = "earlystopping-XGB"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "xgboost"
        estimator.n_trials = 50
        models.append(estimator)

        # XGBoost Regressor - regularized
        estimator = xgboost.XGBRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}

        param_distributions = {
            "estimator__min_child_weight": FloatDistribution(0.1, 10),
            "estimator__subsample": FloatDistribution(0, 1.0),
            "estimator__learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__n_estimators": IntDistribution(3, 300, log=True),
            "estimator__alpha": FloatDistribution(1e-5, 5e-1, log=True),
            "estimator__lambda": FloatDistribution(1e-5, 5e-1, log=True),
        }
        param_grid = {}
        model_name = "regularized-XGB"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "xgboost"
        estimator.n_trials = 50
        models.append(estimator)


        # XGBoost Regressor - histogram
        estimator = xgboost.XGBRegressor(tree_method="hist", n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_distributions = {
            "estimator__min_child_weight": FloatDistribution(0.1, 10),
            "estimator__subsample": FloatDistribution(0, 1.0),
            "estimator__learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__n_estimators": IntDistribution(3, 300, log=True),
            "estimator__alpha": FloatDistribution(1e-5, 5e-1, log=True),
            "estimator__lambda": FloatDistribution(1e-5, 5e-1, log=True),
        }
        param_grid = {}
        model_name = "histogram-XGB"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "xgboost"
        estimator.n_trials = 50
        models.append(estimator)
    except:
        pass

    try:
        import lightgbm
        # lightgbm
        estimator = lightgbm.LGBMRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}

        param_distributions = {}
        param_grid = {}
        model_name = "default-lightgbm"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "lgbm"
        estimator.n_trials = 1
        models.append(estimator)

        # lightgbm
        estimator = lightgbm.LGBMRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {
            "estimator__num_leaves": {"suggest_type": "int", "range": [30, 100]},
            "estimator__feature_fraction": {
                "suggest_type": "uniform",
                "range": [0.05, 0.95],
            },
            "estimator__bagging_fraction": {"suggest_type": "uniform", "range": [0.8, 1.0]},
            "estimator__max_depth": {"suggest_type": "int", "range": [3, 15]},
            "estimator__min_split_gain": {"suggest_type": "log", "range": [1e-3, 1e-1]},
            "estimator__min_child_weight": {"suggest_type": "int", "range": [5, 20]},
        }

        param_distributions = {
            "estimator__num_leaves": IntDistribution(30, 100),
            "estimator__feature_fraction": FloatDistribution(0.05, 0.95),
            "estimator__bagging_fraction": FloatDistribution(0.8, 1.0),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__min_split_gain": FloatDistribution(1e-3, 1e-1, log=True),
            "estimator__min_child_weight": IntDistribution(5, 20),
        }
        param_grid = {}
        model_name = "lightgbm"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "lgbm"
        estimator.n_trials = 50
        models.append(estimator)

        # lightgbm - regularized
        estimator = lightgbm.LGBMRegressor(n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}

        param_distributions = {
            "estimator__num_leaves": IntDistribution(30, 100),
            "estimator__feature_fraction": FloatDistribution(0.05, 0.95),
            "estimator__bagging_fraction": FloatDistribution(0.8, 1.0),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__min_split_gain": FloatDistribution(1e-3, 1e-1, log=True),
            "estimator__min_child_weight": IntDistribution(5, 20),
            "estimator__min_child_weight": IntDistribution(5, 20),
            "estimator__reg_alpha": FloatDistribution(1e-6, 5e-1, log=True),
            "estimator__reg_lambda": FloatDistribution(1e-6, 5e-1, log=True),
        }
        param_grid = {}
        model_name = "regularized--lgbm"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = True
        estimator.library = "lightgbm"
        estimator.n_trials = 50
        models.append(estimator)
    except:
        pass

    try:
        from ngboost.ngboost import NGBoost
        from ngboost.learners import default_tree_learner
        from ngboost.scores import MLE
        from ngboost.distns import Normal, LogNormal

        # NGBoost
        estimator = NGBoost(
            Base=default_tree_learner, Dist=Normal, natural_gradient=True, verbose=False
        )
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "default-NGBoost"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = False
        estimator.library = "ngboost"
        models.append(estimator)

        # NGBoost
        estimator = NGBoost(
            Base=default_tree_learner, Dist=Normal, natural_gradient=True, verbose=False
        )
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__n_estimators": IntDistribution(5, 500, log=True),
            "estimator__learning_rate": FloatDistribution(1e-5, 1e-1, log=True),
        }
        model_name = "NGBoost"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = False
        estimator.library = "ngboost"
        models.append(estimator)

    except:
        pass

    try:
        from lineartree import LinearTreeRegressor, LinearForestRegressor, LinearBoostRegressor
        # ref. https://github.com/cerlymarco/linear-tree
        # LinearTreeRegressor
        estimator = LinearTreeRegressor(base_estimator=LinearRegression(), max_depth=1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "1-depth-LinearTree"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # LinearTreeRegressor
        estimator = LinearTreeRegressor(base_estimator=LinearRegression())
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__max_depth": IntDistribution(1, 15),
            "estimator__min_samples_split": IntDistribution(6, 10),
        }
        model_name = "LinearTree"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # LinearTreeRegressor
        estimator = LinearTreeRegressor(base_estimator=sklearn.linear_model.Ridge())
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__max_depth": IntDistribution(1, 15),
            "estimator__min_samples_split": IntDistribution(6, 10),
            "estimator__base_estimator__alpha": FloatDistribution(1e-2, 1e2, log=True),
        }
        model_name = "RidgeTree"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # LinearForestRegressor
        estimator = LinearForestRegressor(base_estimator=LinearRegression(), n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__n_estimators": IntDistribution(5, 300),
            "estimator__max_depth": IntDistribution(1, 15),
        }

        model_name = "LinearForest"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # LinearForestRegressor
        estimator = LinearForestRegressor(
            base_estimator=LinearRegression(), n_jobs=-1, max_depth=3
        )
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__n_estimators": IntDistribution(5, 300, log=True),
        }

        model_name = "3-depth_LinearForest"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # RidgeForesttRegressor
        estimator = LinearForestRegressor(
            base_estimator=sklearn.linear_model.Ridge(), n_jobs=-1
        )
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__n_estimators": IntDistribution(5, 300, log=True),
            "estimator__max_depth": IntDistribution(1, 15),
            "estimator__base_estimator__alpha": FloatDistribution(1e-2, 1e2, log=True),
        }
        model_name = "RidgeForest"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)

        # RidgeBoostRegressor
        estimator = LinearBoostRegressor(
            base_estimator=sklearn.linear_model.Ridge())
        estimator = MultiOutputRegressor(estimator)

        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__n_estimators": IntDistribution(5, 300, log=True),
            "estimator__max_depth": IntDistribution(1, 15),
            "estimator__base_estimator__alpha": FloatDistribution(1e-5, 5e-1, log=True),
        }
        model_name = "RidgeBoost"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "LinearTree"
        estimator.shap = False
        estimator.library = "lineartree"
        models.append(estimator)
    except:
        pass

    try:
        from sklearn_expansion.linear_model import LinearTreeSHAPRegressor

        # LinearTreeSHAPRegressor(RFR)
        estimator = LinearTreeSHAPRegressor(
            base_estimator=sklearn.ensemble.RandomForestRegressor()
        )
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "default_LinearTree_RFR_SHAPRegressor"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "SHAP"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)

        # LinearTreeSHAPRegressor(RFR - HuberRegressor)
        estimator = LinearTreeSHAPRegressor(
            estimator=sklearn.ensemble.RandomForestRegressor(),
            shapvalue_estimator=LinearTreeRegressor(
                base_estimator=sklearn.linear_model.HuberRegressor(), max_depth=1
            ),
        )
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__estimator__n_estimators": IntDistribution(5, 300, log=True),
            "estimator__shapvalue_estimator__estimator__epsilon": FloatDistribution(
                1.0001, 2.0, log=True
            ),
            "estimator__shapvalue_estimator__max_depth": IntDistribution(1, 6),
        }

        model_name = "LinearTree_RFR_SHAPRegressor"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "SHAP"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)

        # LinearTreeSHAPRegressor(LGBM - HuberRegressor)
        estimator = LinearTreeSHAPRegressor(
            estimator=lightgbm.LGBMRegressor(n_jobs=-1),
            shapvalue_estimator=LinearTreeRegressor(
                base_estimator=sklearn.linear_model.HuberRegressor(), max_depth=1
            ),
        )
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__shapvalue_estimator__estimator__epsilon": FloatDistribution(
                1.0001, 2.0
            ),
            "estimator__shapvalue_estimator__max_depth": IntDistribution(1, 6),
        }

        model_name = "default_LinearTree_LGBM_SHAPRegressor"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "SHAP"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)

        # LinearSHAPRegressor(LGBM - HuberRegressor)
        estimator = LinearTreeSHAPRegressor(
            estimator=lightgbm.LGBMRegressor(n_jobs=-1),
            shapvalue_estimator=sklearn.linear_model.HuberRegressor(),
        )
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {
            "estimator__shapvalue_estimator__epsilon": FloatDistribution(1.0001, 2.0),
        }
        model_name = "Linear_LGBM_SHAPRegressor"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "SHAP"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)
    except:
        pass

    try:
        from bartpy.sklearnmodel import SklearnModel as BartRegressor

        estimator = BartRegressor()
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "BartRegressor"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)

    except:
        pass

    # GBart
    try:
        from gbart.modified_bartpy.sklearnmodel import SklearnModel as GBartRegressor

        estimator = GBartRegressor()  # Use default parameters
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "GBartRegressor"
        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)
    except:
        pass

    try:
        # RuleFit
        from rulefit import RuleFit

        estimator = RuleFit(rfmode="regress", n_jobs=-1)
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "RuleFit"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.type = "others"
        estimator.shap = False
        estimator.library = "others"
        models.append(estimator)
    except:
        pass

    # HuberRegressor
    estimator = sklearn.linear_model.HuberRegressor()
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {
        "estimator__epsilon": FloatDistribution(1.0, 10.0),
        "estimator__alpha": FloatDistribution(1e-7, 0.1, log=True),
    }
    model_name = "HuberRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "Robust"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # RANSACRegressor
    estimator = sklearn.linear_model.RANSACRegressor(
        sklearn.linear_model.LinearRegression(), min_samples=0.7
    )
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "RANSAC_OLS"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "Robust"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # RANSACRegressor
    estimator = sklearn.linear_model.RANSACRegressor(
        sklearn.ensemble.RandomForestRegressor(n_jobs=-1), min_samples=0.7
    )
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "RANSAC_RFR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "Robust"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # RANSACRegressor
    estimator = sklearn.linear_model.RANSACRegressor(
        sklearn.ensemble.ExtraTreesRegressor(), min_samples=0.7
    )
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "RANSAC_ETR"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "Robust"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # PassiveAggressiveRegressor
    estimator = sklearn.linear_model.PassiveAggressiveRegressor()
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_grid = {}
    param_distributions = {
        "estimator__C": FloatDistribution(0.01, 10.0, log=True),
    }
    model_name = "PassiveAggressiveRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "others"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # LassoLarsAIC
    estimator = sklearn.linear_model.LassoLarsIC(criterion="aic")
    param_range = {}
    param_distributions = {}

    param_grid = {}
    model_name = "LassoLarsAIC"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # LassoLarsBIC
    estimator = sklearn.linear_model.LassoLarsIC(criterion="bic")
    param_range = {}
    param_distributions = {}

    param_grid = {}
    model_name = "LassoLarsBIC"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.type = "Linear"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # PoissonRegressor
    estimator = sklearn.linear_model.PoissonRegressor()
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = {"estimator__alpha" : FloatDistribution(1e-6, 1e0, log=True)}

    param_grid = {}
    model_name = "PoissonRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor()
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = {"estimator__power" : FloatDistribution(1.0, 5.0, log=False),
                            "estimator__alpha" : FloatDistribution(1e-5, 1e1, log=True),
                    }

    param_grid = {}
    model_name = "tune_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor(power=0)
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = { "estimator__alpha" : FloatDistribution(1e-5, 5e0, log=True),
                    }

    param_grid = {}
    model_name = "normal_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor(power=1)
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = {"estimator__alpha" : FloatDistribution(1e-5, 5e0, log=True),
                    }

    param_grid = {}
    model_name = "Poisson_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor(power=1.5)
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = {"estimator__alpha" : FloatDistribution(1e-5, 5e0, log=True),
                    }

    param_grid = {}
    model_name = "Compound_Poisson_Gamma_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = False
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor(power=2)
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = { "estimator__alpha" : FloatDistribution(1e-5, 5e0, log=True),
                    }

    param_grid = {}
    model_name = "Gamma_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    # TweedieRegressor
    estimator = sklearn.linear_model.TweedieRegressor(power=3)
    estimator = MultiOutputRegressor(estimator)
    param_range = {}
    param_distributions = { "estimator__alpha" : FloatDistribution(1e-5, 5e0, log=True),
                    }

    param_grid = {}
    model_name = "InverseGamma_TweedieRegressor"

    estimator.model_name = model_name
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.type = "GLM"
    estimator.shap = True
    estimator.library = "sklearn"
    models.append(estimator)

    try:
        # Regularized Greedy Forests(RGF)
        from rgf.sklearn import RGFRegressor
        estimator = RGFRegressor()
        estimator = MultiOutputRegressor(estimator)
        param_range = {}
        param_distributions = {"estimator__max_leaf" : IntDistribution(100, 1000, log=True),
                                "estimator__l2" : FloatDistribution(1e-1, 2.0, log=True),
                                "estimator__learning_rate" : FloatDistribution(0.1, 0.9, log=False),
                        }
        param_grid = {}
        model_name = "RGF"

        estimator.model_name = model_name
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.default = False
        estimator.step_name = "estimator"
        estimator.type = "Tree"
        estimator.shap = False
        estimator.library = "RGF"
        models.append(estimator)
    except:
        pass

    return models

def load_classification_models():
    models = []

    # LogisticRegression
    estimator = sklearn.linear_model.LogisticRegression(class_weight="balanced")
    param_range = {}
    param_grid = {}
    param_distributions = {}

    model_name = "LogisticRegression"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # RidgeClassifier
    estimator = sklearn.linear_model.RidgeClassifier(class_weight="balanced")
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "Ridge"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # SGDClassifier
    estimator = sklearn.linear_model.SGDClassifier(class_weight="balanced")
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "SGD"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # LassoLars
    estimator = sklearn.linear_model.LassoLars(alpha=0.1, normalize=False)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "LassoLars"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # Stochastic Gradient Descent
    estimator = sklearn.linear_model.SGDClassifier(class_weight="balanced")
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "SGDClassifier"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # PassiveAggressiveClassifier
    estimator = sklearn.linear_model.PassiveAggressiveClassifier(
        class_weight="balanced"
    )
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "PassiveAggressiveClassifier"
    estimator.model_name = model_name
    estimator.default = False
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # SGDOneClassSVM
    # New in sklearn v1.0
    if sklearn.__version__ >= "1.0":
        from sklearn import linear_model

        estimator = linear_model.SGDOneClassSVM()
        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "SGDOneClassSVM"
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.shap = True
        models.append(estimator)

    # Linear-SVC
    estimator = sklearn.svm.LinearSVC(class_weight="balanced")
    estimator = MultiOutputClassifier(estimator)

    param_range = {
        "estimator__C": {"suggest_type": "log", "range": [1e-1, 1e2]},
        "estimator__tol": {"suggest_type": "log", "range": [1e-6, 1e-2]},
    }
    param_grid = {}
    param_distributions = {
        "estimator__C": FloatDistribution(1e-1, 1e2, log=True),
        "estimator__tol": FloatDistribution(1e-6, 1e-2, log=True),
    }
    model_name = "LinearSVC"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # SVC (rbf)
    estimator = sklearn.svm.SVC(kernel="rbf", class_weight="balanced")
    estimator = MultiOutputClassifier(estimator)
    param_range = {
        "estimator__C": {"suggest_type": "log", "range": [1e-2, 5e0]},
        "estimator__tol": {"suggest_type": "log", "range": [1e-6, 1e-2]},
    }
    param_distributions = {
        "estimator__C": FloatDistribution(1e-1, 1e2, log=True),
        "estimator__tol": FloatDistribution(1e-6, 1e-2, log=True),
    }
    param_grid = {}
    model_name = "SVC"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # KNeighborsClassifier
    estimator = sklearn.neighbors.KNeighborsClassifier()
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "KNeighbors"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # GaussianProcessClassifier (rbf)
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn import gaussian_process

    rbf_kernel = gaussian_process.kernels.RBF(
        length_scale=1, length_scale_bounds=(0.1, 10)
    )
    estimator = GaussianProcessClassifier(kernel=rbf_kernel)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "GPC_rbf"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # GaussianProcessClassifier (mixed)
    mixed_kernel = (
        gaussian_process.kernels.ConstantKernel(
            constant_value=1, constant_value_bounds=(1e-2, 1e2)
        )
        * gaussian_process.kernels.RBF(length_scale=1, length_scale_bounds=(0.1, 10))
    ) + gaussian_process.kernels.WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-4, 1)
    )
    estimator = sklearn.gaussian_process.GaussianProcessClassifier(kernel=mixed_kernel)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "GPC_mixed"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # DecisionTreeClassifier
    estimator = sklearn.tree.DecisionTreeClassifier(class_weight="balanced")
    param_range = {"max_depth": {"suggest_type": "int", "range": [5, 13]}}
    param_grid = {}
    param_distributions = {
        "max_depth": IntDistribution(5, 13),
    }

    model_name = "DTC"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    # RandomForestClassifier
    estimator = sklearn.ensemble.RandomForestClassifier(class_weight="balanced")
    param_range = {
        "max_depth": {"suggest_type": "int", "range": [3, 13]},
        "n_estimators": {"suggest_type": "int", "range": [5, 200]},
    }
    param_grid = {}
    param_distributions = {
        "max_depth": IntDistribution(3, 13),
        "n_estimators": IntDistribution(5, 200, log=True),
    }
    model_name = "RFC"
    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.shap = True
    models.append(estimator)

    try:
        import xgboost
        # XGBClassifier
        estimator = xgboost.XGBClassifier()
        estimator = MultiOutputClassifier(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "default_XGB"
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.shap = True
        models.append(estimator)

        # XGBClassifier
        estimator = xgboost.XGBClassifier()
        estimator = MultiOutputClassifier(estimator)
        param_range = {
            "estimator__min_child_weight": {"suggest_type": "uniform", "range": [0.1, 10]},
            "estimator__subsample": {"suggest_type": "uniform", "range": [0, 1.0]},
            "estimator__learning_rate": {
                "suggest_type": "loguniform",
                "range": [1e-3, 5e-1],
            },
            "estimator__max_depth": {"suggest_type": "int", "range": [3, 15]},
            "estimator__n_estimators": {"suggest_type": "int", "range": [3, 300]},
        }
        param_grid = {}
        param_distributions = {
            "estimator__min_child_weight": FloatDistribution(0.1, 10),
            "estimator__subsample": FloatDistribution(0, 1.0),
            "estimator__learning_rate": FloatDistribution(1e-3, 5e-1, log=True),
            "estimator__max_depth": IntDistribution(3, 15),
            "estimator__n_estimators": IntDistribution(3, 300, log=True),
        }
        model_name = "XGB"
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.shap = True
        models.append(estimator)
    except:
        pass

    try:
        import lightgbm
        # LGBMClassifier
        estimator = lightgbm.LGBMClassifier()
        estimator = MultiOutputClassifier(estimator)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "default_LGBM"
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.shap = True
        models.append(estimator)

        # LGBMClassifier
        estimator = lightgbm.LGBMClassifier()
        estimator = MultiOutputClassifier(estimator)
        param_range = {
            "estimator__num_leaves": {"suggest_type": "int", "range": [3, 100]},
            "estimator__feature_fraction": {
                "suggest_type": "uniform",
                "range": [0.05, 0.95],
            },
            "estimator__bagging_fraction": {"suggest_type": "uniform", "range": [0.8, 1.0]},
            "estimator__max_depth ": {"suggest_type": "int", "range": [3, 100]},
            "estimator__min_child_weight": {"suggest_type": "int", "range": [5, 20]},
        }
        param_grid = {}
        param_distributions = {
            "estimator__num_leaves": IntDistribution(30, 100),
            "estimator__feature_fraction": FloatDistribution(0.05, 0.95),
            "estimator__bagging_fraction": FloatDistribution(0.8, 1.0),
            "estimator__max_depth": IntDistribution(3, 100, log=True),
            "estimator__min_split_gain": FloatDistribution(1e-3, 1e-1, log=True),
            "estimator__min_child_weight": IntDistribution(5, 20),
        }

        model_name = "LGBM"
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        estimator.shap = True
        models.append(estimator)
    except:
        pass

    return models

def load_clustering_models():
    from sklearn.metrics import make_scorer
    from sklearn_expansion.metrics import k3nerror

    k3n_error = make_scorer(k3nerror, greater_is_better=False)
    models = []

    # PCA
    estimator = sklearn.decomposition.PCA()
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "PCA"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    models.append(estimator)

    # KernelPCA
    kernel = "rbf"
    estimator = sklearn.decomposition.KernelPCA(
        n_components=2, kernel=kernel, n_jobs=-1
    )
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "KernelPCA_{}".format(kernel)

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    models.append(estimator)

    # tSNE
    estimator = TSNE(n_components=2, perplexity=30, n_jobs=-1)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "tSNE"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    models.append(estimator)

    # tSNE-tuning
    estimator = TSNE(n_components=2, n_jobs=-1)
    param_range = {}  # {"perplexity": {"suggest_type": "uniform", "range": [2, 75]}}
    param_grid = {}
    param_distributions = {"perplexity": {"suggest_type": "uniform", "range": [2, 75]}}
    model_name = "tSNE-tune"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    models.append(estimator)

    try:
        from sklearn_expansion.manifold import QSNE

        # qSNE 1.5
        q = 1.5
        estimator = QSNE(n_components=2, q=q, verbose=0, n_jobs=-1)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "qSNE_{}".format(q)
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        models.append(estimator)

        # qSNE 2.5
        q = 2.5
        estimator = QSNE(n_components=2, q=q, verbose=0, n_jobs=-1)
        param_range = {}
        param_grid = {}
        param_distributions = {}

        model_name = "qSNE_{}".format(q)
        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        models.append(estimator)

        # qSNE
        estimator = QSNE(n_components=2, verbose=0, n_jobs=-1)
        param_range = {}  # {"q": {"suggest_type": "uniform", "range": [1, 3]}}
        param_grid = {}
        param_distributions = {"q": {"suggest_type": "uniform", "range": [1, 3]}}

        model_name = "qSNE_{}".format("tune")

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        models.append(estimator)

    except:
        pass

    try:
        import umap

        # UMAP
        estimator = umap.UMAP(n_jobs=-1)
        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "UMAP"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        models.append(estimator)

        # UMAP
        estimator = umap.UMAP(n_jobs=-1)
        param_range = {}
        param_grid = {}
        param_distributions = {
            "n_neighbors": {"suggest_type": "int", "range": [5, 50]},
            "min_dist": {"suggest_type": "log", "range": [0.001, 0.5]},
        }
        model_name = "UMAP-tune"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        # models.append(estimator)

        # ParamtericUMAP
        from umap.parametric_umap import ParametricUMAP

        estimator = ParametricUMAP(n_jobs=-1)
        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "ParametricUMAP"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        # models.append(estimator)

        # DensMAP
        sig_ = signature(umap.UMAP)
        if "densmap" in sig_.parameters:
            estimator = umap.UMAP(densmap=True)
            param_range = {}
            param_grid = {}
            param_distributions = {}
            model_name = "DensMAP"

            estimator.model_name = model_name
            estimator.default = True
            estimator.step_name = "estimator"
            estimator.param_range = param_range
            estimator.param_grid = param_grid
            estimator.param_distributions = param_distributions
            # models.append(estimator)
        else:
            pass

        # PCA+UMAP
        pca_ = sklearn.decomposition.PCA()
        umap_ = umap.UMAP(n_jobs=-1)

        estimator = Pipeline([("pca", pca_), ("UMAP", umap_)])

        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "PCA_UMAP"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        # models.append(estimator)

        # pairplot
        estimator = IdentityMapping()
        param_range = {}
        param_grid = {}
        param_distributions = {}
        model_name = "pairplot"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        models.append(estimator)
    except:
        pass



    # DBSCAN
    estimator = sklearn.cluster.DBSCAN(n_jobs=-1)
    param_range = {}
    param_grid = {}
    param_distributions = {}
    model_name = "DBSCAN"

    estimator.model_name = model_name
    estimator.default = True
    estimator.step_name = "estimator"
    estimator.param_range = param_range
    estimator.param_grid = param_grid
    estimator.param_distributions = param_distributions
    estimator.library = "sklearn"

    # models.append(estimator)

    try:
        from dcekit.generative_model import GTM

        # GTM
        # https://github.com/hkaneko1985/dcekit/blob/master/demo_gtm.py
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambda_in_em_algorithm = 0.001
        number_of_iterations = 300
        display_flag = False

        estimator = GTM(
            shape_of_map,
            shape_of_rbf_centers,
            variance_of_rbfs,
            lambda_in_em_algorithm,
            number_of_iterations,
            display_flag,
        )

        sig_ = signature(GTM)
        params_ = sig_.parameters
        if "set_parmas" not in params_.keys():
            # shape_of_rbf_centers must be a divisor of shape_of_map.

            param_range = {
                "shape_of_map": {"suggest_type": "choice", "range": [[5, 10, 20]]},
                "shape_of_rbf_centers": {"suggest_type": "choice", "range": [[5, 10]]},
                "variance_of_rbfs": {"suggest_type": "uniform", "range": [2, 6]},
                "lambda_in_em_algorithm": {"suggest_type": "log", "range": [1e-7, 1e-2]},
            }
            param_grid = {}
            param_distributions = {}

        else:
            print(f'{estimator} class does not support "set_params".')
            print(f"Hyperparameter search turned off.")
            param_range = {}  # not tune
            param_grid = {}
            param_distributions = {}

        param_range = {}  # no-tune

        model_name = "GTM"

        estimator.model_name = model_name
        estimator.default = True
        estimator.step_name = "estimator"
        estimator.param_range = param_range
        estimator.param_grid = param_grid
        estimator.param_distributions = param_distributions
        # models.append(estimator)
    except:
        pass


    return models


def load_all_models(ml_type="regression"):

    if ml_type == "regression":
        models = load_regression_models()
    elif ml_type == "classification":
        models = load_classification_models()
    elif ml_type == "clustering":
        models = load_clustering_models()
    else:
        models = []

    return models
