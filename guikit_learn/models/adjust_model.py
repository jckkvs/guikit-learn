
import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from optuna.integration import OptunaSearchCV

def adjust_hyperparameter_model(model, X_df, y_df):
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    X_df : _type_
        _description_
    y_df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n_samples = X_df.shape[0]
    n_X_features = X_df.shape[1]

    # PLS n_components の探索範囲の最大値を説明変数の数の範囲内に変更する
    if model.__class__ == Pipeline:
        target_model = model["estimator"]
    else:
        target_model = model

    attribute_names = ["n_components", "estimator__n_components"]
    for attribute_name in attribute_names:
        if hasattr(target_model, attribute_name) == True:
            # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
            param_distributions = getattr(
                target_model, "param_distributions", {}
            )
            # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
            n_components_distribution = param_distributions.get(
                attribute_name, None
            )
            disrtibution_high = getattr(n_components_distribution, "high", 0)
            # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
            if disrtibution_high > n_X_features:
                # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
                setattr(n_components_distribution, "high", n_X_features)
            # supervised_estimatorの引数が説明変数の数を超えている場合、引数の値を変更する
            if getattr(target_model, attribute_name) > n_X_features:
                setattr(target_model, attribute_name, n_X_features)

    if model.__class__ == Pipeline:
        target_model = model["selector"]
    else:
        target_model = model

    attribute_names = ["n_features_to_select", "k_features"]
    # 変数選択手法の変数選択の最大値を説明変数の数に変更する
    # n_features_to_Selectを引数に持つselector: sklearn.SequentialFeatureSelector, ReliefFなど
    # k_featuresを引数に持つselector : mlxtend.SequentialFeatureSelector

    for attribute_name in attribute_names:
        if hasattr(target_model, attribute_name) == True:
            # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
            param_distributions = getattr(target_model, "param_distributions", {})
            # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
            n_features_distribution = param_distributions.get(
                attribute_name, None
            )
            if n_features_distribution is not None:
                disrtibution_high = getattr(n_features_distribution, "high", 0)
                # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
                if disrtibution_high > n_X_features:
                    # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
                    setattr(n_features_distribution, "high", n_X_features-1)

            # selectorの引数が説明変数の数を超えている場合、引数の値を変更する
            initial_attribute = getattr(target_model, attribute_name)
            if isinstance(initial_attribute, int):
                # selectorの引数が説明変数の数を超えている場合、引数の値を変更する
                if getattr(target_model, attribute_name) > n_X_features:
                    setattr(target_model, attribute_name, n_X_features-1)

    attribute_names = ["n_neighbors", "discrete_threshold"]
    # 変数選択手法の変数選択のn_neighborsを説明変数の数-1に変更する
    for attribute_name in attribute_names:
        if hasattr(target_model, attribute_name) == True:
            # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
            param_distributions = getattr(target_model, "param_distributions", {})
            # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
            n_features_distribution = param_distributions.get(
                attribute_name, None
            )
            if n_features_distribution is not None:
                disrtibution_high = getattr(n_features_distribution, "high", 0)
                # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
                if disrtibution_high > n_X_features:
                    # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
                    setattr(n_features_distribution, "high", n_X_features-1)

            initial_attribute = getattr(target_model, attribute_name)
            if isinstance(initial_attribute, int):
                # selectorの引数が説明変数の数を超えている場合、引数の値を変更する
                if getattr(target_model, attribute_name) > n_X_features:
                    setattr(target_model, attribute_name, n_X_features-1)

    # サンプル数が20以下の場合Borutaのpをランダムな相関係数の最大値に設定する
    # https://datachemeng.com/post-4235/
    if getattr(target_model, "model_name", "") == "Boruta":
        max_abs_corr = 0
        if n_samples <= 20:
            for i in range(1000):
                randomize_X = np.random.permutation(np.array(X_df))
                new_df = pd.concat([pd.DataFrame(randomize_X), y_df], axis=1)
                corr_df = new_df.corr().abs()
                each_max_abs_corr = max(corr_df.iloc[:, -1][:-1])
                max_abs_corr = max(max_abs_corr, each_max_abs_corr)

            perc = round(100 * (1 - max_abs_corr))

            print(f"サンプル数が非常に少ないため、Borutaのp値を{perc}に変更します")
            setattr(target_model, "perc", perc)

    return model

def adjust_hyperparameter_arg(model, X_df, y_df):

    n_samples = X_df.shape[0]
    n_X_features = X_df.shape[1]

    if model.__class__ in [GridSearchCV, OptunaSearchCV]:
        # model.estimator = adjust_hyperparameter_arg(model.estimator, X_df, y_df)
        adjust_hyperparameter_arg(model.estimator, X_df, y_df)
        return model
    elif model.__class__ == Pipeline:
        for each_step in model.steps:
            adjust_hyperparameter_arg(each_step[1], X_df, y_df)
        return  model
    elif hasattr(model, "estimator"):
        adjust_hyperparameter_arg(model.estimator, X_df, y_df)
        return model
    else:
        model = model

    # estimatorの引数を調整する
    # n_components の最大値はn_X_features
    attribute_names = ["n_components"]
    for attribute_name in attribute_names:
        if getattr(model, attribute_name, 0) > n_X_features:
            setattr(model, attribute_name, n_X_features)

    # "n_neighbors", "discrete_threshold" の最大値はn_X_features - 1
    attribute_names = ["n_neighbors", "discrete_threshold"]
    for attribute_name in attribute_names:
        if getattr(model, attribute_name, 0) > n_X_features-1:
            setattr(model, attribute_name, n_X_features-1)

    # 説明変数選択にBorutaを用いる場合で, サンプル数が少ない場合, p値を調整する
    if getattr(model, "model_name", "") == "Boruta":
        max_abs_corr = 0
        if n_samples <= 20:
            for i in range(1000):
                randomize_X = np.random.permutation(np.array(X_df))
                new_df = pd.concat([pd.DataFrame(randomize_X), y_df], axis=1)
                corr_df = new_df.corr().abs()
                each_max_abs_corr = max(corr_df.iloc[:, -1][:-1])
                max_abs_corr = max(max_abs_corr, each_max_abs_corr)
            perc = round(100 * (1 - max_abs_corr))
            print(f"サンプル数が非常に少ないため、Borutaのp値を{perc}に変更します")
            setattr(model, "perc", perc)

    return model

def adjust_hyperparameter_range(model, X_df, y_df):
    n_X_features = X_df.shape[1]

    if model.__class__ not in [GridSearchCV, OptunaSearchCV]:
        print("return model")
        return model

    attribute_names = ["n_components", "estimator__n_components", "selector__estimator__n_components"]
    for attribute_name in attribute_names:
        # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
        param_distributions = getattr(
            model, "param_distributions", {}
        )
        # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
        n_components_distribution = param_distributions.get(
            attribute_name, None
        )
        disrtibution_high = getattr(n_components_distribution, "high", 0)
        # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
        if disrtibution_high > n_X_features:
            # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
            setattr(n_components_distribution, "high", n_X_features)


    # 変数選択手法の変数選択の最大値を説明変数の数に変更する
    # n_features_to_Selectを引数に持つselector: sklearn.SequentialFeatureSelector, ReliefFなど
    # k_featuresを引数に持つselector : mlxtend.SequentialFeatureSelector
    attribute_names = ["n_features_to_select", "k_features",
                    "selector__n_features_to_select", "selector__k_features"]

    for attribute_name in attribute_names:
        # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
        param_distributions = getattr(model, "param_distributions", {})
        # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
        n_features_distribution = param_distributions.get(
            attribute_name, None
        )
        if n_features_distribution is not None:
            disrtibution_high = getattr(n_features_distribution, "high", 0)
            # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
            if disrtibution_high > n_X_features:
                # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
                setattr(n_features_distribution, "high", n_X_features-1)

    attribute_names = ["n_neighbors", "discrete_threshold",
                    "selector__n_neighbors", "selector__discrete_threshold",
                    ]
    # 変数選択手法の変数選択のn_neighborsを説明変数の数-1に変更する
    for attribute_name in attribute_names:
        # e.g. param_distributions = {"n_components": IntUniformDistribution(2, 100)}
        param_distributions = getattr(model, "param_distributions", {})
        # e.g. n_components_distribution = IntUniformDistribution(low=2, high=100)
        n_features_distribution = param_distributions.get(
            attribute_name, None
        )
        if n_features_distribution is not None:
            disrtibution_high = getattr(n_features_distribution, "high", 0)
            # ハイパーパラメータの最適化範囲が説明変数の数を超えている場合、最大値を変更する
            if disrtibution_high > n_X_features:
                # IntUniformDistributionのhighを設定しなおせば,supervised_estimatorにも反映される
                setattr(n_features_distribution, "high", n_X_features-1)

    return model

def adjust_hyperparameter_todata(model, X_df, y_df):

    # モデル自身の引数を変更する
    # model = adjust_hyperparameter_arg(model, X_df, y_df)
    adjust_hyperparameter_arg(model, X_df, y_df)

    # ハイパーパラメータの探索範囲を変更する
    # model = adjust_hyperparameter_range(model, X_df, y_df)
    adjust_hyperparameter_range(model, X_df, y_df)

    return model


if __name__ == "__main__":
    from sklearn.cross_decomposition import PLSRegression
    from optuna.distributions import (
        FloatDistribution,
        IntDistribution,
        CategoricalDistribution,
    )

    model = PLSRegression(n_components=30)
    X_df = pd.DataFrame(np.random.random(size=(400,10)))
    y_df = pd.DataFrame(np.random.random(size=(400,1)))
    print("simple model")
    #print(model)
    model = adjust_hyperparameter_todata(model, X_df, y_df)
    print(model)

    model = PLSRegression(n_components=30)
    model_distributions = {"n_components":IntDistribution(3,100)}
    opcv_model = OptunaSearchCV(model, model_distributions)
    print("opcv(model)")

    #print(opcv_model)
    opcv_model = adjust_hyperparameter_todata(opcv_model, X_df, y_df)
    print(opcv_model)

    from sklearn.feature_selection import SelectFromModel
    pipe = Pipeline([("selector", SelectFromModel(PLSRegression(n_components=30))),
                        ("estimator", (PLSRegression(n_components=30)))])

    print("pipeline")
    #print(pipe)
    pipe = adjust_hyperparameter_todata(pipe, X_df, y_df)
    print(pipe)

    pipe = Pipeline([("selector", SelectFromModel(PLSRegression(n_components=30))),
                        ("estimator", (PLSRegression(n_components=30)))])
    pipe_distributions = {"selector__estimator__n_components":IntDistribution(3,100),
                        "estimator__n_components":IntDistribution(3,100)}
    opcv_pipe = OptunaSearchCV(pipe, pipe_distributions)

    print("opcv(pipeline)")
    # print(opcv_pipe)
    opcv_pipe = adjust_hyperparameter_todata(opcv_pipe, X_df, y_df)
    print(opcv_pipe)

    try:
        import BorutaPy
    except:
        pass