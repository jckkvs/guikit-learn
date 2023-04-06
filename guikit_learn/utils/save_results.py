
# 標準ライブラリ
import os
import copy

# third-party
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# from this library
from .base import get_estimator

def save_results(
    model,
    save_path,
    *,
    save_name="",
    model_type=None,
    X_names=None,
    y_names=None,
    mask=None,
):

    if mask is None:
        selector, selector_name = get_estimator(
            model, model_type="selector", remove_multioutput=False
        )
        if hasattr(selector, "get_support") == True:
            mask = selector.get_support()
        elif hasattr(selector, "support_") == True:
            mask = selector.support_
        else:
            mask = [True] * len(X_names)
    else:
        mask = mask

    print("X_names")
    print(X_names)
    X_names = [
        X_name for X_name, each_mask in zip(X_names, mask) if each_mask == True
    ]
    mask = [True] * len(X_names)
    estimator, estimator_name = get_estimator(
        model, model_type=model_type, remove_multioutput=False
    )

    os.makedirs(save_path, exist_ok=True)

    # MultiOutputの場合は、estimatorをstripして再帰
    if estimator.__class__.__name__ in [
        "MultiOutputClassifier",
        "MultiOutputRegressor",
    ]:
        estimators = estimator.estimators_
        for each_estimator, y_name in zip(estimators, y_names):
            each_save_name = f"{save_name}_{y_name}"
            save_results(
                each_estimator,
                save_path,
                save_name=each_save_name,
                model_type=model_type,
                X_names=X_names,
                y_names=[y_name],
                mask=mask,
            )

    save_name = f"{save_name}_{estimator_name}"

    if hasattr(estimator, "feature_importances_") == True:
        save_feature_importances(
            estimator, save_path, save_name=save_name, X_names=X_names
        )

    if (hasattr(estimator, "coef_") == True) or (
        hasattr(estimator, "intercept_") == True
    ):
        save_coef_intercept(
            estimator,
            save_path,
            save_name=save_name,
            X_names=X_names,
            y_names=y_names,
        )

    if hasattr(estimator, "get_rules") == True:
        save_rules(
            estimator,
            save_path,
            save_name=save_name,
            X_names=X_names,
            y_names=y_names,
        )

    if estimator.__class__.__name__ in [
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    ]:
        save_name = f"{save_name}_{estimator_name}"
        save_tree(estimator, save_path, save_name=save_name)

    if estimator.__class__.__name__ in [
        "RandomForestClassifier",
        "RandomForestRegressor",
        #'XGBClassifier', 'XGBRegressor',
        #'LGBMClassifier', 'LGBMRegressor',
    ]:
        trees_path = save_path / "randomforest"
        for dt_idx, dtree in enumerate(estimator.estimators_):
            if dt_idx >= 10:
                break
            each_save_name = f"{save_name}_{estimator_name}_{dt_idx}"
            save_tree(dtree, trees_path, save_name=each_save_name)

def save_tree(estimator, save_path, *, save_name="", feature_names=None):
    save_path = save_path / "tree"
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(20, 20))
    sklearn.tree.plot_tree(
        estimator,
        max_depth=10,
        proportion=True,
        feature_names=feature_names,
        filled=True,
        precision=2,
        fontsize=12,
    )
    plt.savefig(
        save_path / "{}.pdf".format(save_name),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_path / "{}.eps".format(save_name),
        format="eps",
        bbox_inches="tight",
    )
    plt.close()

def save_feature_importances(
    estimator, save_path, *, save_name="", X_names=None
):
    save_path = save_path / "importance"
    os.makedirs(save_path, exist_ok=True)

    bottom_margin = 0.05 + 1.75 * 15 / 100
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"] = 0.125
    plt.rcParams["figure.subplot.bottom"] = bottom_margin  # 下余白 0.15
    plt.rcParams["figure.subplot.right"] = 0.95  # 右余白 0.05
    plt.rcParams["figure.subplot.top"] = 0.95  # 上余白 0.10
    plt.rcParams["font.size"] = 18

    if estimator.__class__.__name__ != "NGBoost":
        importances = pd.Series(estimator.feature_importances_)
        importances = np.array(importances)

        importances_df = pd.DataFrame(importances).T
        if len(importances_df.columns) == len(X_names):
            # 重要度の列数とX_namesの数が一致している場合
            # 説明変数選択等によって、変数の数が異なる場合もありうる
            # save_results関数中でget_supportやsupport_を用いて調整しているが
            # selectorがscikit-learnのモデルではない場合は不一致がありうる

            importances_df.columns = X_names
            label = X_names

        else:
            label = [idx for  idx, i in enumerate(importances)]


        importances_df.to_csv(
            save_path / f"{save_name}.csv", encoding="shift_jisx0213"
        )

        plt.figure(figsize=(7.8, 7.8))
        plt.bar(label, importances)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=24)
        plt.title(f"変数重要度 : {save_name}")
        plt.savefig(
            save_path / f"{save_name}.png", dpi=100, bbox_inches="tight"
        )
        plt.close()
    else:
        for idx, importance_type in enumerate(["loc", "scale"]):
            importances = pd.Series(estimator.feature_importances_[idx])
            importances = np.array(importances)

            label = X_names
            importances_df = pd.DataFrame(importances).T
            importances_df.columns = X_names
            importances_df.to_csv(
                save_path / f"{save_name}_{importance_type}.csv",
                encoding="shift_jisx0213",
            )

            plt.figure(figsize=(7.8, 7.8))
            plt.bar(label, importances)
            plt.xticks(rotation=90)
            plt.xticks(fontsize=24)
            plt.title(f"変数重要度 : {save_name}")
            plt.savefig(
                save_path / f"{save_name}_{importance_type}.png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()

def save_model(
    model, save_path: str, *, save_name: str = "", model_type: str = None
):
    """
    モデルをpickleファイルとして保存する

    Parameters
    ----------
    model : sklearn instance.
        scaler, feature_engineerings, selector, estimator, pipeline等
    save_path : str, Path object.
        保存フォルダ―名
    save_name : str.
        保存ファイル名
    model_type : str, None
        pipelineがmodelとして渡された場合に
        scaler, feature_engineerings, selector, estimatorのいずれを保存するかの指定
    """

    instance, instance_name = get_estimator(
        model, model_type=model_type, remove_multioutput=False
    )
    estimator, _ = get_estimator(
        model, model_type="estimator", remove_multioutput=False
    )
    _, raw_estimator_name = get_estimator(
        model, model_type="estimator", remove_multioutput=True
    )

    new_save_name = (
        f"{save_name}_{model_type}_{raw_estimator_name}_{instance_name}.pkl"
    )

    os.makedirs(save_path, exist_ok=True)
    if hasattr(estimator, "save"):
        # https://umap-learn.readthedocs.io/en/latest/parametric_umap.html
        if hasattr(instance, "save"):
            # each_model.save(save_path)
            pass
        else:
            print(f"{instance} cannot save and piclle.dump. Passed")
    else:
        with open(save_path / new_save_name, "wb") as f:
            cloudpickle.dump(instance, f)

def save_params(
    model,
    save_path: str,
    *,
    save_name: str = "",
    model_type: str = None,
):
    """
    モデルのパラメータをtxtファイルとして保存する

    Parameters
    ----------
    save_path : str, Path object.
        保存フォルダ―名
    save_name : str.
        保存ファイル名
    model_type : str.
        scaler, feature_engineerings, selector, estimator等
    """

    _, estimator_name = get_estimator(model, model_type="estimator")
    instance, instance_name = get_estimator(model, model_type=model_type)
    new_save_name = f"{save_name}_{model_type}_{estimator_name}_{instance_name}.txt"

    os.makedirs(save_path, exist_ok=True)
    with open(save_path / new_save_name, "wb") as file:
        try:
            file.write(f"{instance.__class__.__name__},\n".encode())
        except:
            pass

        file.write(f"{new_save_name},\n".encode())
        for key, value in instance.__dict__.items():
            file.write(f"{key},{value}\n".encode())

        if hasattr(instance, "get_support"):
            file.write(f"{instance.get_support()}\n".encode())

def save_rules(
    estimator,
    save_path,
    *,
    save_name="",
    X_names=None,
    y_names=None,
    X_scaler=None,
    y_scaler=None,
):
    # RuleFitのRuleをCSVに保存する関数
    save_path = save_path / "rules"
    os.makedirs(save_path, exist_ok=True)

    if hasattr(estimator, "get_rules") == True:
        rules = estimator.get_rules()
        rules = rules[rules.coef != 0].sort_values(
            by="support", ascending=False
        )
        num_rules_rule = len(rules[rules.type == "rule"])
        num_rules_linear = len(rules[rules.type == "linear"])

        rules_df = pd.DataFrame(rules)

        def convert_string(string):
            new_string = ""
            string_splits = string.split(" & ")
            from math import log10, floor

            for idx, each_string in enumerate(string_splits):
                if idx != 0:
                    new_string += " & "
                each_string_parts = each_string.split(" ")
                if len(each_string_parts) >= 2:
                    feature_ = each_string_parts[0]
                    # print(feature_)
                    if "feature_" in feature_:
                        feature_index = int(feature_.split("_")[1])
                        if X_names is not None:
                            X_name_ = X_names[feature_index]
                        else:
                            X_name_ = feature_
                    else:
                        feature_index = X_names.index(feature_)
                        X_name_ = feature_

                    quate = each_string_parts[1]
                    value = each_string_parts[2]
                    value = float(value)

                    if X_scaler is not None:
                        scale = X_scaler.scale_[feature_index]
                        mean = X_scaler.mean_[feature_index]
                        value = (value * scale) + mean

                    if value != 0:
                        value = round(
                            value, 3 - int(floor(log10(abs(value)))) - 1
                        )

                    new_string += X_name_ + " " + quate + " " + str(value)
                else:
                    new_string += each_string

            return new_string

        def inverse_y(x):
            if y_scaler is not None:
                # print(y_scaler.scale_[0])
                return x * y_scaler.scale_[0]
            else:
                return x

        rules_df["rule"] = rules_df["rule"].apply(convert_string)
        rules_df["coef"] = rules_df["coef"].apply(inverse_y)

        rules_df.to_csv(
            save_path / f"{save_name}_rules_df.csv",
            encoding="shift_jisx0213",
        )

def save_coef_intercept(
    estimator,
    save_path,
    *,
    save_name="",
    X_names=None,
    y_names=None,
    X_scaler=None,
    y_scaler=None,
):

    save_path_std = save_path / "parameter_std"
    save_path_raw = save_path / "parameter_raw"
    os.makedirs(save_path_std, exist_ok=True)
    os.makedirs(save_path_raw, exist_ok=True)

    if hasattr(estimator, "coef_") == True:
        estimator_coef_std_df = pd.DataFrame(estimator.coef_)
        if X_scaler is not None:
            estimator_coef_raw_df = pd.DataFrame(
                X_scaler.inverse_transform(estimator.coef_)
            )
        else:
            estimator_coef_raw_df = estimator_coef_std_df

        if estimator.__class__.__name__ == "PLSRegression":
            estimator_coef_std_df = estimator_coef_std_df.T
            estimator_coef_raw_df = estimator_coef_raw_df.T

        if len(estimator_coef_std_df.columns) == 1:
            estimator_coef_std_df = estimator_coef_std_df.T
            estimator_coef_raw_df = estimator_coef_raw_df.T

        if (X_names is not None) and (
            len(X_names) == len(estimator_coef_std_df.columns)
        ):
            estimator_coef_std_df.columns = [
                "{}".format(X_name) for X_name in X_names
            ]
            estimator_coef_raw_df.columns = [
                "{}".format(X_name) for X_name in X_names
            ]
        else:
            estimator_coef_std_df.columns = [
                "X_{}".format(i)
                for i in range(len(estimator_coef_std_df.columns))
            ]

            estimator_coef_raw_df.columns = [
                "X_{}".format(i)
                for i in range(len(estimator_coef_raw_df.columns))
            ]

        if hasattr(estimator, "intercept_") == True:

            intercept_ = estimator.intercept_
            if isinstance(intercept_, list) == False:
                intercept_ = [intercept_]

            estimator_intercept_std_df = pd.DataFrame(intercept_)
            estimator_intercept_std_df = estimator_intercept_std_df.T
            estimator_intercept_std_df.columns = ["切片"]
            estimator_parameter_std_df = pd.concat(
                [estimator_intercept_std_df, estimator_coef_std_df], axis=1
            )

            if y_scaler is not None and X_scaler is not None:
                intercept_raw = y_scaler.inverse_transform(intercept_)
                estimator_intercept_raw_df = pd.DataFrame(intercept_raw)
                estimator_intercept_raw_df = estimator_intercept_raw_df.T
                estimator_intercept_raw_df.columns = ["切片"]
                estimator_parameter_raw_df = pd.concat(
                    [estimator_intercept_raw_df, estimator_coef_raw_df],
                    axis=1,
                )

            else:
                estimator_parameter_raw_df = estimator_parameter_std_df

        else:
            estimator_parameter_std_df = estimator_coef_std_df
            estimator_parameter_raw_df = estimator_coef_raw_df

        if y_names is not None:
            estimator_parameter_std_df["y_names"] = pd.DataFrame(y_names)

            estimator_parameter_raw_df["y_names"] = pd.DataFrame(y_names)

        else:
            estimator_parameter_std_df["y_names"] = "y"
            estimator_parameter_raw_df["y_names"] = "y"

        estimator_parameter_std_df = estimator_parameter_std_df.set_index(
            "y_names"
        )
        estimator_parameter_raw_df = estimator_parameter_raw_df.set_index(
            "y_names"
        )

        estimator_parameter_std_df.to_csv(
            save_path_std / f"{save_name}_parameter_std.csv",
            encoding="shift_jisx0213",
        )

        if X_scaler is not None:
            estimator_parameter_raw_df.to_csv(
                save_path_raw / f"{save_name}_parameter_raw.csv",
                encoding="shift_jisx0213",
            )

        for y_name in y_names:
            estimator_parameter_std_df_T = estimator_parameter_std_df.T
            estimator_parameter_std_df_T[
                "ABS"
            ] = estimator_parameter_std_df_T[y_name].abs()
            estimator_parameter_std_df_T = (
                estimator_parameter_std_df_T.sort_values(
                    by="ABS", ascending=False
                )
            )
            estimator_parameter_std_df_barplot = copy.deepcopy(
                estimator_parameter_std_df_T
            )
            estimator_parameter_std_df_barplot = (
                estimator_parameter_std_df_barplot[:15]
            )
            coefficients = estimator_parameter_std_df_barplot[
                y_name
            ].values.flatten()
            feature_name = estimator_parameter_std_df_barplot[y_name].index
            feature_name = list(feature_name)

            plt.figure(figsize=(15, 9))
            plt.bar(feature_name, coefficients)
            plt.xticks(rotation=90)
            plt.title("標準回帰係数_{}".format(save_name))
            plt.savefig(
                save_path_std
                / "{}_{}_barplot.png".format(save_name, y_name),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()
