import gc
from inspect import signature
import os
import sys
import time
import traceback
import math

# Third-party Libary
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
import shap
import numpy as np

# from this library
# from ._base import add_text_topng, round_significant_digits
from ..utils.base import add_text_topng, round_significant_digits
from ..utils.base import get_estimator

matplotlib.use("Agg")


def explainer_shap(
    estimator,
    X,
    y,
    X_scaler,
    y_scaler,
    X_selector,
    X_names,
    X_transformed_names,
    y_names,
    score_dict,
    theme_path,
    model_name=None,
    info_df=None,
    datetime_df=None,
    except_explainers=None,
    class_names_list=None,
):

    # shapを計算するかどうか事前に設定した値を取得。Falseの場合はshap計算しない
    # 事前に設定していない場合はSHAPを計算する
    print(f"estimator {estimator}")    
    estimator_, _ = get_estimator(estimator, model_type="estimator", remove_multioutput=False)    
    shap_ = getattr(estimator_, "shap", True)

    print(f"shap_ {shap_}")

    if shap_ == False:        
        print("skip calculate the shap value")
        return

    os.makedirs(theme_path / "shap", exist_ok=True)
    if except_explainers is None:
        except_explainers = []

    if info_df is None:
        info_df = pd.DataFrame()

    if datetime_df is None:
        datetime_df = pd.DataFrame()

    X_raw_df = pd.DataFrame(X_scaler.inverse_transform(X))
    y_raw_df = pd.DataFrame(y_scaler.inverse_transform(y))
    X_raw_df.columns = X_names
    y_raw_df.columns = y_names

    X_df = pd.DataFrame(X)
    X_df.columns = X_transformed_names

    def calculate_shap(
        estimator,
        X,
        model_name,
        n_output,
        idx_output=None,
        multioutput=False,
        y_name="",
        class_names_list=None,
    ):
        # shapを計算するかどうか事前に設定した値を取得。Falseの場合はshap計算しない
        # 事前に設定していない場合はSHAPを計算する

        if model_name is None:
            model_name = estimator.__class__.__name__
        try:
            if estimator.__class__.__name__ in [
                "RandomForestRegressor",
                "DecisionTreeRegressor",
                "RandomForestClassifier",
                "DecisionTreeClassifier",
                "LGBMRegressor",
                "LGBMClassifier",
                "ExtraTreeRegressor",
                "ExtraTreeClassifier",
                "ExtraTreesRegressor",
                "ExtraTreesClassifier",
                "NGBoost",
            ]:
                print("TreeExplainer")
                # refer https://shap.readthedocs.io/en/latest/generated/shap.explainers.Tree.html
                explainer = shap.TreeExplainer(
                    model=estimator,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )

            elif estimator.__class__.__name__ == ["XGBRegressor", "XGBClassifier"]:

                print("TreeExplainer - XGB")
                explainer = shap.TreeExplainer(
                    model=estimator,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )

            elif (hasattr(estimator, "coef_") == True) and (
                estimator.__class__.__name__ not in ["PLSRegression", "SGDOneClassSVM", "RuleFit"]
            ):
                print("LinearExplainer", estimator.__class__.__name__)

                # refer https://shap.readthedocs.io/en/latest/generated/shap.explainers.Linear.html

                explainer = shap.LinearExplainer(
                    estimator, X, feature_perturbation="interventional"
                )

            else:
                # KernelExplainerの計算は非常に重たいので扱いに注意
                print("KernelExplainer")
                print(estimator.__class__.__name__)

                time.sleep(5)
                explainer = shap.KernelExplainer(estimator.predict, X)

                # https://dsbowen.github.io/gshap/kernel_explainer/
                # https://medium.com/analytics-vidhya/shap-part-2-kernel-shap-3c11e7a971b1

            print("except_explainers")
            print(except_explainers)
            print(explainer.__class__.__name__)
            print(explainer.__class__.__name__ in except_explainers)

            if explainer.__class__.__name__ in except_explainers:
                return

            # KernelExplainer を使う際のサンプル数の上限値　#実際には説明変数の数でも絞ったほうがいいが。。
            shap_max_num = 200

            if "Kernel" in explainer.__class__.__name__ and len(X) > shap_max_num:
                # 上限を超える場合はランダムサンプリング
                shap_X = shap.sample(X, shap_max_num)
            else:
                shap_X = X

            shap_X_raw = X_scaler.inverse_transform(shap_X)
            shap_X_raw = pd.DataFrame(shap_X_raw)
            shap_X_raw.columns = shap_X.columns

            shap_values = explainer.shap_values(X=shap_X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values_raw = y_scaler.inverse_transform(shap_values)

            print("type(shap_values)")
            print(type(shap_values))
            print("type(shap_values_raw)")
            print(type(shap_values_raw))

            # visualize the first prediction's explanation (use matplotlib = True to avoid Javascript)
            # 複数の目的変数の処理

            for output_index in range(n_output):
                fig = plt.figure()

                print("n_output : ", n_output)

                if n_output >= 2:
                    shap_value = shap_values[output_index]
                    y_name = y_names[output_index]
                    if class_names_list is not None:
                        class_names = class_names_list[output_index]
                    else:
                        class_names = None

                elif n_output == 1:
                    shap_value = shap_values
                    if y_name is None:
                        y_name = y_names[0]

                    if class_names_list is not None:
                        class_names = class_names_list[0]
                    else:
                        class_names = None

                    if (isinstance(shap_value, list) == True) and (
                        type(shap_value[0]).__module__ == np.__name__
                    ):
                        shap_value = np.array(shap_value[0])

                    else:
                        pass

                if multioutput == False:
                    idx_output = output_index

                save_name = "{m}_{o}".format(m=model_name, o=y_name)

                # SHAPで表示する説明変数の数
                max_display = 15

                # モデル名、評価方法、R2、MAE
                text_df = pd.DataFrame()
                text_df.loc[0, "model_name"] = model_name
                if len(score_dict) != 0:

                    text_df.loc[0, "evaluate"] = f'{score_dict["eval_method"]}'

                    for k, v in score_dict.items():
                        if k not in [
                            "r2_score",
                            "mean_absolute_error",
                            "accuracy_score",
                            "auc",
                        ]:
                            continue
                        else:
                            score_ = round_significant_digits(v, 3)
                            text_df.loc[0, str(k)] = f"{k}:{score_}"
                else:
                    text_df.loc[0, "evaluate"] = "NAN"
                    text_df.loc[0, "score_1"] = "NAN"
                    text_df.loc[0, "score_2"] = "NAN"

                def shap_plots(
                    shap_value,
                    shap_X,
                    feature_names=X_transformed_names,
                    class_names=class_names,
                    plot_type="bar",
                    max_display=max_display,
                    plot_size=(10, 8),
                    show=False,
                ):
                    try:
                        shap.summary_plot(
                            shap_value,
                            shap_X,
                            feature_names=feature_names,
                            class_names=class_names,
                            plot_type=plot_type,
                            max_display=max_display,
                            plot_size=(10, 8),
                            show=False,
                        )

                        plt.gcf().set_size_inches(10, 10)
                        plt.title(y_name, fontsize=18)
                        plt.rcParams.update({"font.size": 20})
                        save_path = (
                            theme_path / "shap" / f"{save_name}_shap_{plot_type}_values.png"
                        )
                        plt.savefig(save_path, bbox_inches="tight")

                        try:
                            plt.savefig(save_path, bbox_inches="tight")
                        except:
                            plt.savefig(save_path)

                        plt.close()

                        # SHAPの画像にモデル名、精度などを付与
                        # add_text_topng(save_path, theme_path, text_df)

                    except:
                        traceback.print_exc()
                        time.sleep(0.5)

                    return

                for plot_type in ["bar", "dot", "violin"]:
                    shap_plots(
                        shap_value=shap_value,
                        shap_X=shap_X,
                        feature_names=X_transformed_names,
                        class_names=class_names,
                        plot_type=plot_type,
                        max_display=max_display,
                        plot_size=(10, 8),
                        show=False,
                    )

                # shap_valuesのCSV化
                df_shap_value = pd.DataFrame(shap_value)

                shap_X_before_selected = X_selector.inverse_transform(shap_X)
                shap_X_before_scaled = X_scaler.inverse_transform(shap_X_before_selected)
                shap_X_before_scaled = pd.DataFrame(shap_X_before_scaled)
                shap_X_before_scaled.columns = X_names

                if y_scaler.__class__.__name__ == "StandardScaler":
                    y_means = y_scaler.mean_
                    y_vars = y_scaler.var_
                    y_scale = y_vars[idx_output] ** 0.5

                elif y_scaler.__class__.__name__ == "MinMaxScaler":
                    y_mins = y_scaler.min_
                    y_scales = y_scaler.scale_
                    y_scale = y_scale[idx_output]

                else:
                    y_scale = 1

                shap_value_raw = shap_value * y_scale

                df_shap_value_raw = pd.DataFrame(shap_value_raw)

                df_shap_value.columns = X_transformed_names
                df_shap_value_raw.columns = X_transformed_names

                df_shap_value = pd.concat(
                    [info_df, shap_X_before_scaled, df_shap_value_raw], axis=1
                )
                df_shap_value.to_csv(
                    theme_path / "shap" / f"{save_name}_shap_values.csv",
                    encoding="shift_jisx0213",
                )

                # 説明変数ごとのSHAP図
                for X_index, X_name in enumerate(X_transformed_names):
                    # 正規化
                    save_folder = theme_path / "shap" / y_name / "std"
                    os.makedirs(save_folder, exist_ok=True)

                    dependence_plot_args = {
                        "ind": X_name,
                        "interaction_index": X_name,
                        "shap_values": shap_value,
                        "features": shap_X,
                        "show": False,
                        "xmin": min(X_df[X_name]),
                        "xmax": max(X_df[X_name]),
                        "ymin": -0.75,
                        "ymax": 0.75,
                    }

                    # dependence_plot が ymin, ymaxに対応しているか
                    # 尚、pip installで通常インストールされるshapは対応していない。

                    sig_ = signature(shap.dependence_plot)
                    params = sig_.parameters
                    if "ymin" not in params.keys():
                        del dependence_plot_args["ymin"]
                    if "ymax" not in params.keys():
                        del dependence_plot_args["ymax"]

                    shap_plot = shap.dependence_plot(**dependence_plot_args)

                    plt.gcf().set_size_inches(10, 6)
                    plt.title("{}_{}".format(X_name, y_name), fontsize=18)

                    plt.rcParams.update({"font.size": 20})
                    save_path = save_folder / f"{save_name}_{X_name}_force_plot.png"
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.cla()
                    plt.clf()
                    plt.close()
                    plt.close("all")

                    add_text_topng(save_path, theme_path, text_df)

                    # 生データ
                    save_folder = theme_path / "shap" / y_name / "raw"
                    os.makedirs(save_folder, exist_ok=True)

                    plt.figure()
                    dependence_plot_args = {
                        "ind": X_name,
                        "interaction_index": X_name,
                        "shap_values": shap_value_raw,
                        "features": shap_X_raw,
                        "show": False,
                        "xmin": min(X_raw_df[X_name]),
                        "xmax": max(X_raw_df[X_name]),
                        "ymin": -(y_scale) * 0.75,
                        "ymax": (y_scale) * 0.75,
                    }

                    # old version does not have 'ymin', 'ymax' argument
                    sig_ = signature(shap.dependence_plot)
                    params_ = sig_.parameters
                    if "ymin" not in params_.keys():
                        del dependence_plot_args["ymin"]
                    if "ymax" not in params_.keys():
                        del dependence_plot_args["ymax"]

                    shap_plot = shap.dependence_plot(**dependence_plot_args)

                    plt.gcf().set_size_inches(10, 6)
                    plt.title("{}_{}".format(X_name, y_name), fontsize=18)

                    plt.rcParams.update({"font.size": 20})
                    save_path = save_folder / f"{save_name}_{X_name}_force_plot_raw.png"
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.cla()
                    plt.clf()
                    plt.close()
                    plt.close("all")
                    add_text_topng(save_path, theme_path, text_df)

            shap_importance = np.mean(np.abs(shap_values), axis=0)
        except:
            with open(theme_path / "shap" / f'{model_name}_error.log', 'a') as f:
                traceback.print_exc(file=f)

    # MultiOutputRegressorの場合の処理
    if estimator.__class__.__name__ in [
        "MultiOutputRegressor",
        "MultiOutputClassifier",
    ]:
        for feature_index, y_name in enumerate(y_names):

            each_model = estimator.estimators_[feature_index]

            calculate_shap(
                each_model,
                X,
                model_name,
                1,
                idx_output=feature_index,
                multioutput=True,
                y_name=y_name,
                class_names_list=class_names_list,
            )

    else:

        calculate_shap(
            estimator,
            X,
            model_name,
            len(y_names),
            idx_output=None,
            multioutput=False,
            y_name=None,
            class_names_list=class_names_list,
        )

    return


from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression


def segmented_regression(X, y):
    model = LinearTreeRegressor(base_estimator=LinearRegression(), max_depth=1)
    model.fit(X, y)

    model_s = model.summary()
    threshold = model_s[0].get("th", None)
    if threshold is not None:
        coef_ = [i.coef_ for i in model_s[0]["models"]]
        intercept_ = [i.intercept_ for i in model_s[0]["models"]]
    else:
        coef_ = model_s[0]["models"].coef_
        intercept_ = model_s[0]["models"].intercept_

    result = {
        "model": model,
        "threshold": threshold,
        "coef_": coef_,
        "intercept_": intercept_,
    }

    return result
