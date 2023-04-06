# refer https://note.nkmk.me/python-ast-literal-eval/
import ast
import argparse
import copy
import datetime
from functools import partial
import itertools
import importlib.util
from inspect import signature
from tkinter import Label
import joblib
from joblib import memory
import os
from pathlib import Path
import platform
import cloudpickle
import pprint
import psutil
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

# warnings.simplefilter("ignore")

# Third-party Libary
import category_encoders as ce
import chardet
import cloudpickle
import japanize_matplotlib
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from memory_profiler import profile
import openpyxl
from openpyxl.styles import Alignment
import optuna
from optuna.integration import OptunaSearchCV
import pandas as pd
import PySimpleGUI as sg  # LGPL
from scipy.stats import rankdata
import numpy as np
from boruta import BorutaPy
import xgboost

matplotlib.use("Agg")

## sklearn
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.metrics import recall_score, precision_score, log_loss

from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    LeavePOut,
    GroupKFold,
    LeaveOneGroupOut,
    TimeSeriesSplit,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import _num_features, _num_samples

# original library
# from sklearn_expansion.identity_mapping import IdentityMapping
from sklearn_expansion.feature_selection import SelectByModel
from sklearn_expansion.model_selection import OptunaSearchClustering  # OptunaSearchCV,

from sklearn_expansion.preprocessing import ImportanceScaler


from sklearn_expansion.model_selection import WalkForwardValidation
from sklearn_expansion.model_selection import cross_val_predict

# from this library
from .datasets._load_csv import load_dataset
from .estimators._estimators import load_all_models
try:
    from .estimators._tabnet import load_tabnet
    from .estimators._automl import load_automl
except:
    pass
from .optimizers._optimizers import load_tuning_models
from .explainers._explainers import explainer_shap
from .explorers._explorers import shapiro_wilk, correlation_coefficient, pairplot
from .evaluaters._base import save_true_predict_result
from .feature_selection._selectors import load_selectors
from .models.adjust_model import adjust_hyperparameter_todata
from .models.load_models import load_models
from .utils.base import add_text_topng, round_significant_digits
from .utils.base import recursive_replace
from .utils.base import get_estimator
from .utils.create_model import make_pipeline, make_pipelines
from .utils.create_model import make_args, get_model_params, model_fit, wrap_searchcv
from .utils.save_results import save_model, save_params, save_results
from .utils.utils import delayed_start, make_archive_then_delete
from .model_selection.cv import load_splliter_dict


def supervised_learning(
    models,
    X_df,
    y_df,
    *,
    sv_type="regression",
    X_scaler=StandardScaler(),
    y_scaler=StandardScaler(),
    raw_df=None,
    info_df=None,
    datetime_df=None,
    group_df=None,
    evaluate_sample_weight_df=None,
    training_sample_weight_df=None,
    evaluate_splitter=None,
    optimize_setting=None,
    evaluate_setting=None,
    explainer_setting=None,
    theme_name: str = None,
    save_path: str = None,
    n_jobs: int = -2,
    encoding:str ="shift_jisx0213"):

    """
    機械学習を実行するメインプログラム

    Parameters
    ----------
    models: list[estimator]
        sklearn-API compatible estimator
    X_df: pd.DataFrame
    y_df: pd.DataFrame
    X_scaler : estimator]
        X scaler.
    y_scalers : estimator
        y scalers.
    optimize_setting : list of dict.
        Optimize setting.
    evaluate_setting : list of dict.
        Evaluate setting of holdout, cv, dcv.
    explainer_setting : list of dict.
        Explain setting of shap, pdp others.
    theme_name : str.
        Use theme_name as a part of save_path
    save_path : str, Path object.
        save_path
    n_jobs : int
        The argument of joblib.Parallel
    """
    start_time = time.time()

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path / "version" , exist_ok=True)

    libraries = [sklearn, optuna, np, pd]
    with open(save_path / "version.txt", "w") as f:
        for each_library in libraries:
            try:
                f.write(f"{each_library.__name__} {each_library.__version__}")
                f.write("\n")    
            except:
                pass

    print("Data reading...")
    if raw_df is not None:
        raw_df.to_csv(save_path / "raw_data.csv", encoding=encoding)

    n_X_features = _num_features(X_df)

    if group_df is not None:
        if len(group_df.columns) != 0:
            group = group_df.iloc[:, 0]
        else:
            group = None
    else:
        group = None

    if evaluate_sample_weight_df is not None:
        if len(evaluate_sample_weight_df.columns) != 0:
            evaluate_sample_weight_df = np.array(evaluate_sample_weight_df).flatten()
        else:
            evaluate_sample_weight_df = None
    else:
        evaluate_sample_weight_df = None

    if training_sample_weight_df is not None:
        if len(training_sample_weight_df.columns) != 0:
            training_sample_weight_df = np.array(training_sample_weight_df).flatten()
        else:
            training_sample_weight_df = None
    else:
        training_sample_weight_df = None

    print("evaluate_sample_weight_df : ", evaluate_sample_weight_df)
    print("training_sample_weight_df : ", training_sample_weight_df)

    if sv_type == "classification":
        y_names = y_df.columns
        ce_OrdinalEncoder = ce.OrdinalEncoder(cols=y_names)
        y_df = ce_OrdinalEncoder.fit_transform(y_df)
        class_names_dict = ce_OrdinalEncoder.category_mapping
        class_names_list = [i["mapping"].values.tolist() for i in class_names_dict]

    else:
        class_names_list = None

    X_names = X_df.columns
    y_names = y_df.columns
    info_names = info_df.columns

    print("Data exploring...")
    explorer(X_df, y_df, save_path)

    print("Scaling...")
    X_scaled = X_scaler.fit_transform(X_df)
    y_scaled = y_scaler.fit_transform(y_df)

    scaler_path = save_path / f"scaler"
    save_model(X_scaler, scaler_path, save_name="X_scaler", model_type="scaler")
    save_model(y_scaler, scaler_path, save_name="y_scaler", model_type="scaler")

    X_scaled_df = pd.DataFrame(X_scaled, columns=X_names)
    y_scaled_df = pd.DataFrame(y_scaled, columns=y_names)

    # evaluate_cv_setting = evaluate_setting["evaluate_cv_setting"]
    # evaluate_dcv_setting = evaluate_setting["evaluate_dcv_setting"]
    # evaluate_holdout_setting = evaluate_setting["evaluate_holdout_setting"]
    # evaluate_all_setting = evaluate_setting["evaluate_all_setting"]

    print(n_X_features)
    print(models)
    models = [adjust_hyperparameter_todata(i, X_df, y_df) for i in models]
    print(models)

    # Hold-out test
    if False: # evaluate_holdout_setting["do"]:
        print("holdout evaluating...")
        # cv_scores_df = evaluater(evaluate_setting, optimized_models, X_df, y_scaled_df, datetime_df, group, sample_weight,
        #                         sv_type=sv_type, class_names_list=class_names_list)

        # train-test
        test_size = test_size
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_scaled_df)

        if cross_optimize:
            cross_optimize()
            best_model
            evaluater(best_model)

        if holdout_optimize:
            holdout_optimize()
            evaluater(best_model)

        print("cv_scores : ", cv_scores)

    # Double-Cross-Validation
    dcv = True
    if dcv==True:
        print("dcv evaluating...")

        pprint.pprint("models")
        pprint.pprint(models)

        eval_path = save_path / "eval" / "dcv"
        dcv_scores_df_list = joblib.Parallel(n_jobs=1)(
            joblib.delayed(evaluater)(
                model,
                X_df,
                y_scaled_df,
                X_scaler,
                y_scaler,
                X_names,
                y_names,
                datetime_df,
                group,
                evaluate_sample_weight_df,
                eval_path,
                sv_type=sv_type,
                class_names_list=class_names_list,
                model_idx=model_idx,
                splitter=evaluate_splitter
            )
            for model_idx, model in enumerate(models)
        )

        dcv_scores_df = pd.DataFrame()
        for each_scores_df in dcv_scores_df_list:
            dcv_scores_df = pd.concat([dcv_scores_df, each_scores_df], axis=0)

        print("save_path")
        print(save_path)

        dcv_scores_df = dcv_scores_df.reset_index()
        dcv_scores_df.to_csv(
            save_path / "dcv_scores.csv", encoding=encoding, index=True
        )
        dcv_scores_df.to_excel(
            save_path / "dcv_scores.xlsx", index=True, encoding=encoding
        )
        print("cross_optimize")

        # SearchCVをfit
        # best_estimator_などが生成される
        [
            each_pipeline.fit(X_df, y_scaled_df, groups=group)
            for each_pipeline in models
        ]

        # クソコード注意
        # models.pyで各estimatorに対して、shapを計算するかなどのデフォルト値をattribute(estiamtor.shap=False or True)として設定しています。
        # しかしながら、OptunaSearchCV(estimator)とし、学習後のbest_estimator_は内部でclone(estimator)されます
        # すると、models.pyで設定したデフォルト値はcloneした際に引き継がれません。
        # cloneで引き継がれるのは、各クラスの__init__で定義されたattributeのみのためです。

        # 従って、tune_pipelines_dcvにはestimator.shapは定義されていません。
        # ゆえに苦渋の決断として、tune_pipelines_dcv [ OptunaSearchCv(Pipeline) ] とpipelines_dcv [Pipeline]　をzipで回しています
        # clone(estimator)で引き継がれる様に変更することも出来ると思いますが、それは更なるクソコードを生み出します。

        best_pipelines_dcv = []
        for each_model in models:
            # GridSearchCV もしくはOptunaSearchCVでwrapされていない
            # つまりbase_estimator_を持っていない場合はスキップ
            if hasattr(each_model, "best_estimator_") == False:
                print(f"each_model {each_model} has no best_estimator_")
                best_pipelines_dcv.append(each_model)
                continue

            default_estimator, _ = get_estimator(
                each_model,
                remove_pipeline=True,
                remove_searcher=True,
                remove_multioutput=False,
            )

            best_estimator_in_SearchCV = each_model.best_estimator_

            best_estimator_, _ = get_estimator(
                best_estimator_in_SearchCV,
                remove_pipeline=True,
                remove_searcher=True,
                remove_multioutput=False,
            )

            each_shap = getattr(default_estimator, "shap", True)
            each_type = getattr(default_estimator, "type", "others")

            setattr(best_estimator_, "shap", each_shap)
            setattr(best_estimator_, "type", each_type)
            best_pipelines_dcv.append(best_estimator_in_SearchCV)

        best_models = best_pipelines_dcv

        workbook_excel = openpyxl.load_workbook(
            filename=save_path / "dcv_scores.xlsx"
        )
        worksheet = workbook_excel.worksheets[0]
        font = openpyxl.styles.Font(name="HGｺﾞｼｯｸM")
        for row in worksheet:
            for cell in row:
                worksheet[cell.coordinate].font = font
                worksheet[cell.coordinate].alignment = Alignment(
                    wrap_text=True, vertical="center", horizontal="center"
                )

        worksheet.column_dimensions["A"].width = 7
        worksheet.column_dimensions["B"].width = 7
        worksheet.column_dimensions["B"].alignment = Alignment(
            wrap_text=True, horizontal="left"
        )
        worksheet.column_dimensions["C"].width = 20
        worksheet.column_dimensions["C"].alignment = Alignment(
            wrap_text=True, horizontal="left"
        )
        worksheet.column_dimensions["D"].number_format = "0.00"
        worksheet.column_dimensions["D"].width = 20
        worksheet.column_dimensions["D"].alignment = Alignment(
            wrap_text=True, vertical="center"
        )
        worksheet.column_dimensions["E"].width = 20
        worksheet.column_dimensions["E"].alignment = Alignment(
            wrap_text=True, vertical="center"
        )
        worksheet.column_dimensions["F"].width = 20
        worksheet.column_dimensions["F"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )
        worksheet.column_dimensions["F"].width = 20
        worksheet.column_dimensions["F"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )
        worksheet.column_dimensions["G"].width = 20
        worksheet.column_dimensions["G"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )
        worksheet.column_dimensions["H"].width = 25
        worksheet.column_dimensions["H"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )
        worksheet.column_dimensions["I"].width = 40
        worksheet.column_dimensions["I"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )
        worksheet.column_dimensions["J"].width = 40
        worksheet.column_dimensions["J"].alignment = Alignment(
            wrap_text=True, horizontal="center"
        )

        for i in range(100):
            worksheet.row_dimensions[i].height = 13.5

        workbook_excel.save(save_path / "dcv_scores.xlsx")

    else:
        print("dcv is not choosed")
        dcv_scores_df = None
        best_pipelines_dcv = []

    # Crossoptimize  & validation -deprecated
    if False: #evaluate_cv_setting["do"]:
        print("cv evaluating...")
        print("X_scaler : ", X_scalers)
        print("missing_data_processes : ", missing_data_processes)
        print("feature_engineerings : ", feature_engineerings)
        print("selectors : ", selectors)

        X_scalers_parts = [i["parts"] for i in X_scalers]
        missing_data_processes_parts = [i["parts"] for i in missing_data_processes]
        feature_engineerings_parts = [i["parts"] for i in feature_engineerings]
        selector_parts = [i["parts"] for i in selectors]
        model_parts = [i["parts"] for i in supervised_estimators]

        ## selectorがestimatorを用いる場合
        ## selector内のestimatorのハイパラ事前最適化
        estimator_in_selector_parts = [
            ("estimator_in_selector", i[1].estimator)
            if hasattr(i[1], "estimator")
            else ("estimator_in_selector", FunctionTransformer())
            for i in selector_parts
        ]
        pre_tune_sets = list(
            itertools.product(
                X_scalers_parts,
                missing_data_processes_parts,
                feature_engineerings_parts,
                estimator_in_selector_parts,
            )
        )
        print("pretune of estimator in selector...")
        pre_tune_pipeline = [Pipeline(i) for i in pre_tune_sets]
        [
            each_pipeline.fit(X_df, y_scaled_df)
            for each_pipeline in pre_tune_pipeline
        ]
        print("pretuned")

        best_estimator_in_selector = [
            i.steps[-1][1].best_estimator_
            if hasattr(i.steps[-1][1], "best_estimator_")
            else i.steps[-1][1]
            for i in pre_tune_pipeline
        ]
        selectors = [i["parts"] for i in selectors]
        # deepcopyする必要性はないが、他のDCVやHoldoutに影響を与えないため。
        selector_with_tuned_estimator_parts = copy.deepcopy(selectors)
        [
            setattr(i[1], "estimator", j)
            for i, j in zip(
                selector_with_tuned_estimator_parts, best_estimator_in_selector
            )
            if hasattr(i[1], "estimator")
        ]

        ### 事前フィット
        ### PLS-VIPなどフィルターメソッドでは不要
        ### Borutaの場合、説明変数選択を全データに対してフィットして実施
        ### GAの場合、説明変数選択をCVでフィットして実施。
        tuned_selector_sets = list(
            itertools.product(
                X_scalers_parts,
                missing_data_processes_parts,
                feature_engineerings_parts,
                selector_with_tuned_estimator_parts,
            )
        )

        selector_pipe = [Pipeline(i) for i in tuned_selector_sets]
        print("selector fitting...")

        [
            model_fit(X_df, y_scaled_df, sample_weight)
            for idx, i in enumerate(selector_pipe)
            if i.steps[-1][1].__class__.__name__
            in ["SelectByBoruta", "SelectByGA", "SelectByGACV"]
        ]
        print("selectors fitted")

        prefitted_selector_parts = [i.steps[-1] for i in selector_pipe]
        [
            setattr(i[1], "prefit", True)
            for i in prefitted_selector_parts
            if i[1].__class__.__name__
            in ["SelectByBoruta", "SelectByGA", "SelectByGACV"]
        ]

        pipelines_parts = [
            copy.deepcopy(i)
            for i in itertools.product(
                X_scalers_parts,
                missing_data_processes_parts,
                feature_engineerings_parts,
                prefitted_selector_parts,
                model_parts,
            )
        ]
        pipelines_cv = [Pipeline(i) for i in pipelines_parts]
        optimize_splitter = optimize_setting["splitter"]
        optimize_cv_splitter_args = optimize_setting["splitter_args"]
        # optimize_split = get_n_splits_(optimize_splitter,  X_df, y=y_scaled_df , group=group, splitter_args=optimize_cv_splitter_args)

        searcher_class = optimize_setting["searcher"]
        searcher_args = optimize_setting["searcher_args"]

        tune_pipelines_cv = [
            searcher_class(
                pipe,
                params,
                cv=optimize_splitter(**optimize_cv_splitter_args),
                **searcher_args,
            )
            for pipe, params in zip(pipelines_cv, pipeline_of_ranges)
        ]
        # tune_pipelines_cv = [OptunaSearchCV(pipe, params) for pipe, params in zip(pipelines_cv, pipeline_of_ranges)]

        """
        def hyperparameter_fit(model, X, y, sample_weight):
            estimator, estimator_name = get_estimator(model)
            print('{} tuning...'.format(estimator_name))

            if estimator_name not in  ['TabNetRegressor']:# , 'Ag_TabularPredictor']:
                model_fit(model, X_df,y_scaled_df, sample_weight)
            else:
                #model_fit(model, X_df,y_scaled_df, sample_weight)
                model_fit(model, np.array(X_df),np.array(y_scaled_df), sample_weight)

            print('{} tuned'.format(estimator))
            return
        """

        print("Hyperparameter tuning...")
        [model_fit(i, X_df, y_scaled_df, sample_weight) for i in tune_pipelines_cv]
        best_pipelines_cv = [i.best_estimator_ for i in tune_pipelines_cv]
        print("Hyperparameter tuned")

        eval_path = save_path / "eval" / "splitter"
        cv_scores_df = evaluater(
            evaluate_cv_setting,
            best_pipelines_cv,
            X_df,
            y_scaled_df,
            X_scaler,
            y_scaler,
            X_names,
            y_names,
            datetime_df,
            group,
            evaluate_sample_weight_df,
            eval_path,
            sv_type=sv_type,
            class_names_list=class_names_list,
        )

        sort_score = evaluate_cv_setting["scorer"][0]
        cv_scores_df.sort_values(sort_score, ascending=False, inplace=True)

        print("cv_scores_df")
        print(cv_scores_df)

        time.sleep(0.5)

        cv_scores_df.to_csv(save_path / "cv_scores.csv", encoding=encoding)
        cv_scores_df.to_excel(
            save_path / "cv_scores.xlsx", index=False, encoding=encoding
        )

        workbook_excel = openpyxl.load_workbook(
            filename=save_path / "cv_scores.xlsx"
        )
        worksheet = workbook_excel.worksheets[0]
        font = openpyxl.styles.Font(name="HGｺﾞｼｯｸM")
        for row in worksheet:
            for cell in row:
                worksheet[cell.coordinate].font = font
                worksheet[cell.coordinate].alignment = Alignment(
                    wrap_text=True, vertical="center", horizontal="center"
                )

        worksheet.column_dimensions["A"].width = 20
        worksheet.column_dimensions["B"].width = 35
        worksheet.column_dimensions["B"].alignment = Alignment(
            wrap_text=True, horizontal="left"
        )
        worksheet.column_dimensions["C"].width = 35
        worksheet.column_dimensions["C"].alignment = Alignment(
            wrap_text=True, horizontal="left"
        )
        worksheet.column_dimensions["D"].number_format = "0.00"
        worksheet.column_dimensions["D"].width = 35
        worksheet.column_dimensions["D"].alignment = Alignment(
            wrap_text=True, vertical="center"
        )
        worksheet.column_dimensions["E"].width = 10
        worksheet.column_dimensions["E"].alignment = Alignment(
            wrap_text=True, vertical="center"
        )
        worksheet.column_dimensions["F"].width = 10
        worksheet.column_dimensions["F"].alignment = Alignment(
            wrap_text=True, horizontal="left"
        )
        worksheet.column_dimensions["F"].width = 35

        workbook_excel.save(save_path / "cv_scores.xlsx")

    else:
        best_pipelines_cv = []
        cv_scores_df = None

    if dcv_scores_df is not None:
        print("best models is dcv")
        best_models = best_pipelines_dcv
        scores_df_list = dcv_scores_df_list
        scores_df = dcv_scores_df
    else:
        print("best models is cv")
        best_models = best_pipelines_cv
        scores_df = cv_scores_df

    print("best_models")
    print(best_models)

    # refit to all-data
    print("refit to all-data...")

    [
        model_fit(i, X_df, y_scaled_df, sample_weight=training_sample_weight_df)
        for i in best_models
    ]

    eval_path = save_path / "eval" / "alldata"

    alldata_scores_df_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(evaluater)(
            model,
            X_df,
            y_scaled_df,
            X_scaler,
            y_scaler,
            X_names,
            y_names,
            datetime_df,
            group,
            evaluate_sample_weight_df,
            eval_path,
            sv_type=sv_type,
            class_names_list=class_names_list,
            model_idx=model_idx,
        )
        for model_idx, model in enumerate(best_models)
    )

    for cv_name, each_best_model in zip(["dcv"], [best_models]):
        estimator_to_all_path = save_path / f"model_all_{cv_name}"

        [
            save_params(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="pipeline",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_params(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="pipelscalerine",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_params(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="feature_engineerings",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_params(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="selector",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_params(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="estimator",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_model(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="pipeline",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_model(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="scaler",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_model(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="feature_engineerings",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_model(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="selector",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_model(
                i,
                estimator_to_all_path,
                save_name=f"{idx}",
                model_type="estimator",
            )
            for idx, i in enumerate(each_best_model)
        ]
        [
            save_results(
                i,
                save_path,
                save_name=idx,
                model_type="estimator",
                X_names=X_names,
                y_names=y_names,
            )
            for idx, i in enumerate(each_best_model)
        ]


    print("explaining..")
    print(explainer_setting)
    print(scores_df)

    name_path = save_path / "name"
    os.makedirs(name_path, exist_ok=True)
    with open(name_path / f"X_names.pkl", "wb") as f:
        cloudpickle.dump(X_names, f)

    with open(name_path / f"y_names.pkl", "wb") as f:
        cloudpickle.dump(y_names, f)

    df_path = save_path / "df"
    os.makedirs(df_path, exist_ok=True)
    with open(df_path / f"X_df.pkl", "wb") as f:
        cloudpickle.dump(y_scaled_df, f)
    with open(df_path / f"y_df.pkl", "wb") as f:
        cloudpickle.dump(y_df, f)
    with open(df_path / f"y_scaled_df.pkl", "wb") as f:
        cloudpickle.dump(y_scaled_df, f)

    with open(df_path / f"info_df.pkl", "wb") as f:
        cloudpickle.dump(info_df, f)
    with open(df_path / f"datetime_df.pkl", "wb") as f:
        cloudpickle.dump(datetime_df, f)

    with open(df_path / f"evaluate_sample_weight_df.pkl", "wb") as f:
        cloudpickle.dump(evaluate_sample_weight_df, f)
    with open(df_path / f"training_sample_weight_df.pkl", "wb") as f:
        cloudpickle.dump(training_sample_weight_df, f)
    with open(df_path / f"group_df.pkl", "wb") as f:
        cloudpickle.dump(group, f)

    def open_folder(path):
        subprocess.run(f"explorer {path}")

    open_folder(save_path)
    end_time = time.time()
    elapsed_time = round((end_time - start_time) / 60, 1)

    with open(save_path / f"time {elapsed_time} min", "wb") as file:
        pass

    print(f"elapsed_time {elapsed_time}")
    print("finished !")

    # explainer(SHAP)
    joblib.Parallel(n_jobs=1)(
        joblib.delayed(explainer)(
            explainer_setting,
            model,
            X_df,
            y_scaled_df,
            X_scaler,
            y_scaler,
            X_names=X_names,
            y_names=y_names,
            score_df=score_df,
            save_path=save_path,
            info_df=info_df,
            datetime_df=datetime_df,
            model_idx=idx,
        )
        for idx, (model, score_df) in enumerate(zip(best_models, scores_df_list))
    )

    print("explained")

    target_folders = [
        save_path / "clustering",
        save_path / "eval" / "alldata",
        save_path / "eval" / "dcv",
        save_path / "randomforest",
        save_path / "importance",
        save_path / "model_all_dcv",
        save_path / "shap",
        save_path / "tree",
    ]

    for target_folder in target_folders:
        make_archive_then_delete(target_folder)

    # tmpフォルダの削除
    if os.path.exists(save_path / "tmp") == True:
        shutil.rmtree(save_path / "tmp")

    return best_models, scores_df_list

def reader(
    csv_path,
    time_column_num,
    info_column_num,
    group_column_num,
    evaluate_sample_weight_column_num,
    training_sample_weight_column_num,
    X_column_num,
    y_column_num,
):
    """csvから学習用データを呼び出す関数

    Parameters
    ----------
    csv_path : str
        CSVファイルのパス
    time_column_num : int
        時系列情報の列数 0 or 1
    info_column_num : int
        学習に用いないIDなどの列数
    group_column_num : int
        各インスタンスのグループの列数 0 or 1
    evaluate_sample_weight_column_num : int
        評価時のsample_weightの列数 1
    training_sample_weight_column_num : int
        学習時のsample_weightの列数 1
    X_column_num : int
        説明変数の列数
    y_column_num : int
        目的変数の列数

    Returns
    -------
    X_raw_df : pd.DataFrame
        説明変数のDataFrame
    y_raw_df : pd.DataFrame
        目的変数のDataFrame
    time_raw_df : pd.DataFrame
        時系列のDataFrame
    group_raw_df : pd.DataFrame
        グループのDataFrame
    evaluate_sample_weight_raw_df : pd.DataFrame
        評価時のsample_weightのDataFrame
    training_sample_weight_raw_df : pd.DataFrame
        学習時のsample_weightのDataFrame
    info_raw_df : pd.DataFrame
        IDなどのDataFrame
    raw_data_df : pd.DataFrame
        元CSVファイルの全データ
    encoding : str
        chardetで判定したCSVファイルのエンコーディング

    """
    print(csv_path)
    csv_path = Path(csv_path)

    with open(csv_path, "rb") as f:
        char_dict = chardet.detect(f.read())
        confidence = char_dict["confidence"]
        encoding_ = char_dict["encoding"]
        print("csvファイルのencoding:{}  判定信頼度:{}".format(encoding_, confidence))

        if confidence <= 0.5:
            encoding = "shift_jisx0213"
        else:
            encoding = encoding_

    raw_data_df = pd.read_csv(open(csv_path, encoding=encoding))

    def replace_codes(x):
        code_regex = re.compile(
            "[!\"$%&'\\\\*,./:;<=>?@[\\]^`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
        )
        x = code_regex.sub("", x)
        x = x.strip()
        replace_list = [" ", "　", "/", ":", "-", "/n/r", "/r/n", "\n"]
        for code in replace_list:
            x = x.replace(code, "_")
        return x

    raw_data_df.rename(columns=replace_codes)
    columns_name = raw_data_df.columns
    new_columns_name = [replace_codes(i) for i in columns_name]
    raw_data_df.columns = new_columns_name

    print(time_column_num)
    print(info_column_num)
    print(group_column_num)
    print(evaluate_sample_weight_column_num)
    print(training_sample_weight_column_num)
    print(X_column_num)
    print(y_column_num)

    time_col = time_column_num
    info_col = time_column_num + info_column_num
    group_col = time_column_num + info_column_num + group_column_num
    evaluate_sample_weight_col = (
        time_column_num
        + info_column_num
        + group_column_num
        + evaluate_sample_weight_column_num
    )
    training_sample_weight_col = (
        time_column_num
        + info_column_num
        + group_column_num
        + evaluate_sample_weight_column_num
        + training_sample_weight_column_num
    )
    X_col = (
        time_column_num
        + info_column_num
        + group_column_num
        + evaluate_sample_weight_column_num
        + training_sample_weight_column_num
        + X_column_num
    )
    y_col = (
        time_column_num
        + info_column_num
        + group_column_num
        + evaluate_sample_weight_column_num
        + training_sample_weight_column_num
        + X_column_num
        + y_column_num
    )

    time_raw_df = raw_data_df.iloc[:, 0:time_col]
    info_raw_df = raw_data_df.iloc[:, time_col:info_col]
    group_raw_df = raw_data_df.iloc[:, info_col:group_col]
    evaluate_sample_weight_raw_df = raw_data_df.iloc[
        :, group_col:evaluate_sample_weight_col
    ]
    training_sample_weight_raw_df = raw_data_df.iloc[
        :, evaluate_sample_weight_col:training_sample_weight_col
    ]
    X_raw_df = raw_data_df.iloc[:, training_sample_weight_col:X_col]
    y_raw_df = raw_data_df.iloc[:, X_col:y_col]

    if time_raw_df.empty:
        time_raw_df = None
    if group_raw_df.empty:
        group_raw_df = None
    if evaluate_sample_weight_raw_df.empty:
        evaluate_sample_weight_raw_df = None
    if training_sample_weight_raw_df.empty:
        training_sample_weight_raw_df = None

    return (
        X_raw_df,
        y_raw_df,
        time_raw_df,
        group_raw_df,
        evaluate_sample_weight_raw_df,
        training_sample_weight_raw_df,
        info_raw_df,
        raw_data_df,
        encoding,
    )

def evaluater(
    model,
    X,
    y,
    X_scaler,
    y_scaler,
    X_names,
    y_names,
    datetime_df,
    group,
    sample_weight,
    save_path,
    save_name="",
    sv_type="regression",
    class_names_list=None,
    model_idx=0,
    *,
    scorer=None,
    splitter=None,
    splitter_args=None,
    n_jobs=-1,
):
    """評価関数

    Parameters
    ----------
    model : sklearn.estimator or sklearn.Pipeline or SearchCV
        評価したいモデル、パイプライン、GridSearchCVもしくはOptunaSearchCV
    X : numpy.ndarray or pd.DataFrame
        説明変数
    y : numpy.ndarray or pd.DataFrame
        目的変数
    X_scaler : scaler
        説明変数用のscaler
    y_scaler : scaler
        目的変数用のscaler
    X_names : list[str]
        説明変数の列名
    y_names : list[str]
        目的変数の列名
    datetime_df : np.ndarrays or pd.DataFrame
        時系列情報のデータフレーム
    group : np.ndarrays or pd.DataFrame
        グループのデータフレーム
    sample_weight : np.ndarrays or pd.DataFrame
        評価時のsample_weight
    save_path : str
        保存先フォルダのパス
    save_name : str, optional
        保存名, by default ""
    sv_type : str, optional
        回帰もしくは分類問題の指定 by default "regression"
    class_names_list : list[str], optional
        分類問題の各クラス名, by default None
    model_idx : int, optional
        モデルのインデックス, by default 0
    scorer : scorer, optional
        評価関数, by default None
    splitter : splitter, optional
        評価時のCV方法, by default None
    splitter_args : dict, optional
        CVの引数, by default None
    n_jobs : int, optional
        n_jobs, by default -1

    Returns
    -------
    each_scores_df : pd.DataFrame
        _評価結果のデータフレーム
    """    
    if splitter_args is None:
        splitter_args = {}

    y_raw = y_scaler.inverse_transform(y)
    y_ndim = len(pd.DataFrame(y).columns)

    scores_dict = {}

    model_name = model.__class__.__name__
    scaler, scaler_name = get_estimator(model, model_type="scaler")
    feature_engineering, feature_engineerings_name = get_estimator(
        model, model_type="feature_engineerings"
    )
    selector, selector_name = get_estimator(model, model_type="selector")
    estimator, estimator_name = get_estimator(
        model, model_type="estimator", remove_multioutput=False
    )

    sig = signature(estimator.fit)
    estimator_fit_params = sig.parameters

    if ("sample_weight" in estimator_fit_params) and (sample_weight is not None):
        fit_params = {}
        fit_params["estimator__sample_weight"] = sample_weight
    else:
        fit_params = None

    if splitter is not None:
        # eval_cv = splitter(**splitter_args).split(X, y, group)
        # eval_cv = splitter(**splitter_args)
        eval_cv = splitter

        if estimator_name in ["TabNetRegressor"]:
            X = np.array(X)
            y = np.reshape(np.array(y), (-1))
        else:
            X = np.array(X)
            y = np.array(y).reshape(-1, y_ndim)

        y_pred = cross_val_predict(
            model,
            X,
            y,
            groups=group,
            cv=eval_cv,
            fit_params=fit_params,
            n_jobs=n_jobs,
        )
    else:
        if fit_params is None:
            fit_params = {}

        eval_cv = None
        model.fit(X, y, **fit_params)
        y_pred = model.predict(X)

    y_pred_raw = y_scaler.inverse_transform(y_pred.reshape(-1, y_ndim))

    each_scores_dict = {}
    if model.__class__.__name__ == "Pipeline":

        scaler_idxs = [
            idx
            for idx, (name, class_) in enumerate(model.steps)
            if "scaler" in str(name)
        ]
        for idx, scaler_idx in enumerate(scaler_idxs):
            each_scores_dict["scaler{}".format(idx)] = model[scaler_idx]

        feature_eng_idxs = [
            idx
            for idx, (name, class_) in enumerate(model.steps)
            if "feature_engineerings" in str(name)
        ]
        for idx, fe_idx in enumerate(feature_eng_idxs):
            each_scores_dict["feature_engineerings{}".format(idx)] = model[fe_idx]

        selector_idxs = [
            idx
            for idx, (name, class_) in enumerate(model.steps)
            if "selector" in str(name)
        ]
        for idx, selector_idx in enumerate(selector_idxs):
            each_scores_dict["selector{}".format(idx)] = model[selector_idx]

        estimator_idxs = [
            idx
            for idx, (name, class_) in enumerate(model.steps)
            if "estimator" in str(name)
        ]

        for idx, estimator_idx in enumerate(estimator_idxs):
            if "SearchCV" in str(model[estimator_idx].__class__.__name__):
                each_scores_dict["estimator{}".format(idx)] = model[
                    estimator_idx
                ].estimator

                if hasattr(model[estimator_idx], "param_range") == True:
                    each_scores_dict["estimator_param{}".format(idx)] = model[
                        estimator_idx
                    ].param_range
                elif hasattr(model[estimator_idx], "param_grid") == True:
                    each_scores_dict["estimator_param{}".format(idx)] = model[
                        estimator_idx
                    ].param_grid
                elif hasattr(model[estimator_idx], "param_distributions") == True:
                    each_scores_dict["estimator_param{}".format(idx)] = model[
                        estimator_idx
                    ].param_distributions

                each_scores_dict["searcher{}".format(idx)] = model[
                    estimator_idx
                ].__class__.__name__
                each_scores_dict["searcher_cv{}".format(idx)] = model[
                    estimator_idx
                ].cv

            else:
                each_scores_dict["estimator{}".format(idx)] = model[estimator_idx]

    else:
        each_scores_dict["{model.__class__.__name__}"] = estimator_name

    each_scores_dict["scaler"] = scaler_name
    each_scores_dict["feature_engineerings"] = feature_engineerings_name
    each_scores_dict["selector_name"] = getattr(
        selector, "model_name", selector_name
    )
    each_scores_dict["selector"] = selector
    each_scores_dict["estimator_name"] = getattr(
        estimator, "model_name", estimator_name
    )
    each_scores_dict["estimator"] = estimator

    r2 = r2_score(y, y_pred)
    each_scores_dict["eval_method"] = str(eval_cv)

    each_scores_dict["r2_score"] = r2

    scores_dict[model_name] = each_scores_dict

    if len(y_names) == 1:
        each_mae = mean_absolute_error(y, y_pred)
        each_mae *= y_scaler.scale_[0]
        each_scores_dict[f"mae {y_names[0]}"] = each_mae

    else:
        for y_index, y_name in enumerate(y_names):
            each_mae = mean_absolute_error(y[:, y_index], y_pred[:, y_index])
            each_mae *= y_scaler.scale_[y_index]
            each_scores_dict[f"mae {y_name}"] = each_mae

    for y_index, y_name in enumerate(y_names):
        plt_title = "{} {}".format(y_name, model_name)
        each_save_path = save_path / y_name / "std"
        if class_names_list is None:
            class_names = None
        else:
            class_names = class_names_list[y_index]

        os.makedirs(each_save_path, exist_ok=True)
        each_save_name = f"{save_name}_{model_idx}_{estimator_name}"
        save_true_predict_result(
            y,
            y_pred,
            save_path=each_save_path,
            save_name=each_save_name,
            plt_title=plt_title,
            plt_label="predict",
            score_dict=each_scores_dict,
            sv_type=sv_type,
            labels=class_names,
        )

    for y_index, y_name in enumerate(y_names):
        plt_title = "{} {}".format(y_name, model_name)
        each_save_path = save_path / y_name
        if class_names_list is None:
            class_names = None
        else:
            class_names = class_names_list[y_index]

        os.makedirs(each_save_path, exist_ok=True)
        each_save_name = f"{save_name}_{model_idx}_{estimator_name}"
        save_true_predict_result(
            y_raw,
            y_pred_raw,
            save_path=each_save_path,
            save_name=each_save_name,
            plt_title=plt_title,
            plt_label="predict",
            score_dict=each_scores_dict,
            sv_type=sv_type,
            labels=class_names,
        )

    each_scores_df = pd.DataFrame([each_scores_dict])

    return each_scores_df

def explainer(
    evaluate_setting,
    model,
    X,
    y,
    X_scaler,
    y_scaler,
    X_names,
    y_names,
    score_df,
    save_path,
    info_df,
    datetime_df,
    class_names_list=None,
    model_idx=0,
):
    """_summary_

    Parameters
    ----------
    evaluate_setting : _type_
        _description_
    model : _type_
        _description_
    X : _type_
        _description_
    y : _type_
        _description_
    X_scaler : _type_
        _description_
    y_scaler : _type_
        _description_
    X_names : _type_
        _description_
    y_names : _type_
        _description_
    score_df : _type_
        _description_
    save_path : _type_
        _description_
    info_df : _type_
        _description_
    datetime_df : _type_
        _description_
    class_names_list : _type_, optional
        _description_, by default None
    model_idx : int, optional
        _description_, by default 0
    """    
    try:
        shap_except_explainers = evaluate_setting["shap"]
        print(f"score_df : {score_df}")
        print(f"model {model}")

        scaler, scaler_name = get_estimator(model, model_type="scaler")
        fe, fe_name = get_estimator(model, model_type="feature_engineerings")
        selector, selector_name = get_estimator(model, model_type="selector")
        _, estimator_name = get_estimator(model, model_type="estimator")

        estimator_index = [
            idx
            for idx, (name, class_) in enumerate(model.steps)
            if "estimator" in str(name)
        ][0]

        estimator = model[estimator_index]
        transformer = model[:-1]

        support = selector.get_support()
        feature_names = np.array(X_names)
        X_transformed_names = feature_names[support]
        X_transformed = transformer.transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=X_transformed_names)

        selector_name = selector_name.replace("FunctionTransformer", "")

        model_name = "{}_{}_{}".format(model_idx, selector_name, estimator_name)

        print("score_df")
        print(score_df)

        score_dict = score_df.to_dict()

        shap_values = explainer_shap(
            estimator=estimator,
            X=X_transformed_df,
            y=y,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
            X_selector=selector,
            X_names=X_names,
            X_transformed_names=X_transformed_names,
            y_names=y_names,
            score_dict=score_dict,
            theme_path=save_path,
            model_name=model_name,
            info_df=info_df,
            datetime_df=datetime_df,
            except_explainers=shap_except_explainers,
            class_names_list=class_names_list,
        )
    except:
        pass

    return

def explorer(X, y=None, save_path=Path(os.path.expanduser("~"))):
    if y is None:
        y = pd.DataFrame()
    if save_path is None:
        save_path = Path(os.path.expanduser("~"))

    shapiro_wilk(X, y, save_path)
    correlation_coefficient(X, y, save_path, method="pearson")
    correlation_coefficient(X, y, save_path, method="mvue")
    # correlation_coefficient(X, y, save_path, method='dj')
    n_X = len(X)
    n_y = len(y)
    if n_X + n_y <= 25:
        pairplot(X, y, save_path)
    else:
        pass
    return

def clustering(
    pipelines,
    X,
    y=None,
    *,
    X_names=None,
    y_names=None,
    info=None,
    info_names=None,
    group=None,
    save_path=Path(os.path.expanduser("~")),
):
    print(pipelines)
    """クラスタリング関数
        PCA, tSNE, UMAPなどのクラスタリング手法を適用する
    """
    save_path = save_path / "clustering"

    os.makedirs(save_path, exist_ok=True)

    if y is None:
        y = [None] * (np.array(X).shape[0])

    if group is not None:
        if len(group) == 0 :
            group = None

    X_np = np.array(X)
    y_np = np.array(y)
    info_np = np.array(info)
    group_np = np.array(group)

    # best_pipelines = []
    for idx, each_pipe in enumerate(pipelines):
        each_estimator, each_estimator_name = get_estimator(
            each_pipe, model_type="estimator"
        )
        print("clustering_name")
        print(each_estimator_name)

        if each_estimator_name == "GTM":
            if len(X.columns) <= 2:
                # GTMは内部でPCA(n_components=3)の処理を行うため、
                # 説明変数が2以下の場合はスキップgk
                continue

        if hasattr(each_estimator, "fit_transform") == True:
            X_scores = each_pipe.fit_transform(X_np)
            X_scores_df = pd.DataFrame(X_scores)
            X_scores_df.to_csv(save_path / f"{idx}_{each_estimator_name}.csv")

            plt.rcParams.update({"font.size": 20})

            fig_plain = plt.figure(figsize=(7.8, 7.8))
            ax_plain = fig_plain.add_subplot(1, 1, 1)
            ax_plain.scatter(X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30)
            ax_plain.set_title(each_estimator_name)

            xlim = ax_plain.get_xlim()
            ylim = ax_plain.get_ylim()

            max_xlim = max([abs(i) for i in xlim])
            max_ylim = max([abs(i) for i in ylim])
            max_lim = max(max_xlim, max_ylim)

            ax_plain.set_xlim(-max_lim, max_lim)
            ax_plain.set_ylim(-max_lim, max_lim)
            ax_plain.set_yticks = ax_plain.get_xticks()

            fig_plain.savefig(
                save_path
                / "{}_{}_{}.png".format(idx, each_estimator_name, "plain"),
                dpi=100,
                bbox_inches="tight",
            )

            # PCAなどの線形変換の場合、BIプロットを追加
            if hasattr(each_estimator, "components_"):
                pc0 = each_estimator.components_[0]
                pc1 = each_estimator.components_[1]
                cm_X = cm.get_cmap("tab10", pc0.shape[0])
                print(cm_X)
                fig_new = plt.figure(figsize=(7.8, 7.8))
                ax_plain = fig_new.add_subplot(1, 1, 1)
                ax_plain.scatter(X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30)
                ax_plain.set_title(each_estimator_name)

                xlim = ax_plain.get_xlim()
                ylim = ax_plain.get_ylim()

                max_xlim = max([abs(i) for i in xlim])
                max_ylim = max([abs(i) for i in ylim])
                max_lim = max(max_xlim, max_ylim)

                ax_plain.set_xlim(-max_lim, max_lim)
                ax_plain.set_ylim(-max_lim, max_lim)
                ax_plain.set_yticks = ax_plain.get_xticks()

                magnification_0 = 1.0 / max(pc0)
                magnification_1 = 1.0 / max(pc1)

                for i in range(pc0.shape[0]):
                    plt.arrow(
                        0,
                        0,
                        pc0[i] * magnification_0,
                        pc1[i] * magnification_1,
                        color=cm_X(i),
                        width=0.07,
                    )

                    plt.text(
                        pc0[i] * magnification_0 * 1.3,
                        pc1[i] * magnification_1 * 1.3,
                        X_names[i],
                        color=cm_X(i),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

                fig_new.savefig(
                    save_path
                    / "{}_{}_{}.png".format(idx, each_estimator_name, "BI"),
                    dpi=100,
                )  # bbox_inches="tight",

            if hasattr(each_estimator, "explained_variance_ratio_"):
                ev_ratio = each_estimator.explained_variance_ratio_
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                hist = np.r_[0.0, ev_ratio] + 1e-9
                ax1.bar(
                    np.arange(0, len(hist)),
                    hist,
                    color="m",
                    width=1.0,
                    alpha=0.5,
                    edgecolor="black",
                )
                ax1.set_xlabel("Principle components")
                ax1.set_ylabel("Contribution ratio")

                ax2 = ax1.twinx()
                line = np.r_[0.0, ev_ratio.cumsum()]
                obj1 = ax1.plot(
                    np.arange(0, len(line)), line, "o-", c="c", alpha=0.5
                )
                ax2.set_ylabel("Cumulative contribution rate")

                ax1.set_ylim(0.0, 1.0)
                ax2.set_ylim(0.0, 1.0)

                ax2.minorticks_on()
                ax2.grid(True, "both")

                fig.savefig(
                    save_path
                    / "{}_{}_{}.png".format(idx, each_estimator_name, "ev_ratio"),
                    dpi=100,
                )

                ev_ratio_df = pd.DataFrame(ev_ratio).T
                ev_ratio_df.columns = [
                    f"PCA {i}" for i in range(len(ev_ratio_df.columns))
                ]
                ev_ratio_df.to_csv(
                    save_path / f"{idx}_{each_estimator_name}_ev_ratio.csv",
                    encoding="shift_jisx0213",
                )

            n_X = X_np.shape[1]
            for each_target in range(n_X):
                cm_X = cm.get_cmap("jet", X_np.shape[0])
                rankdata_X = rankdata(X_np[:, each_target])
                rankdata_X = [int(i) for i in rankdata_X]

                fig_new = plt.figure(figsize=(7.8, 7.8))
                ax_plain = fig_new.add_subplot(1, 1, 1)
                ax_plain.scatter(X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30)
                ax_plain.set_title(each_estimator_name)

                xlim = ax_plain.get_xlim()
                ylim = ax_plain.get_ylim()

                max_xlim = max([abs(i) for i in xlim])
                max_ylim = max([abs(i) for i in ylim])
                max_lim = max(max_xlim, max_ylim)

                ax_plain.set_xlim(-max_lim, max_lim)
                ax_plain.set_ylim(-max_lim, max_lim)
                ax_plain.set_yticks = ax_plain.get_xticks()

                for i, each_X in zip(
                    range(X_scores.shape[0]), X_np[:, each_target]
                ):
                    if each_X is not None:
                        if isinstance(each_X, list) == True:
                            each_X = each_X[0]

                        each_X = round_significant_digits(each_X, 3)
                        plt.text(
                            X_scores[i, 0],
                            X_scores[i, 1],
                            str(each_X),
                            c=cm_X(rankdata_X[i]),
                        )

                if X_names is not None:
                    X_name = X_names[each_target]
                    plt.title(f"{each_estimator_name} - {X_name}")
                else:
                    X_name = f"X_{i}"

                X_folder_path = save_path / X_name
                os.makedirs(X_folder_path, exist_ok=True)

                fig_new.savefig(
                    X_folder_path
                    / "{}_{}_{}.png".format(idx, each_estimator_name, X_name),
                    dpi=100,
                )  # , bbox_inches="tight")
                plt.clf()

            if y is not None:
                if len(y_np.shape)  == 2:
                    y_np_shape = y_np.shape[1]
                elif len(y_np.shape) == 1:
                    y_np_shape = 1
                else:
                    ValueError("y_np.shape has no length.")

                n_y = y_np_shape
                for each_target in range(n_y):
                    cm_y = cm.get_cmap("jet", y_np.shape[0])
                    rankdata_y = rankdata(y_np[:, each_target])
                    rankdata_y = [int(i) for i in rankdata_y]

                    fig_new = plt.figure(figsize=(7.8, 7.8))
                    ax_plain = fig_new.add_subplot(1, 1, 1)
                    ax_plain.scatter(
                        X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30
                    )
                    ax_plain.set_title(each_estimator_name)

                    xlim = ax_plain.get_xlim()
                    ylim = ax_plain.get_ylim()

                    max_xlim = max([abs(i) for i in xlim])
                    max_ylim = max([abs(i) for i in ylim])
                    max_lim = max(max_xlim, max_ylim)

                    ax_plain.set_xlim(-max_lim, max_lim)
                    ax_plain.set_ylim(-max_lim, max_lim)
                    ax_plain.set_yticks = ax_plain.get_xticks()

                    for i, each_y in zip(
                        range(X_scores.shape[0]), y_np[:, each_target]
                    ):
                        if each_y is not None:
                            if isinstance(each_y, list) == True:
                                each_y = each_y[0]

                            each_y = round_significant_digits(each_y, 3)
                            plt.text(
                                X_scores[i, 0],
                                X_scores[i, 1],
                                str(each_y),
                                c=cm_y(rankdata_y[i]),
                            )

                        if y_names is not None:
                            y_name = y_names[each_target]
                            plt.title(f"{each_estimator_name} - {y_name}")
                        else:
                            y_name = f"y_{i}"

                        y_folder_path = save_path / y_name
                        os.makedirs(y_folder_path, exist_ok=True)
                    fig_new.savefig(
                        y_folder_path
                        / "{}_{}_{}.png".format(idx, each_estimator_name, y_name),
                        dpi=100,
                    )  # , bbox_inches="tight")
                    plt.clf()

            if info is not None:
                n_info = info_np.shape[1]
                for each_target in range(n_info):
                    fig_new = plt.figure(figsize=(7.8, 7.8))
                    ax_plain = fig_new.add_subplot(1, 1, 1)
                    ax_plain.scatter(
                        X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30
                    )
                    ax_plain.set_title(each_estimator_name)

                    xlim = ax_plain.get_xlim()
                    ylim = ax_plain.get_ylim()

                    max_xlim = max([abs(i) for i in xlim])
                    max_ylim = max([abs(i) for i in ylim])
                    max_lim = max(max_xlim, max_ylim)

                    ax_plain.set_xlim(-max_lim, max_lim)
                    ax_plain.set_ylim(-max_lim, max_lim)
                    ax_plain.set_yticks = ax_plain.get_xticks()

                    a_info = info_np[:, each_target]

                    n_unique_each_info = len(set(a_info))
                    cm_info = cm.get_cmap("jet", n_unique_each_info)
                    le = LabelEncoder()
                    unique_info = le.fit_transform(a_info)

                    for i, each_info, unique_idx in zip(
                        range(X_scores.shape[0]),
                        info_np[:, each_target],
                        unique_info,
                    ):

                        if each_info is not None:
                            if isinstance(each_info, list) == True:
                                each_info = each_info[0]

                            plt.text(
                                X_scores[i, 0],
                                X_scores[i, 1],
                                str(each_info),
                                c=cm_info(unique_idx),
                            )

                    if info_names is not None:
                        info_name = info_names[each_target]
                    else:
                        info_name = f"info_{each_target}"

                    plt.title(f"{each_estimator_name} - {info_name}")
                    info_folder_path = save_path / info_name
                    os.makedirs(info_folder_path, exist_ok=True)

                    fig_new.savefig(
                        info_folder_path
                        / "{}_{}_{}.png".format(
                            idx, each_estimator_name, info_name
                        ),
                        dpi=100,
                    )  # , bbox_inches="tight")
                    plt.clf()

            if group is not None:
                n_group = 1
                for each_target in range(n_group):
                    fig_new = plt.figure(figsize=(7.8, 7.8))
                    ax_plain = fig_new.add_subplot(1, 1, 1)
                    ax_plain.scatter(
                        X_scores[:, 0], X_scores[:, 1], c="dimgray", s=30
                    )
                    ax_plain.set_title(each_estimator_name)

                    xlim = ax_plain.get_xlim()
                    ylim = ax_plain.get_ylim()

                    max_xlim = max([abs(i) for i in xlim])
                    max_ylim = max([abs(i) for i in ylim])
                    max_lim = max(max_xlim, max_ylim)

                    ax_plain.set_xlim(-max_lim, max_lim)
                    ax_plain.set_ylim(-max_lim, max_lim)
                    ax_plain.set_yticks = ax_plain.get_xticks()

                    a_group = group_np[:]
                    n_unique_each_group = len(set(a_group))
                    cm_group = cm.get_cmap("jet", n_unique_each_group)
                    le = LabelEncoder()
                    unique_group = le.fit_transform(a_group)

                    for i, each_group, unique_idx in zip(
                        range(X_scores.shape[0]),
                        group_np[:],
                        unique_group,
                    ):

                        if each_group is not None:
                            if isinstance(each_group, list) == True:
                                each_group = each_group[0]

                            plt.text(
                                X_scores[i, 0],
                                X_scores[i, 1],
                                str(each_group),
                                c=cm_group(unique_idx),
                            )

                    group_folder_path = save_path / "group"
                    os.makedirs(group_folder_path, exist_ok=True)

                    fig_new.savefig(
                        group_folder_path
                        / f"{idx}_{each_estimator_name}_group.png",
                        dpi=100,
                    )  # , bbox_inches="tight")

                    plt.clf()

        else:
            each_pipe.fit(X)
            if hasattr(each_pipe, "best_estimator_") == True:
                best_pipe = each_pipe.best_estimator_
            else:
                best_pipe = each_pipe

            # best_pipelines.append(best_pipe)

            if hasattr(each_estimator, "labels_") == True:
                X_scores = each_estimator.labels_
                X_scores_df = pd.DataFrame(X_scores)
                X_scores_df.columns = ["DBSCAN"]
                X_df = pd.concat([pd.DataFrame(X), X_scores_df], ignore_index=True)
                X_df.to_csv(
                    save_path / "{}_{}.csv".format(idx, each_estimator_name),
                    index=False,
                    encoding="shift_jisx0213",
                )
            elif each_estimator.__class__.__name__ == "GTM":
                responsibilities = best_pipe.responsibility(input_dataset)
                means, modes = best_pipe.means_modes(input_dataset)

                # plot the mean of responsibilities
                plt.rcParams["font.size"] = 18
                plt.figure(figsize=figure.figaspect(1))
                plt.scatter(means[:, 0], means[:, 1], c=color)
                plt.ylim(-1.1, 1.1)
                plt.xlim(-1.1, 1.1)
                plt.xlabel("z1 (mean)")
                plt.ylabel("z2 (mean)")
                plt.title(each_estimator_name)
                plt.savefig(
                    save_path / "{}_{}_mean.png".format(idx, each_estimator_name),
                    dpi=100,
                    bbox_inches="tight",
                )

                # plot the mode of responsibilities
                plt.figure(figsize=figure.figaspect(1))
                plt.scatter(modes[:, 0], modes[:, 1], c=color)
                plt.ylim(-1.1, 1.1)
                plt.xlim(-1.1, 1.1)
                plt.xlabel("z1 (mode)")
                plt.ylabel("z2 (mode)")
                plt.title(each_estimator_name)
                plt.savefig(
                    save_path / "{}_{}_mode.png".format(idx, each_estimator_name),
                    dpi=100,
                    bbox_inches="tight",
                )

            else:
                print()

        if each_estimator_name == "FunctionTransformer":
            names = list(copy.deepcopy(X_names))
            if y is not None:
                scores_ = np.concatenate([y, X_scores], axis=1)
                names.extend(list(y_names))
            else:
                scores_ = X_scores

            print(scores_.shape)
            print(names)

            combinations = list(
                itertools.combinations(np.arange(scores_.shape[1]), 2)
            )

            for each_combination in combinations:
                X_idx = each_combination[0]
                y_idx = each_combination[1]
                each_fig = plt.figure(figsize=(7.8, 7.8))
                each_ax = each_fig.add_subplot(1, 1, 1)
                each_ax.scatter(
                    scores_[:, X_idx], scores_[:, y_idx], c="dimgray", s=30
                )
                each_ax.set_title(f"{names[X_idx]}-{names[y_idx]}")

                xlim = each_ax.get_xlim()
                ylim = each_ax.get_ylim()

                max_xlim = max([abs(i) for i in xlim])
                max_ylim = max([abs(i) for i in ylim])
                max_lim = max(max_xlim, max_ylim)

                each_ax.set_xlim(-max_lim, max_lim)
                each_ax.set_ylim(-max_lim, max_lim)
                each_ax.set_yticks = each_ax.get_xticks()

                combination_path = save_path / "combination"
                os.makedirs(combination_path, exist_ok=True)

                each_fig.savefig(
                    combination_path / f"{names[X_idx]}-{names[y_idx]}.png",
                    dpi=100,
                )  # , bbox_inches="tight")

                continue

                if group is not None:
                    # break

                    print("group", group)
                    n_group = 1
                    for each_target in range(n_group):
                        each_fig_new = copy.deepcopy(each_fig)
                        a_group = group_np[:]
                        n_unique_each_group = len(set(a_group))
                        cm_group = cm.get_cmap("jet", n_unique_each_group)
                        le = LabelEncoder()
                        unique_group = le.fit_transform(a_group)

                        for i, each_group, unique_idx in zip(
                            range(X_scores.shape[0]),
                            group_np[:],
                            unique_group,
                        ):
                            print(each_group)
                            if each_group is not None:
                                if isinstance(each_group, list) == True:
                                    each_group = each_group[0]

                                plt.text(
                                    scores_[:, X_idx],
                                    scores_[:, y_idx],
                                    str(each_group),
                                    c=cm_group(unique_idx),
                                )

                        combination_group_folder_path = combination_path / "group"
                        os.makedirs(combination_group_folder_path, exist_ok=True)

                        each_fig_new.savefig(
                            combination_group_folder_path
                            / f"{names[X_idx]}-{names[y_idx]}_group.png",
                            dpi=100,
                        )  # , bbox_inches="tight")

                        plt.clf()

    best_pipelines = [
        i.best_estimator_ if hasattr(i, "best_estimator_") else i
        for idx, i in enumerate(pipelines)
    ]
    param_path = save_path / "param"

    [
        save_params(
            i, param_path, save_name=f"{idx}_pipeline", model_type="pipeline"
        )
        for idx, i in enumerate(best_pipelines)
    ]

    cloudpickle_path = save_path / "cloudpickle"

    [
        save_model(i, cloudpickle_path, save_name=f"{idx}", model_type="pipeline")
        for idx, i in enumerate(best_pipelines)
    ]
    [
        save_model(i, cloudpickle_path, save_name=f"{idx}", model_type="scaler")
        for idx, i in enumerate(best_pipelines)
    ]
    [
        save_model(
            i,
            cloudpickle_path,
            save_name=f"{idx}_f_engineer",
            model_type="feature_engineerings",
        )
        for idx, i in enumerate(best_pipelines)
    ]
    [
        save_model(i, cloudpickle_path, save_name=f"{idx}", model_type="selector")
        for idx, i in enumerate(best_pipelines)
    ]
    [
        save_model(
            i,
            cloudpickle_path,
            save_name=f"{idx}_estimator",
            model_type="estimator",
        )
        for idx, i in enumerate(best_pipelines)
    ]

    return

def main_gui():
    """PySimpleGUIを用いた画面作成＆学習実行

    """

    warnings.simplefilter("ignore")  # , 'DataConversionWarning')
    sg.theme("DarkBrown6")

    X_scaler_dict = {
        "FunctionTransformer": {
            "display_name": "なし",
            "instance": FunctionTransformer(),
            "default": False,
            "argument": "",
        },
        "StandardScaler": {
            "display_name": "正規化",
            "instance": StandardScaler(),
            "default": True,
            "argument": "",
        },
        "MinMaxScaler": {
            "display_name": "標準化",
            "instance": MinMaxScaler(),
            "default": False,
            "argument": "",
        },
        "ImportanceScaler": {
            "display_name": "適応的標準化",
            "instance": ImportanceScaler(),
            "default": False,
            "argument": "",
        },
    }

    y_scaler_dict = {
        "FunctionTransformer": {
            "display_name": "なし",
            "instance": FunctionTransformer(),
            "default": False,
            "argument": "",
        },
        "StandardScaler": {
            "display_name": "正規化",
            "instance": StandardScaler(),
            "default": True,
            "argument": "",
        },
        "MinMaxScaler": {
            "display_name": "標準化",
            "instance": MinMaxScaler(),
            "default": False,
            "argument": "",
        },
        "ImportanceScaler": {
            "display_name": "適応的標準化",
            "instance": ImportanceScaler(),
            "default": False,
            "argument": "",
        },
    }

    missing_data_dict = {
        "SimpleImputer": {
            "display_name": "SimpleImputer",
            "instance": SimpleImputer(),
            "default": True,
            "argument": "strategy:'mean'",
        },
        "KNNImputer": {
            "display_name": "KNNImputer",
            "instance": KNNImputer(),
            "default": False,
            "argument": "strategy:'mean'",
        },
        "IterativeImputer": {
            "display_name": "IterativeImputer",
            "instance": IterativeImputer(),
            "default": False,
            "argument": "",
        },
    }

    feature_engineering_dict = {
        "PolynomialFeatures": {
            "display_name": "交差項",
            "instance": PolynomialFeatures(),
            "default": False,
            "argument": "",
        },
        "morgan_fingerprint": {
            "display_name": "morgan_fingerprint",
            "instance": "morgan_fingerprint",
            "default": False,
            "argument": "",
        },
        "lagfeature": {
            "display_name": "lagfeature",
            "instance": "lagfeature()",
            "default": False,
            "argument": "",
        },
    }

    loaded_selectors = load_selectors(ml_type="regression")

    randint = 42

    op_splitter_dict = {
        "KFold_shuffle": {
            "display_name": "KFold_shuffle",
            "instance": KFold(shuffle=True),
            "default": True,
            "argument": f"n_splits:5, shuffle:True, random_state:{randint}",
        },
        "KFold_not_shuffle": {
            "display_name": "KFold_not_shuffle",
            "instance": KFold(shuffle=False),
            "default": False,
            "argument": "n_splits:5, shuffle:False",
        },
        "LeaveOneOut": {
            "display_name": "LeaveOneOut",
            "instance": LeaveOneOut(),
            "default": False,
            "argument": "",
        },
        "LeavePOut": {
            "display_name": "LeavePOut",
            "instance": LeavePOut(p=2),
            "default": False,
            "argument": "p:2",
        },
        "GroupKFold": {
            "display_name": "GroupKFold",
            "instance": GroupKFold(),
            "default": False,
            "argument": "n_splits:5",
        },
        "LeaveOneGroupOut": {
            "display_name": "LeaveOneGroupOut",
            "instance": LeaveOneGroupOut(),
            "default": False,
            "argument": "",
        },
        "TimeSeriesSplit": {
            "display_name": "TimeSeriesSplit",
            "instance": TimeSeriesSplit(),
            "default": False,
            "argument": "n_splits:5, max_train_size=None, test_size=None, gap=0",
        },
        "WalkForwardValidation": {
            "display_name": "WalkForwardValidation",
            "instance": WalkForwardValidation(),
            "default": False,
            "argument": "train_size:7, test_size:1, gap:0",
        },
    }

    searcher_dict = {
        "GridSearchCV": {
            "display_name": "GridSearchCV",
            "class": GridSearchCV,
            "default": False,
            "argument": "scoring:'neg_mean_squared_error'",
        },
        "OptunaSearchCV": {
            "display_name": "OptunaSearchCV",
            "class": OptunaSearchCV,
            "default": True,
            "argument": f"n_trials:20, timeout:600, scoring:'neg_mean_squared_error', random_state:{randint}",
        },
        "None": {
            "display_name": "No tuning",
            "class": FunctionTransformer,
            "default": False,
            "argument": "",
        },
    }

    ev_cv_splitter_dict = {
        "KFold_shuffle": {
            "display_name": "KFold_shuffle",
            "instance": KFold(shuffle=True),
            "default": True,
            "argument": f"n_splits:5, shuffle:True, random_state:{randint}",
        },
        "KFold_not_shuffle": {
            "display_name": "KFold_not_shuffle",
            "instance": KFold(shuffle=False),
            "default": False,
            "argument": "n_splits:5, shuffle:False",
        },
        "LeaveOneOut": {
            "display_name": "LeaveOneOut",
            "instance": LeaveOneOut(),
            "default": False,
            "argument": "",
        },
        "LeavePOut": {
            "display_name": "LeavePOut",
            "instance": LeavePOut(p=2),
            "default": False,
            "argument": "p:2",
        },
        "GroupKFold": {
            "display_name": "GroupKFold",
            "instance": GroupKFold(),
            "default": False,
            "argument": "n_splits:5",
        },
        "LeaveOneGroupOut": {
            "display_name": "LeaveOneGroupOut",
            "instance": LeaveOneGroupOut(),
            "default": False,
            "argument": "",
        },
        "TimeSeriesSplit": {
            "display_name": "TimeSeriesSplit",
            "instance": TimeSeriesSplit(),
            "default": False,
            "argument": "n_splits:5, max_train_size=None, test_size=None, gap=0",
        },
        "WalkForwardValidation": {
            "display_name": "WalkForwardValidation",
            "instance": WalkForwardValidation(),
            "default": False,
            "argument": "train_size:7, test_size:1, gap:0",
        },
    }

    ev_dcv_splitter_dict = {
        "KFold_shuffle": {
            "display_name": "KFold_shuffle",
            "instance": KFold(shuffle=True),
            "default": True,
            "argument": f"n_splits:5, shuffle:True, random_state:{randint}",
        },
        "KFold_not_shuffle": {
            "display_name": "KFold_not_shuffle",
            "instance": KFold(shuffle=False),
            "default": False,
            "argument": "n_splits:5, shuffle:False",
        },
        "LeaveOneOut": {
            "display_name": "LeaveOneOut",
            "instance": LeaveOneOut(),
            "default": False,
            "argument": "",
        },
        "LeavePOut": {
            "display_name": "LeavePOut",
            "instance": LeavePOut(p=2),
            "default": False,
            "argument": "",
        },
        "GroupKFold": {
            "display_name": "GroupKFold",
            "instance": GroupKFold(),
            "default": False,
            "argument": "n_splits:5",
        },
        "LeaveOneGroupOut": {
            "display_name": "LeaveOneGroupOut",
            "instance": LeaveOneGroupOut(),
            "default": False,
            "argument": "",
        },
        "TimeSeriesSplit": {
            "display_name": "TimeSeriesSplit",
            "instance": TimeSeriesSplit(),
            "default": False,
            "argument": "n_splits:5, max_train_size=None, test_size=None, gap=0",
        },
        "WalkForwardValidation": {
            "display_name": "WalkForwardValidation",
            "instance": WalkForwardValidation(),
            "default": False,
            "argument": "train_size:7, test_size:1, gap:0",
        },
    }

    # GUI - main 
    def create_main_layout():
        # FilesBrowse はenable_events=Trueが有効ではないため、[sg.Input(key='_FILEBROWSE_', enable_events=True, visible=False)]を用いた
        # refer https://github.com/PySimpleGUI/PySimpleGUI/issues/850
        # target='_FILEBROWSE_',

        layout_reader = [
            [
                sg.FilesBrowse(
                    "CSV読込", key="csv_path", file_types=(("CSV ファイル", "*.csv"),)
                ),
                sg.InputText(
                    r"C:\Users\11665307\Desktop\diabetes.csv",
                    size=(150, 1),
                    key="csv_path_",
                    enable_events=True,
                ),
            ],
            [
                sg.Text("時系列", size=(25, 1), justification="center"),
                sg.Text("info", size=(25, 1), justification="center"),
                sg.Text("グループ", size=(25, 1), justification="center"),
                sg.Text("評価時重み", size=(25, 1), justification="center"),
                sg.Text("学習時重み", size=(25, 1), justification="center"),
                sg.Text("説明変数", size=(25, 1), justification="center"),
                sg.Text("目的変数", size=(25, 1), justification="center"),
            ],
            [
                sg.InputText(
                    "0",
                    size=(25, 1),
                    key="timeseries_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "0",
                    size=(25, 1),
                    key="info_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "0",
                    size=(25, 1),
                    key="group_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "0",
                    size=(25, 1),
                    key="evaluate_sample_weight_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "0",
                    size=(25, 1),
                    key="training_sample_weight_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "10",
                    size=(25, 1),
                    key="input_num",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "1",
                    size=(25, 1),
                    key="output_num",
                    justification="center",
                    enable_events=True,
                ),
            ],
            [
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="timeseries_column_list",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="info_column_list",
                    justification="center",
                    enable_events=False,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="group_column_list",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="evaluate_sample_weight_column_list",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="training_sample_weight_column_list",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="input_column_list",
                    justification="center",
                    enable_events=True,
                ),
                sg.InputText(
                    "",
                    size=(25, 1),
                    key="output_column_list",
                    justification="center_column_list",
                    enable_events=True,
                ),
            ],]

        frame_reader = sg.Frame("", layout_reader)  # , size=(500,300))

        layout_theme = [
            [sg.Text("テーマ名", size=(15, 1)), sg.InputText("", key="theme_name", size=(60,1))],
        ]

        layout_supervised = [
            [
                sg.Text("教師あり学習", size=(15, 1)),
                sg.Checkbox(
                    "Supervised", default=True, key="supervised", enable_events=True
                ),
                sg.Radio(
                    "Regression",
                    default=True,
                    key="sv_type__regression",
                    group_id="ml_type",
                    enable_events=True,
                ),
                sg.Radio(
                    "Classification",
                    default=False,
                    key="sv_type__classification",
                    group_id="ml_type",
                    enable_events=True,
                ),
            ],
        ]

        layout_clustering = [
            [
                sg.Text("教師なし学習", size=(15, 1)),
                sg.Checkbox(
                    "Clustering & Visualization",
                    default=False,
                    key="us_type__clustering",
                    enable_events=True,
                ),
            ]
        ]

        layout_training_predict = [
            [
                sg.Radio(
                    "学習",
                    default=True,
                    key="training",
                    group_id="train_predict",
                    enable_events=True,
                ),
                sg.Radio(
                    "学習済み",
                    default=False,
                    key="trained",
                    group_id="train_predict",
                    enable_events=True,
                ),
            ]
        ]

        X_scaler_set = [
            [
                sg.Radio(
                    text=value["display_name"],
                    key=f"X_scaler__{key}",
                    default=value["default"],
                    group_id="X_scaler",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"], key=f"X_scaler_args__{key}", size=(20, 1)
                ),
            ]
            for key, value in X_scaler_dict.items()
        ]

        y_scaler_set = [
            [
                sg.Radio(
                    text=value["display_name"],
                    key=f"y_scaler__{key}",
                    default=value["default"],
                    group_id="y_scaler",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"], key=f"y_scaler_args__{key}", size=(20, 1)
                ),
            ]
            for key, value in y_scaler_dict.items()
        ]

        missing_data_set = [
            [
                sg.Radio(
                    text=value["display_name"],
                    key=f"missing_data__{key}",
                    default=value["default"],
                    group_id="missing_data",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"],
                    key=f"missing_data_args__{key}",
                    size=(20, 1),
                ),
            ]
            for key, value in missing_data_dict.items()
        ]

        feature_engineering_set = [
            [
                sg.InputText(
                    value["display_name"],
                    key="feature_engineering_order__{key}",
                    size=(5, 1),
                    justification="center",
                ),
                sg.Text(key, size=(12, 1)),
                sg.InputText(
                    "",
                    key="feature_engineering_args__{}".format(value["argument"]),
                    size=(20, 1),
                ),
            ]
            for key, value in feature_engineering_dict.items()
        ]

        layout_scaler = [
            [sg.Text("X_標準化", size=(15, 1)), sg.Text("**args", size=(15, 1))],
            *X_scaler_set,
            [sg.Text("y_標準化", size=(15, 1)), sg.Text("**args", size=(15, 1))],
            *y_scaler_set,
        ]

        layout_missing_data = [
            [sg.Text("欠損値処理", size=(15, 1)), sg.Text("**args", size=(15, 1))],
            *missing_data_set,
        ]

        layout_feature_engineering = [
            [
                sg.Text("処理順", size=(15, 1)),
                sg.Text("処理種", size=(15, 1)),
                sg.Text("**args", size=(15, 1)),
            ],
            *feature_engineering_set,
            [
                sg.FilesBrowse(
                    "処理読込", key="file_path", file_types=(("pyファイル", "*.py"),)
                ),
                sg.Text("from", size=(5, 1)),
                sg.InputText(size=(10, 1)),
                sg.Text("import", size=(5, 1)),
                sg.InputText("例) StandardScaler", key="user_defined_class"),
                sg.Text(".set_params(", size=(10, 1)),
                sg.InputText("例) copy:True, with_mean:True,", key="user_defined_arg"),
                sg.Text(")", size=(2, 1)),
            ],
        ]

        tg_pre = sg.TabGroup(
            [
                [
                    sg.Tab("標準化", layout_scaler),
                    sg.Tab("欠損値", layout_missing_data),
                    sg.Tab("特徴量エンジニアリング", layout_feature_engineering),
                ]
            ]
        )

        tab_preprocessing = sg.Tab("前処理", [[tg_pre]])  # , size=(500,300))

        selector_set = [
            [
                sg.Checkbox(
                    text=selector.model_name,
                    key=f"selector__{selector.model_name}",
                    default=selector.default,
                    size=(15, 1),
                ),
                sg.InputText(
                    "", key=f"selector_args__{selector.model_name}", size=(20, 1)
                ),
            ]
            for selector in loaded_selectors
        ]

        layout_selector = [
            [sg.Text("変数選択", size=(15, 1)), sg.Text("**args", size=(15, 1))],
            *selector_set,
            [
                sg.FilesBrowse(
                    "処理読込", key="file_path", file_types=(("pyファイル", "*.py"),)
                ),
                sg.Text("from", size=(5, 1)),
                sg.InputText(size=(10, 1), key="user_defined_selector_py"),
                sg.Text("import", size=(5, 1)),
                sg.InputText("例) VarianceThreshold", key="user_defined_selector_class"),
                sg.Text(".set_params(", size=(10, 1)),
                sg.InputText("", key="user_defined_selector_args"),
                sg.Text(")", size=(2, 1)),
            ],
        ]

        tab_selector = sg.Tab("変数選択", layout_selector)

        op_splitter_set = [
            [
                sg.Radio(
                    text=key,
                    key=f"optimize_cv_splitter__{key}",
                    default=value["default"],
                    group_id="optimize_cv_splitter",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"],
                    key=f"optimize_cv_splitter_args__{key}",
                    size=(50, 1),
                ),
            ]
            for key, value in op_splitter_dict.items()
        ]

        searcher_set = [
            [
                sg.Radio(
                    text=value["display_name"],
                    key=f"optimize_searcher__{key}",
                    default=value["default"],
                    group_id="optimize_searcher",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"],
                    key=f"optimize_searcher_args__{key}",
                    size=(50, 1),
                ),
            ]
            for key, value in searcher_dict.items()
        ]

        """
        optimize_score_regrssion_dict = {
                        'default(r2_score)' : ['None',   True],
                        'MAE'               : ['neg_mean_absolter_error',   False],
                        'MSE'               : ['neg_mean_squared_error',   False],
                        'r2'                : ['r2_score',    False],
                        'r2lm'              : ['r2lm_score',  False],
                        'User-Defined'      : ['',  False],
                    }
        """

        optimize_score_regrssion_list = [
            "None",
            "r2_score",
            "r2lm_score",
            "neg_mean_absolter_error",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
        ]

        optimize_score_classification_list = [
            "None",
            "accuracy",
            "average_precision",
            "f1",
            "roc_auc",
            "recall",
        ]

        ev_cv_splitter_set = [
            [
                sg.Radio(
                    text=key,
                    key=f"evaluate_cv_splitter__{key}",
                    default=value["default"],
                    group_id="evaluate_cv_splitter",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"],
                    key=f"evaluate_cv_splitter_args__{key}",
                    size=(50, 1),
                ),
            ]
            for key, value in ev_cv_splitter_dict.items()
        ]

        ev_dcv_splitter_set = [
            [
                sg.Radio(
                    text=key,
                    key=f"evaluate_dcv_splitter__{key}",
                    default=value["default"],
                    group_id="evaluate_dcv_splitter",
                    size=(15, 1),
                ),
                sg.InputText(
                    value["argument"],
                    key=f"evaluate_dcv_splitter_args__{key}",
                    size=(50, 1),
                ),
            ]
            for key, value in ev_dcv_splitter_dict.items()
        ]

        tabgroup_evaluater_setting = sg.TabGroup(
            [
                [
                    sg.Tab(
                        "Hold-out",
                        [
                            [
                                sg.Text("Test_ratio (0-1):", size=(15, 1)),
                                sg.InputText("", key="test_ratio", size=(4, 1)),
                            ]
                        ],
                        key="ev_tab__holdout",
                    )
                ],
                [
                    sg.Tab(
                        "Double-Cross-Validation",
                        [
                            [sg.Text("DCV-Splitter:", size=(6, 1))],
                            # [ sg.Radio('optimizeと同じ', key='evaluate_dcv__as_optimize',default=True, group_id='evaluate_dcv_splitter')],
                            *ev_dcv_splitter_set,
                            [
                                sg.FilesBrowse(
                                    "処理読込",
                                    key="file_path",
                                    file_types=(("pyファイル", "*.py"),),
                                ),
                                sg.Text("from", size=(5, 1)),
                                sg.InputText(
                                    size=(10, 1), key="ev_user_defined_dcv_splitter_py"
                                ),
                                sg.Text("import", size=(5, 1)),
                                sg.InputText(
                                    "例) StratifiedKFold",
                                    key="ev_user_defined_dcv_splitter_class",
                                ),
                            ],
                        ],
                        key="ev_tab__doublecrossvalidation",
                    )
                ],
                [
                    sg.Tab(
                        "Cross-Validation",
                        [
                            [sg.Text("CV-Splitter:", size=(6, 1))],
                            # [ sg.Radio('optimizeと同じ', key='evaluate_cv__as_optimize',default=True, group_id='evaluate_cv_splitter')],
                            *ev_cv_splitter_set,
                            [
                                sg.FilesBrowse(
                                    "処理読込",
                                    key="file_path",
                                    file_types=(("pyファイル", "*.py"),),
                                ),
                                sg.Text("from", size=(5, 1)),
                                sg.InputText(
                                    size=(10, 1), key="ev_user_defined_cv_splitter_py"
                                ),
                                sg.Text("import", size=(5, 1)),
                                sg.InputText(
                                    "例) StratifiedKFold",
                                    key="ev_user_defined_cv_splitter_class",
                                ),
                            ],
                        ],
                        key="ev_tab__crossvalidation",
                    )
                ],
            ],
            key="ev_tabgroup__h_cv_dcv",
        )

        layout_evaluater = [
            [
                sg.Text("機械学習 evaluate", size=(15, 1)),
                sg.Checkbox(
                    "Holdout-evaluate", default=False, key="ml_holdout_evaluate"
                ),
                sg.Checkbox(
                    "Double-Cross-validation",
                    default=True,
                    key="ml_double_cross_validate",
                ),
                sg.Checkbox(
                    "CrossOpt & CrossEval (deprecated)",
                    default=False,
                    key="ml_cross_opt_evaluate",
                ),
            ],
            [tabgroup_evaluater_setting],
        ]

        tab_evaluater = sg.Tab("評価手法", layout_evaluater)  # , size=(500,300))

        tabgroup_modeller_setting = sg.TabGroup(
            [
                [
                    sg.Tab(
                        "Hold-out",
                        [
                            [
                                sg.Text("Validation_ratio (0-1):", size=(15, 1)),
                                sg.InputText("", key="validation_ratio", size=(4, 1)),
                            ]
                        ],
                        key="op_tab__holdout",
                    )
                ],
                [
                    sg.Tab(
                        "Cross-Validation",
                        [
                            [sg.Text("CV-Splitter:", size=(6, 1))],
                            *op_splitter_set,
                            [
                                sg.FilesBrowse(
                                    "処理読込",
                                    key="file_path",
                                    file_types=(("pyファイル", "*.py"),),
                                ),
                                sg.Text("from", size=(5, 1)),
                                sg.InputText(
                                    size=(10, 1), key="op_user_defined_splitter_py"
                                ),
                                sg.Text("import", size=(5, 1)),
                                sg.InputText(
                                    "例) StratifiedKFold",
                                    key="op_user_defined_splitter_class",
                                ),
                            ],
                        ],
                        key="op_tab__crossvalidation",
                    )
                ],
            ],
            key="op_tabgroup__h_cv",
        )

        layout_modeller = [
            [
                sg.Checkbox("標準モデル", default=True, key="load_model__load_all_models"),
                sg.Text("standard_models", size=(10, 1)),
                sg.Button("モデル選択", size=(10, 1), key="select_standard_models"),
            ],
            [
                sg.Checkbox("TabNet", default=False, key="load_model__load_tabnet"),
                sg.Text("TabNet", size=(20, 1)),
                sg.Checkbox("automl", default=False, key="load_model__load_automl"),
                sg.InputText("", size=(20, 1)),
                sg.Checkbox(
                    "追加モデル", default=False, key="load_model__user_defined_model"),
                sg.FolderBrowse("フォルダ選択", key="modeller_path"),
                sg.InputText("", size=(20, 1)),
            ],
            [
                sg.Checkbox(
                    "MultiOutputRegressor", default=False, key="MultiOutputRegressor"
                )
            ],
            [sg.Text("Searcher:", size=(12, 1))],
            *searcher_set,
            [
                sg.Text("Scorer:", size=(12, 1)),
                sg.Combo(
                    optimize_score_regrssion_list,
                    default_value=optimize_score_regrssion_list[0],
                    key="optimize_regression_scorer",
                    size=(10, 6),
                    visible=True,
                ),
                sg.Combo(
                    optimize_score_classification_list,
                    default_value=optimize_score_classification_list[0],
                    key="optimize_classification_scorer",
                    size=(10, 6),
                    visible=False,
                ),
            ],
            [
                sg.Text("hyperparameter optimize", size=(15, 1)),
                sg.Checkbox(
                    "Holdout-optimize",
                    default=False,
                    key="ml_holdout_optimize",
                    visible=True,
                    enable_events=True,
                ),
                sg.Checkbox(
                    "Cross-optimize",
                    default=True,
                    key="ml_cross_optimize",
                    visible=True,
                    enable_events=True,
                ),
            ],
            [tabgroup_modeller_setting],
        ]

        tab_modeller = sg.Tab("モデリング", layout_modeller)  # , size=(500,300))

        layout_train_by_alldata = [
            [
                sg.Text("ハイパーパラメータの採用手法", size=(15, 1)),
                sg.Radio(
                    "train-val-test",
                    default=False,
                    key="optimize_method__holdout-validation-optimize",
                    group_id="optimize_method",
                    enable_events=True,
                ),
                sg.Radio(
                    "train(cross-optimize)-test",
                    default=False,
                    key="optimize_method__holdout-cross-optimize",
                    group_id="optimize_method",
                    enable_events=True,
                ),
                sg.Radio(
                    "cross-optimize",
                    default=True,
                    key="optimize_method__cross-optimize",
                    group_id="optimize_method",
                    enable_events=True,
                ),
            ],
            [
                sg.Text("モデル作成", size=(15, 1)),
                sg.Checkbox("全データで訓練", default=True, key="ml_train_by_alldata"),
            ],
        ]

        tab_train_by_alldata = sg.Tab("全データ学習", layout_train_by_alldata)

        layout_explainer = [
            [
                sg.Text("SHAP", size=(15, 1)),
                sg.Checkbox("SHAP", default=True, key="shap"),
                sg.Checkbox("SHAP-KernelExplainer", default=True, key="shap_kernel"),
                sg.Checkbox("SHAP-DeepExplainer", default=False, key="shap_deep"),
            ],
            [
                sg.Text("PDP", size=(15, 1)),
                sg.Checkbox("1D", default=True, key="1D_PDP"),
                sg.Checkbox("2D", default=False, key="2D_PDP"),
            ],
            [
                sg.Text("LiNGAM", size=(15, 1)),
                sg.Checkbox("LiNGAM", default=False, key="LiNGAM"),
                sg.Checkbox("VARLiNGAM", default=False, key="VARLiNGAM"),
            ],
        ]

        tab_explainer = sg.Tab("可視化", layout_explainer)
        tabgroup_explainer = sg.TabGroup([[tab_explainer]])

        n_cpu = psutil.cpu_count(logical=False)
        layout_n_jobs = [
            [
                sg.Text("n_jobs", size=(10, 1)),
                sg.Radio(
                    "Auto",
                    default=True,
                    key="n_jobs__auto",
                    group_id="n_jobs",
                    enable_events=True,
                ),
                sg.Radio(
                    "Max",
                    default=False,
                    key="n_jobs__max",
                    group_id="n_jobs",
                    enable_events=True,
                ),
                sg.Radio(
                    "Middle",
                    default=False,
                    key="n_jobs__middle",
                    group_id="n_jobs",
                    enable_events=True,
                ),
                sg.Radio(
                    "Single",
                    default=False,
                    key="n_jobs__single",
                    group_id="n_jobs",
                    enable_events=True,
                ),
            ],
            [
                sg.Text("fit", size=(20, 1)),
                sg.InputText(f"-1", size=(3, 1), key="n_jobs__model"),
            ],
            [
                sg.Text("cross_val_predict", size=(20, 1)),
                sg.InputText(f"-1", size=(3, 1), key="n_jobs__cross_val_predict"),
            ],
            [
                sg.Text("SearchCV", size=(20, 1)),
                sg.InputText(f"1", size=(3, 1), key="n_jobs__SearchCV"),
            ],
            [
                sg.Text("SHAP", size=(20, 1)),
                sg.InputText(f"1", size=(3, 1), key="n_jobs__SHAP"),
            ],
        ]

        tab_n_jobs = sg.Tab("n_jobs", layout_n_jobs)

        layout_delayed_start = [
            [
                sg.Text("CPU使用率(%)", size=(20, 1)),
                sg.InputText("80", size=(3, 1), key="delayed_start__cpu_percent"),
            ],
            [
                sg.Text("遅延時間(s)", size=(20, 1)),
                sg.InputText("", size=(3, 1), key="delayed_start__time"),
            ],
        ]
        tab_delayed_start = sg.Tab("遅延実行", layout_delayed_start)

        tabgroup_setting = sg.TabGroup([[tab_n_jobs,
                                        tab_delayed_start]])
        layout_starter = [[sg.Submit(button_text="start")]]
        tabgroup_training = sg.TabGroup(
            [
                [
                    tab_preprocessing,
                    tab_selector,
                    tab_evaluater,
                    tab_modeller,
                    tab_train_by_alldata,
                    # tab_clustering
                ]
            ]
        )

        layout_load_trained_model = [
            [
                sg.Radio(
                    "学習済みモデル読込",
                    default=False,
                    key="load_trained_model",
                    group_id="train",
                    enable_events=True,
                ),
                sg.FolderBrowse("フォルダ選択", key="trained_model_path"),
                sg.InputText("trainedmodel", size=(20, 1)),
            ]
        ]

        tab_load_trained_model = sg.Tab(
            "学習済みモデル読込", layout_load_trained_model
        )

        tabgroup_predict = sg.TabGroup([[tab_load_trained_model]])

        tabgroup_train_predict = sg.TabGroup(
            [
                [
                    sg.Tab("学習", [[tabgroup_training]]),
                    sg.Tab("予測", [[tabgroup_predict]]),
                    sg.Tab("可視化", [[tabgroup_explainer]]),
                    sg.Tab("設定", [[tabgroup_setting]]),
                ]
            ]
        )

        tab_simple = sg.Tab("Simple", [])
        tab_advance = sg.Tab("Advanced", [[tabgroup_train_predict]])
        tabgroup_simple_advance = sg.TabGroup([[tab_simple, tab_advance]])

        layout = [
            [frame_reader],
            layout_theme,
            layout_supervised,
            layout_clustering,
            layout_training_predict,
            layout_starter,
            [tabgroup_simple_advance],
        ]

        return layout

    # GUI - モデル選択
    def select_model(models_list) -> list:
        # pprint.pprint(models_list)
        """pysimplegui to select models

        Parameters
        ---------
        models_list : list
            estimators list
            e.g. [LinearRegression(),  RandomForestRegressor()]


        CheckBox_dict :dict
                {"Linear":
                    [
                        sg.CheckBox("LinearRegression",
                        key="models__LinearRegression",
                        default=True,
                        size=(15,1),
                        sg.CheckBox("Lasso",
                        key="models__Lasso",
                        default=True,
                        size=(15,1),
                    ],
                "Tree":
                    [
                        sg.CheckBox("DecisionTreeRegressor",
                        key="models__DecisionTreeRegressor",
                        default=True,
                        size=(15,1),
                        sg.CheckBox("RandomForestRegressor",
                        key="models__RandomForestRegressor",
                        default=True,
                        size=(15,1),
                    ],
                        }
        layout_Linear : list[list[sg.class]]
            e.g.[[sg.Button("LinearRegression", key="models__LinearRegression")]]
        layout_tree : list(list(sg.class))
            e.g.[[sg.Button("RandomForest", key="models__RandomForest")]]
        layout_others : list(list(sg.class))
            e.g.[[sg.Button("RuleFit", key="models__RuleFit")]]

        layout : list[list[sg.class]]
                [[sg.Frame("Linear", layout_linear),
                  sg.Frame("Tree", layout_tree),
                  sg.Frame("Other", layout_other),
                    ]]
        """

        CheckBox_dict = {}
        model_name_type_dict = {}
        model_type_name_dict = {}

        for model in models_list:
            model_type = getattr(model, "type", "others")
            model_name = get_estimator(model, remove_multioutput=False)[1]
            model_name_type_dict[model_name] = model_type
            model_type_name_dict.setdefault(model_type, []).append(model_name)

            default = getattr(model, "default", False)

            ckb = [
                sg.Checkbox(
                    model_name,
                    key=f"models__{model_name}",
                    default=default,
                    size=(15, 1),
                )
            ]
            CheckBox_dict.setdefault(model_type, []).append(ckb)

        layout = []
        for k, v in CheckBox_dict.items():
            each_sg = sg.Frame(k, v)
            each_sg.VerticalAlignment = "top"
            layout.append(each_sg)
        layout = [layout]

        select_parts = []
        for k, v in CheckBox_dict.items():
            sgb = sg.Button(f"{k}", size=(20, 1), key=f"select_{k}")
            sgb.VerticalAlignment = "top"
            select_parts.append(sgb)
        select_parts = [select_parts]

        cancel_parts = []
        for k, v in CheckBox_dict.items():
            sgb = sg.Button(f"cancel", size=(20, 1), key=f"cancel_{k}")
            sgb.VerticalAlignment = "top"
            cancel_parts.append(sgb)
        cancel_parts = [cancel_parts]
        model_types = CheckBox_dict.keys()
        select_keys = [f"select_{i}" for i in model_types]
        cancel_keys = [f"cancel_{i}" for i in model_types]

        select_all = [
            sg.Button("全選択", size=(10, 1), key="select_all"),
            sg.Button("全解除", size=(10, 1), key="cancel_all"),
        ]

        recommend_model_names = []
        for model_type_, model_names in model_type_name_dict.items():
            if model_type_ in ["Linear", "Kernel", "Tree"]:
                recommend_model_names.extend(model_names)

            for each_model_name in model_names:
                if each_model_name in ["RuleFit"]:
                    recommend_model_names.append(each_model_name)

        recommend_model_names = [
            i
            for i in recommend_model_names
            if i not in ["BartRegressor", "GBartRegressor"]
        ]
        recommend_button = [sg.Button("推奨モデル", size=(10, 1), key="recommend")]
        button = [sg.Button("決定", size=(10, 1), key="select")]
        layout_model_select = [
            layout,
            select_parts,
            cancel_parts,
            select_all,
            recommend_button,
            button,
        ]
        window_model_select = sg.Window("Select the model", layout_model_select)

        while True:
            event_model_select, values_model_select = window_model_select.read()
            if event_model_select == "select_all":
                [
                    window_model_select[k].Update(True)
                    for k, v in values_model_select.items()
                    if ("models__" in str(k))
                ]

            if event_model_select == "cancel_all":
                [
                    window_model_select[k].Update(False)
                    for k, v in values_model_select.items()
                    if ("models__" in str(k))
                ]

            if event_model_select == "recommend":
                for k, v in values_model_select.items():
                    if "models__" in str(k):
                        if k.replace("models__", "") in recommend_model_names:
                            window_model_select[k].Update(True)
                        else:
                            window_model_select[k].Update(False)

            if event_model_select in select_keys:
                model_type = event_model_select.replace("select_", "")
                [
                    window_model_select[k].Update(True)
                    for k, v in values_model_select.items()
                    if (
                        ("models__" in str(k))
                        and (
                            model_name_type_dict.get(k.replace("models__", ""), "")
                            == model_type
                        )
                    )
                ]

            if event_model_select in cancel_keys:
                model_type = event_model_select.replace("cancel_", "")
                [
                    window_model_select[k].Update(False)
                    for k, v in values_model_select.items()
                    if (
                        ("models__" in str(k))
                        and (
                            model_name_type_dict.get(k.replace("models__", ""), "")
                            == model_type
                        )
                    )
                ]

            # 選択ウインドウが閉じたら（None）もしくは 決定ボタンを押したら（select）
            if (event_model_select is None) or (event_model_select == "select"):
                print("model selected")
                # [print(model.model_name) for model in models_list]
                # GUI上で選択されたモデル"名"のリスト
                if values_model_select is not None:
                    selected_models_list = [
                        k
                        for k, v in values_model_select.items()
                        if ("models__" in str(k)) and ((v == True))
                    ]
                else:
                    selected_models_list = []
                # 選択されたモデルのdefaultをTrueに変更
                [
                    setattr(model, "default", True)
                    if f"models__{model.model_name}" in selected_models_list
                    else setattr(model, "default", False)
                    for model in models_list
                ]
                break

        window_model_select.close()

        return models_list

    main_layout = create_main_layout()
    window = sg.Window("Machine-Learning", main_layout, location=(100, 100))

    # GUI部分
    # initilize
    standard_supervised_estimators = None

    while True:
        event, values = window.read()

        if event is None:
            print("exit")
            break

        theme_name = None

        if event == "csv_path_":
            csv_path = Path(values["csv_path_"])
            parent_folder_path = csv_path.parent.parent
            parent_folder_name = parent_folder_path.name
            window["theme_name"].Update(parent_folder_name)

        theme_name = values["theme_name"].replace("\n", "")

        if event == "group_num":
            if values["group_num"].isdigit() == True:
                if int(values["group_num"]) >= 1:
                    window["optimize_cv_splitter__GroupKFold"].Update(visible=True)
                    window["optimize_cv_splitter_args__GroupKFold"].Update(visible=True)

                    window["optimize_cv_splitter__LeaveOneGroupOut"].Update(
                        visible=True
                    )
                    window["optimize_cv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=True
                    )

                    window["evaluate_cv_splitter__GroupKFold"].Update(visible=True)
                    window["evaluate_cv_splitter_args__GroupKFold"].Update(visible=True)

                    window["evaluate_cv_splitter__LeaveOneGroupOut"].Update(
                        visible=True
                    )
                    window["evaluate_cv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=True
                    )

                    window["evaluate_dcv_splitter__GroupKFold"].Update(visible=True)
                    window["evaluate_dcv_splitter_args__GroupKFold"].Update(
                        visible=True
                    )

                    window["evaluate_dcv_splitter__LeaveOneGroupOut"].Update(
                        visible=True
                    )
                    window["evaluate_dcv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=True
                    )

                elif int(values["group_num"]) == 0:
                    window["optimize_cv_splitter__KFold_shuffle"].Update(True)
                    window["evaluate_cv_splitter__KFold_shuffle"].Update(True)
                    window["evaluate_dcv_splitter__KFold_shuffle"].Update(True)

                    window["evaluate_cv_splitter__GroupKFold"].Update(False)
                    window["evaluate_cv_splitter__LeaveOneGroupOut"].Update(False)
                    window["evaluate_dcv_splitter__GroupKFold"].Update(False)
                    window["evaluate_dcv_splitter__LeaveOneGroupOut"].Update(False)
                    window["evaluate_cv_splitter__GroupKFold"].Update(False)
                    window["evaluate_cv_splitter__LeaveOneGroupOut"].Update(False)

                    window["optimize_cv_splitter__GroupKFold"].Update(visible=False)
                    window["optimize_cv_splitter__LeaveOneGroupOut"].Update(
                        visible=False
                    )
                    window["evaluate_cv_splitter__GroupKFold"].Update(visible=False)
                    window["evaluate_cv_splitter__LeaveOneGroupOut"].Update(
                        visible=False
                    )
                    window["evaluate_dcv_splitter__GroupKFold"].Update(visible=False)
                    window["evaluate_dcv_splitter__LeaveOneGroupOut"].Update(
                        visible=False
                    )

                    window["optimize_cv_splitter_args__GroupKFold"].Update(
                        visible=False
                    )
                    window["optimize_cv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=False
                    )
                    window["evaluate_cv_splitter_args__GroupKFold"].Update(
                        visible=False
                    )
                    window["evaluate_cv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=False
                    )
                    window["evaluate_dcv_splitter_args__GroupKFold"].Update(
                        visible=False
                    )
                    window["evaluate_dcv_splitter_args__LeaveOneGroupOut"].Update(
                        visible=False
                    )

        if event == "input_num":
            if (values["input_num"] != "") and (values["input_num"].isdigit() == True):
                if int(values["input_num"]) == 0:
                    print("inputには1以上の値を入力してください")

        if event == "output_num":
            if values["output_num"].isdigit() == True:
                if int(values["output_num"]) >= 2:
                    window["MultiOutputRegressor"].Update(True)

        # 列数が入力された場合、更新する
        if event in ["timeseries_num",
                    "info_num",
                    "group_num",
                    "evaluate_sample_weight_num",
                    "training_sample_weight_num",
                    "input_num",
                    "output_num"]:
            try:
                timeseries_num = int(values["timeseries_num"])
                info_num = int(values["info_num"])
                group_num = int(values["group_num"])
                evaluate_sample_weight_num = int(values["evaluate_sample_weight_num"])
                training_sample_weight_num = int(values["training_sample_weight_num"])
                input_num = int(values["input_num"])
                output_num = int(values["output_num"])

                timeseries_col = timeseries_num
                info_col = timeseries_num + info_num
                group_col = timeseries_num + info_num + group_num
                evaluate_sample_weight_col = (
                    timeseries_num + info_num + group_num
                    + evaluate_sample_weight_num
                )
                training_sample_weight_col = (
                    timeseries_num + info_num + group_num
                    + evaluate_sample_weight_num
                    + training_sample_weight_num
                )
                input_col = (
                    timeseries_num + info_num + group_num
                    + evaluate_sample_weight_num
                    + training_sample_weight_num
                    + input_num
                )
                output_col = (
                    timeseries_num + info_num + group_num
                    + evaluate_sample_weight_num
                    + training_sample_weight_num
                    + input_num
                    + output_num
                )

                window["timeseries_column_list"].Update(f"0:{timeseries_col}")
                window["info_column_list"].Update(f"{timeseries_col}:{info_col}")
                window["group_column_list"].Update(f"{info_col}:{group_col}")
                window["evaluate_sample_weight_column_list"].Update(f"{group_col}:{evaluate_sample_weight_col}")
                window["training_sample_weight_column_list"].Update(f"{evaluate_sample_weight_col}:{training_sample_weight_col}")
                window["input_column_list"].Update(f"{training_sample_weight_col}:{input_col}")
                window["output_column_list"].Update(f"{input_col}:{output_col}")

            except:
                pass


        if event == "training":
            window["load_trained_model"].Update(False)

        if event == "trained":
            window["load_trained_model"].Update(True)

        if event == "supervised":
            if values["supervised"] == True:
                window["sv_type__regression"].Update(True)
                window["optimize_regression_scorer"].Update(visible=True)
                window["optimize_classification_scorer"].Update(visible=False)
                window["y_scaler__StandardScaler()"].Update(True)

            if values["supervised"] == False:
                window["sv_type__regression"].Update(False)
                window["sv_type__classification"].Update(False)
                window["optimize_regression_scorer"].Update(visible=False)
                window["optimize_classification_scorer"].Update(visible=False)

        if event == "sv_type__regression":
            window["supervised"].Update(True)
            window["optimize_regression_scorer"].Update(visible=True)
            window["optimize_classification_scorer"].Update(visible=False)
            window["y_scaler__StandardScaler"].Update(True)

        if event == "sv_type__classification":
            window["supervised"].Update(True)
            window["optimize_regression_scorer"].Update(visible=False)
            window["optimize_classification_scorer"].Update(visible=True)
            window["y_scaler__FunctionTransformer"].Update(True)

        if values["ml_cross_optimize"] == True:
            window["op_tab__crossvalidation"].Update(visible=True)
            window["op_tab__holdout"].Update(visible=False)
            window["op_tab__crossvalidation"].select()

        elif values["ml_holdout_optimize"] == True:
            window["op_tab__crossvalidation"].Update(visible=False)
            window["op_tab__holdout"].Update(visible=True)
            window["op_tab__holdout"].select()

        if event == "select_standard_models":
            if standard_supervised_estimators is None:
                standard_supervised_estimators = []
                sv_types = [
                    k.replace("sv_type__", "")
                    for k, v in values.items()
                    if ("sv_type__" in str(k)) and (v == True)
                ]
                for sv_type in sv_types:
                    if values["load_model__load_all_models"] == True:
                        standard_supervised_estimators.extend(
                            load_all_models(ml_type=sv_type)
                        )

            standard_supervised_estimators = select_model(
                standard_supervised_estimators
            )

            print("standard_supervised_estimators")
            print(standard_supervised_estimators)
            """
            for supervised_estimator in standard_supervised_estimators:
                pass
                #print(getattr(supervised_estimator, "default", None))
            """

        if event == "n_jobs__auto":
            window["n_jobs__model"].Update(-1)
            window["n_jobs__cross_val_predict"].Update(-1)
            window["n_jobs__SearchCV"].Update(1)
            window["n_jobs__SHAP"].Update(-1)

        if event == "n_jobs__max":
            window["n_jobs__model"].Update(-1)
            window["n_jobs__cross_val_predict"].Update(-1)
            window["n_jobs__SearchCV"].Update(-1)
            window["n_jobs__SHAP"].Update(-1)

        if event == "n_jobs__middle":
            window["n_jobs__model"].Update(-1)
            window["n_jobs__cross_val_predict"].Update(1)
            window["n_jobs__SearchCV"].Update(1)
            window["n_jobs__SHAP"].Update(1)

        if event == "n_jobs__single":
            window["n_jobs__model"].Update(1)
            window["n_jobs__cross_val_predict"].Update(1)
            window["n_jobs__SearchCV"].Update(1)
            window["n_jobs__SHAP"].Update(1)

        if event == "start":
            csv_path = Path(values["csv_path_"])
            if values["timeseries_num"] != "" and (values["timeseries_num"].isdigit() == True):
                time_column_num = int(values["timeseries_num"])
            else:
                time_column_num = 0

            if (values["info_num"] != "") and (values["info_num"].isdigit() == True):
                info_column_num = int(values["info_num"])
            else:
                info_column_num = 0

            if (values["group_num"] != "") and (values["group_num"].isdigit() == True):
                group_column_num = int(values["group_num"])
            else:
                group_column_num = 0

            if (values["evaluate_sample_weight_num"] != "") and (
                values["evaluate_sample_weight_num"].isdigit() == True
            ):
                evaluate_sample_weight_column_num = int(
                    values["evaluate_sample_weight_num"]
                )
            else:
                evaluate_sample_weight_column_num = 0

            if (values["training_sample_weight_num"] != "") and (
                values["training_sample_weight_num"].isdigit() == True
            ):
                training_sample_weight_column_num = int(
                    values["training_sample_weight_num"]
                )
            else:
                training_sample_weight_column_num = 0

            if (values["input_num"] != "") and (values["input_num"].isdigit() == True):
                X_column_num = int(values["input_num"])
            else:
                print("inputには値を入力してください")
                continue

            if (values["output_num"] != "") and (values["output_num"].isdigit() == True):
                y_column_num = int(values["output_num"])
            else:
                print("output : 0 には現時点で未対応です")
                continue

            (
                X_df,
                y_df,
                datetime_df,
                group_df,
                evaluate_sample_weight_df,
                training_sample_weight_df,
                info_df,
                raw_df,
                encoding,
            ) = reader(
                csv_path,
                time_column_num,
                info_column_num,
                group_column_num,
                evaluate_sample_weight_column_num,
                training_sample_weight_column_num,
                X_column_num,
                y_column_num,
            )

            X_scaler = [
                k for k, v in values.items() if ("X_scaler__" in str(k)) and (v == True)
            ][0]
            X_scaler_string = X_scaler.replace("X_scaler__", "")
            # e.g. X_scaler:StandardScaler()
            X_scaler = X_scaler_dict[X_scaler_string]["instance"]
            X_scaler.step_name = "X_scaler"
            X_scalers = [X_scaler]

            if values["supervised"] == True:
                supervised = True
            else:
                supervised = False

            y_scaler = [
                k for k, v in values.items() if ("y_scaler__" in str(k)) and (v == True)
            ][0]
            y_scaler_string = y_scaler.replace("y_scaler__", "")
            # e.g. y_scaler:StandardScaler()
            y_scaler = y_scaler_dict[y_scaler_string]["instance"]
            y_scaler.step_name = "y_scaler"
            y_scalers = [y_scaler]

            missing_data = [
                k
                for k, v in values.items()
                if ("missing_data__" in str(k)) and (v == True)
            ][0]
            missing_data = missing_data.replace("missing_data__", "")
            missing_data = eval(missing_data)()
            missing_data.step_name = "missing"
            imputers = [missing_data]

            feature_engineering_dict = {
                k: v
                for k, v in values.items()
                if ("feature_engineering_order__" in str(k))
                and (str.isdigit(v) == True)
            }
            feature_engineering_list = sorted(
                feature_engineering_dict.items(), key=lambda x: x[1]
            )
            print("feature_engineering_list", feature_engineering_list)
            engineers = []
            for fe_index, each_fe in enumerate(feature_engineering_list):
                each_fe = each_fe.replace("feature_engineering_order__", "")
                each_fe = eval(each_fe)()
                each_fe.step_name = f"feature_engineering{fe_index}"
                engineers.append(each_fe)

            if len(engineers) == 0:
                fe = FunctionTransformer()
                fe.step_name = "feature_engineering0"
                engineers.append(fe)

            engineers = [engineers[0]]

            choosed_selector_dict = {
                k.replace("selector__", ""): v
                for k, v in values.items()
                if ("selector__" in str(k)) and ((v == True))
            }

            print("choosed_selector_dict")
            print(choosed_selector_dict)

            if len(choosed_selector_dict) == 0:
                choosed_selector_dict = {"None": True}

                print("No selector is choosed. Do not select variables")
                print(choosed_selector_dict)
                time.sleep(0.5)

            ml_types = [
                k for k, v in values.items() if ("sv_type__" in str(k)) and (v == True)
            ]

            print("ml_types")
            print(ml_types)

            if len(ml_types) != 0:
                ml_type = ml_types[0].replace("sv_type__", "")
                print("ml_type")
                print(ml_type)
                # _selectors.pyから記述子選択手法を読み込む
                all_selectors = load_selectors(ml_type=ml_type)
                print("all_selectors")
                print(all_selectors)
                # GUIで選択した記述子選択手法に絞る
                selectors = [
                    selector
                    for selector in all_selectors
                    if selector.model_name in choosed_selector_dict.keys()
                ]
                print("selectors")
                print(selectors)
            else:
                selectors = []

            if values["training"] == True:
                supervised_estimators = []
                sv_types = [
                    k.replace("sv_type__", "")
                    for k, v in values.items()
                    if ("sv_type__" in str(k)) and (v == True)
                ]

                for sv_type in sv_types:
                    if values["load_model__load_all_models"] == True:
                        if standard_supervised_estimators is None:
                            standard_supervised_estimators = load_all_models(
                                ml_type=sv_type
                            )
                        else:
                            pass

                        selected_supervised_estimators = [
                            estimator
                            for estimator in standard_supervised_estimators
                            if getattr(estimator, "default", True) == True
                        ]
                        supervised_estimators.extend(selected_supervised_estimators)

                    if values["load_model__load_tabnet"] == True:
                        supervised_estimators.extend(load_tabnet(ml_type=sv_type))

                    if values["load_model__load_automl"] == True:
                        supervised_estimators.extend(load_automl(ml_type=sv_type))

                print("supervised_estimators")
                print(supervised_estimators)
                # time.sleep(15)

                unsupervised_estimators = []
                us_types = [
                    k.replace("us_type__", "")
                    for k, v in values.items()
                    if ("us_type__" in str(k)) and (v == True)
                ]

                for us_type in us_types:
                    unsupervised_estimators.extend(load_all_models(ml_type=us_type))

            elif values["load_trained_model"] == True:
                # TODO 学習済みのモデルを読み込む
                # supervised_estimators = load_model_pickel(values["trained_model_path"])
                raise ValueError("学習済みモデルのロードは未実装です")
                supervised_estimators = None
            else:
                supervised_estimators = None

            n_jobs__model = int(values["n_jobs__model"])

            supervised_estimators = [
                supervised_estimator for supervised_estimator in supervised_estimators
            ]
            unsupervised_estimators = [
                unsupervised_estimator
                for unsupervised_estimator in unsupervised_estimators
            ]

            evaluate_holdout_setting = {}
            evaluate_holdout_setting["do"] = values["ml_holdout_evaluate"]

            if values["ml_holdout_evaluate"] == True:
                evaluate_holdout_setting["test_ratio"] = float(values["test_ratio"])

            evaluate_dcv_setting = {}
            evaluate_dcv_setting["do"] = values["ml_double_cross_validate"]

            CV_dict = load_splliter_dict()

            if values["ml_double_cross_validate"] == True:
                evaluate_dcv_splitter = [
                    k
                    for k, v in values.items()
                    if ("evaluate_dcv_splitter__" in str(k)) and (v == True)
                ][0]
                evaluate_dcv_splitter = evaluate_dcv_splitter.replace(
                    "evaluate_dcv_splitter__", ""
                )
                evaluate_dcv_splitter = CV_dict[evaluate_dcv_splitter]["instance"]
                print(evaluate_dcv_splitter)
                evaluate_dcv_splitter_args = [
                    v
                    for k, v in values.items()
                    if ("evaluate_dcv_splitter_args__" in str(k))
                    and (
                        values[
                            "evaluate_dcv_splitter__{}".format(
                                k.replace("evaluate_dcv_splitter_args__", "")
                            )
                        ]
                        == True
                    )
                ][0]

                evaluate_dcv_splitter_dict = make_args(evaluate_dcv_splitter_args)

                for k, v in evaluate_dcv_splitter_dict.items():
                    if hasattr(evaluate_dcv_splitter, k):
                        setattr(evaluate_dcv_splitter, k, v)


                evaluate_dcv_setting["splitter"] = evaluate_dcv_splitter
                evaluate_dcv_setting["splitter_args"] = evaluate_dcv_splitter_args

                if values["sv_type__regression"] == True:
                    evaluate_dcv_setting["scorer"] = ["r2_score", "mean_absolute_error"]
                elif values["sv_type__classification"] == True:
                    evaluate_dcv_setting["scorer"] = ["accuracy_score", "auc"]
                else:
                    evaluate_dcv_setting["scorer"] = ["r2_score", "mean_absolute_error"]

            evaluate_cv_setting = {}
            evaluate_cv_setting["do"] = values["ml_cross_opt_evaluate"]

            if values["ml_cross_opt_evaluate"] == True:

                evaluate_cv_splitter = [
                    k
                    for k, v in values.items()
                    if ("evaluate_cv_splitter__" in str(k)) and (v == True)
                ][0]
                evaluate_cv_splitter = evaluate_cv_splitter.replace(
                    "evaluate_cv_splitter__", ""
                )
                evaluate_cv_splitter = CV_dict[evaluate_cv_splitter]["instance"]

                evaluate_cv_setting["splitter"] = evaluate_cv_splitter

                evaluate_cv_splitter_args = [
                    v
                    for k, v in values.items()
                    if ("evaluate_cv_splitter_args__" in str(k))
                    and (
                        values[
                            "evaluate_cv_splitter__{}".format(
                                k.replace("evaluate_cv_splitter_args__", "")
                            )
                        ]
                        == True
                    )
                ][0]
                evaluate_cv_splitter_args = make_args(evaluate_cv_splitter_args)
                evaluate_cv_setting["splitter_args"] = evaluate_cv_splitter_args

                if values["sv_type__regression"] == True:
                    evaluate_cv_setting["scorer"] = ["r2_score", "mean_absolute_error"]
                elif values["sv_type__classification"] == True:
                    evaluate_cv_setting["scorer"] = ["accuracy_score", "auc"]
                else:
                    evaluate_cv_setting["scorer"] = ["r2_score", "mean_absolute_error"]

            evaluate_all_setting = {}
            evaluate_all_setting["do"] = True
            evaluate_all_setting["splitter"] = None
            evaluate_all_setting["splitter_args"] = None

            if values["sv_type__regression"] == True:
                evaluate_all_setting["scorer"] = ["r2_score", "mean_absolute_error"]
            elif values["sv_type__classification"] == True:
                evaluate_all_setting["scorer"] = ["accuracy_score", "auc"]
            else:
                evaluate_all_setting["scorer"] = ["r2_score", "mean_absolute_error"]

            evaluate_setting = {
                "evaluate_cv_setting": evaluate_cv_setting,
                "evaluate_dcv_setting": evaluate_dcv_setting,
                "evaluate_holdout_setting": evaluate_holdout_setting,
                "evaluate_all_setting": evaluate_all_setting,
            }

            optimize_splitter = [
                k
                for k, v in values.items()
                if ("optimize_cv_splitter__" in str(k)) and (v == True)
            ][0].replace("optimize_cv_splitter__", "")
            optimize_splitter = CV_dict[optimize_splitter]["instance"]

            optimize_cv_splitter_args = [
                v
                for k, v in values.items()
                if ("optimize_cv_splitter_args__" in str(k))
                and (
                    values[
                        "optimize_cv_splitter__{}".format(
                            k.replace("optimize_cv_splitter_args__", "")
                        )
                    ]
                    == True
                )
            ][0]
            optimize_cv_splitter_args = optimize_cv_splitter_args.replace(
                "optimize_cv_splitter_args__", ""
            )
            optimize_cv_splitter_dict = make_args(optimize_cv_splitter_args)

            for k, v in optimize_cv_splitter_dict.items():
                if hasattr(optimize_splitter, k):
                    setattr(optimize_splitter, k, v)


            optimize_searcher_key = [
                k
                for k, v in values.items()
                if ("optimize_searcher__" in str(k)) and (v == True)
            ][0].replace("optimize_searcher__", "")

            optimize_searcher_class = searcher_dict[optimize_searcher_key]["class"]
            print("optimize_searcher_class")
            print(optimize_searcher_class)

            optimize_searcher_args = [
                v
                for k, v in values.items()
                if ("optimize_searcher_args__" in str(k))
                and (
                    values[
                        "optimize_searcher__{}".format(
                            k.replace("optimize_searcher_args__", "")
                        )
                    ]
                    == True
                )
            ][0]
            optimize_searcher_args = optimize_searcher_args.replace(
                "optimize_searcher_args__", ""
            )
            optimize_searcher_args = make_args(optimize_searcher_args)
            sig_optimizer = signature(optimize_searcher_class)
            n_jobs__SearchCV = int(values["n_jobs__SearchCV"])
            if "n_jobs" in sig_optimizer.parameters:
                # n_jobsが指定されていない場合はn_jobs__SearchCVを代入する
                optimize_searcher_args.setdefault("n_jobs", n_jobs__SearchCV)

            optimize__scorer = values["optimize_regression_scorer"]

            if values["ml_cross_optimize"] == True:
                optimize_setting = {}
                optimize_setting["splitter"] = optimize_splitter
                optimize_setting["splitter_args"] = optimize_cv_splitter_args

            elif values["ml_holdout_optimize"] == True:
                print("not supported yet..")
                time.sleep(10)
                optimizer = None

            else:
                optimizer = None

            SearchCV = partial(optimize_searcher_class, **optimize_searcher_args)

            optimize_setting["searcher"] = optimize_searcher_class
            optimize_setting["scorer"] = optimize__scorer
            optimize_setting["searcher_args"] = optimize_searcher_args

            shap_except_explainers = []
            if values["shap"] == False:
                shap_except_explainers.extend(["Tree", "Linear"])

            if values["shap_kernel"] == False:
                shap_except_explainers.append("Kernel")

            if values["shap_deep"] == False:
                shap_except_explainers.append("Deep")

            PDPs = []
            if values["1D_PDP"] == True:
                PDPs.append("1D_PDP")

            print("shap_except_explainers")
            print(shap_except_explainers)
            time.sleep(0.5)
            explainer_setting = {"shap": shap_except_explainers, "pdp": PDPs}

            n_jobs = values["n_jobs__cross_val_predict"]
            n_cpu = psutil.cpu_count(logical=False)
            if n_jobs.isdigit() == True:
                n_jobs = int(n_jobs)

                if n_jobs >= n_cpu:
                    n_jobs = n_cpu

            else:
                if "max" in n_jobs:
                    n_jobs = -1
                else:
                    n_jobs = -2

            print("show the setting...")

            delayed_start_cpu_percent = values["delayed_start__cpu_percent"]
            delayed_start_time = values["delayed_start__time"]

            if delayed_start_cpu_percent.isdigit() == True:
                threshold_cpu = float(delayed_start_cpu_percent)
                print(f"cpu使用率が{threshold_cpu}%以下でプログラム開始します")
                delayed_start(threshold_cpu)
            elif delayed_start_time.isdigit() == True:
                delayed_time = float(delayed_start_time)
                print(f"{delayed_time}秒間sleepします")
                time.sleep(delayed_time)

            cache = memory.Memory(location=tempfile.TemporaryDirectory().name, verbose=0)

            pipelines = make_pipelines(
                                    imputers=imputers,
                                    scalers=X_scalers,
                                    engineers=engineers,
                                    selectors=selectors,
                                    estimators=supervised_estimators,
                                    memory=cache
                                )

            SearchCV = partial(optimize_searcher_class, cv=optimize_splitter, **optimize_searcher_args)
            wrapped_models = [wrap_searchcv(i, SearchCV) for i in pipelines]

            # 尚、計算実行時にプロセス優先度を通常に戻す
            print("プロセス優先度を下げます")
            proc = psutil.Process(os.getpid())
            if platform.system() == 'Windows':
                proc.nice( psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                proc.nice(5)

            dt = datetime.datetime.now()
            dt_str = dt.strftime("%Y%m%d")

            save_path = None
            if save_path is None:
                dt = datetime.datetime.now()
                dt_name = dt.strftime("%Y%m%d_%H%M%S")

                user_path = os.path.expanduser("~")
                user_path = Path(user_path)

                if (theme_name is None) or str(theme_name) == "":
                    save_path = user_path / "analyze" / dt_str / dt_name
                else:
                    save_path = user_path / "analyze" / theme_name / dt_str / dt_name

            supervised_learning(
                wrapped_models,
                X_df,
                y_df,
                info_df=info_df,
                datetime_df=datetime_df,
                group_df=group_df,
                evaluate_sample_weight_df=evaluate_sample_weight_df,
                training_sample_weight_df=training_sample_weight_df,
                theme_name=theme_name,
                evaluate_splitter=evaluate_dcv_splitter,
                save_path=save_path)

            clustering(
                unsupervised_estimators,
                X_df,
                y_df,
                X_names=X_df.columns,
                y_names=y_df.columns,
                info=info_df,
                info_names=info_df.columns,
                group=group_df,
                save_path=save_path)
            print("finish clustering")

    # セクション 4 - ウィンドウの破棄と終了
    window.close()

def main():

    # 実行開始時にプロセス優先度をあげる
    # 尚、計算実行時には通常に戻す
    print("プロセス優先度を一時的にあげます")

    proc = psutil.Process(os.getpid())
    if platform.system() == 'Windows':
        proc.nice( psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    else:
        proc.nice(-15)

    parser = argparse.ArgumentParser()

    # 引数には大別して5つ, ＋可視化に関する設定、計算条件設定
    # データセット（X,y,groups, sample_weight, timeseries）
    # 回帰 もしくは 分類
    # 教師なし学習
    # モデリングに関する設定(モデル自身の情報)
    # ハイパーパラメータチューニングに関する情報)
    # 評価に関する設定(CV)
    # 可視化に関する設定
    # 計算条件設定
    parser.add_argument("--ui", default="GUI")
    parser.add_argument("--gui", action='store_true')
    parser.add_argument("--cui", action='store_true')
    parser.add_argument("--load_dataset", default=None, type=Path)
    parser.add_argument("--load_dataset_config", default=None, type=Path)
    parser.add_argument("--supervised", default="regression")
    parser.add_argument("--unsupervised", default=False)
    parser.add_argument("--load_model", default=None, type=Path)
    parser.add_argument("--load_model_config", default="", type=Path)
    parser.add_argument("--load_evaluate", default=None, type=Path)
    parser.add_argument("--load_explainer", default=None, type=Path)
    parser.add_argument("--load_setting", default=None, type=Path)
    parser.add_argument("--delayed_start", default=None, type=str)
    args = parser.parse_args()
    print("Print args")
    print(args)
    print(args.ui)

    if args.gui == True:
        ui = "gui"
    elif args.ui.lower() == "gui":
        ui = "gui"
    else:
        ui = "cui"

    if ui == "gui":
        print("start gui")
        main_gui()

    else:
        if args.delayed_start is not None:
            if "%" in args.delayed_start:
                threshold_cpu = float(args.delayed_start.split("%")[0])
                print(f"cpu使用率が{threshold_cpu}%以下でプログラム開始します")
                delayed_start(threshold_cpu)
            elif "s" in args.delayed_start:
                delayed_time = float(args.delayed_start.split("s")[0])
                print(f"{delayed_time}秒間sleepします")
                time.sleep(delayed_time)
            else :
                delayed_time = float(args.delayed_start)
                print(f"{delayed_time}秒間sleepします")
                time.sleep(delayed_time)

        if args.load_model is not None:
            print(args.load_model)
            model_spec = importlib.util.spec_from_file_location("load_models", args.load_model)
            model_func = importlib.util.module_from_spec(model_spec)
            model_spec.loader.exec_module(model_func)
            models = model_func(args.supervised)
        else:
            print("args.load_model is not given." )
            if args.load_model_config is not None:
                models = load_models(args.load_model_config)
            else:
                models = load_models()

        if args.load_dataset is not None:
            print(args.load_dataset)
            dataset_spec = importlib.util.spec_from_file_location("load_dataset", args.load_dataset)
            dataset_func = importlib.util.module_from_spec(dataset_spec)
            dataset_spec.loader.exec_module(dataset_func)
        else:
            print("args.load_dataset is not given." )
            dataset_func = load_dataset

        if args.load_dataset_config is not None:
            dataset_func = dataset_func(args.load_dataset_config)
        else:
            dataset_func = dataset_func()

        (X_raw_df,
        y_raw_df,
        time_raw_df,
        group_raw_df,
        evaluate_sample_weight_raw_df,
        training_sample_weight_raw_df,
        info_raw_df,
        raw_df,
        encoding) = dataset_func()

        config_path = args.ini_dataset
        if config_path is not None:
            dataset_func = partial(dataset_func, config_path=config_path)

        print(X_raw_df)
        print(y_raw_df)
        time.sleep(5)

        supervised_learning(
            models,
            X_raw_df,
            y_raw_df,
            raw_df=raw_df,
            info_df=info_raw_df,
            datetime_df=time_raw_df,
            group_df=group_raw_df,
            evaluate_sample_weight_df=evaluate_sample_weight_raw_df,
            training_sample_weight_df=training_sample_weight_raw_df,
            theme_name="")

if __name__ == "__main__":
    main()