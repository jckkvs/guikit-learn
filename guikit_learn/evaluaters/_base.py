import os
import math
from tkinter.tix import Y_REGION

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image

from ._plot import save_confusion_matrix, save_scatter


def save_true_predict_result(
    y_true,
    y_pred,
    *,
    save_path,
    save_name,
    ax_max=None,
    ax_min=None,
    X=None,
    X_names=None,
    y_names=None,
    plt_title="",
    plt_label="predict",
    score_dict=None,
    y_true_2=None,
    y_pred_2=None,
    plt_label_2="predict_2",
    score_dict_2=None,
    labels=None,
    sv_type="regression",
):

    if sv_type == "classification":
        save_confusion_matrix(
            y_true,
            y_pred,
            save_path=save_path,
            save_name=save_name,
            plt_title="",
            plt_label="predict",
            score_dict=score_dict,
            y_true_2=y_true_2,
            y_pred_2=y_pred_2,
            plt_label_2="plt_label_2",
            score_dict_2=score_dict_2,
            labels=labels,
        )

    elif sv_type == "regressioin":
        save_scatter(
            y_true,
            y_pred,
            save_path=save_path,
            save_name=save_name,
            ax_max=ax_max,
            ax_min=ax_min,
            plt_title="",
            plt_label="predict",
            score_dict=score_dict,
            y_true_2=y_true_2,
            y_pred_2=y_pred_2,
            plt_label_2="predict_2",
            score_dict_2=score_dict_2,
        )
    else:
        save_scatter(
            y_true,
            y_pred,
            save_path=save_path,
            save_name=save_name,
            ax_max=ax_max,
            ax_min=ax_min,
            plt_title="",
            plt_label="predict",
            score_dict=score_dict,
            y_true_2=y_true_2,
            y_pred_2=y_pred_2,
            plt_label_2="predict_2",
            score_dict_2=score_dict_2,
        )

    # true - predictをCSVに保存

    if y_names is None:
        if len(y_true.shape) == 2:
            n_y = y_true.shape[1]
        else:
            n_y = 1

        y_names = [f"y_{i}" for i in range(n_y)]

    y_true_names = [f"true_{i}" for i in y_names]
    y_pred_names = [f"pred_{i}" for i in y_names]

    y_true = pd.DataFrame(np.array(y_true), columns=y_true_names)
    y_pred = pd.DataFrame(np.array(y_pred), columns=y_pred_names)

    y_df = pd.concat([y_true, y_pred], axis=1)

    if X is not None:
        if X_names is None:

            if len(X.shape) == 2:
                n_X = X.shape[1]
            else:
                n_X = 1

            X_names = [f"y_{i}" for i in range(n_X)]
        X_df = pd.DataFrame(X, columns=X_names)
    else:
        X_df = pd.DataFrame()

    result_df = pd.concat([X_df, y_df], axis=1)

    result_df.to_csv(save_path / f"{save_name}.csv", encoding="shift_jisx0213")
