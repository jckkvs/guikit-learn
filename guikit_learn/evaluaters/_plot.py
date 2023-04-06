import gc
import os
import time
import traceback
import math

# Third-party Libary
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import shapiro



# from this library
# from ._base import add_text_topng, round_significant_digits
from ..utils.base import add_text_topng, round_significant_digits


def save_confusion_matrix(
    y_true,
    y_pred,
    *,
    save_path,
    save_name,
    X=None,
    plt_title="",
    plt_label="predict",
    score_dict=None,
    y_true_2=None,
    y_pred_2=None,
    plt_label_2="predict_2",
    score_dict_2=None,
    labels=None,
):
    if labels is None:
        labels = sorted(list(set(y_true)))

    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True, cmap="YlGn")

    plt.savefig(save_path / f"{save_name}_confusion_matrix.jpg", bbox_inches="tight")
    plt.close()


def save_scatter(
    y_true,
    y_pred,
    *,
    save_path,
    save_name,
    ax_max=None,
    ax_min=None,
    ticks=None,
    X=None,
    plt_title="",
    plt_label="predict",
    score_dict=None,
    y_true_2=None,
    y_pred_2=None,
    plt_label_2="predict_2",
    score_dict_2=None,
):

    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"] = 0.15
    plt.rcParams["figure.subplot.bottom"] = 0.15
    plt.rcParams["figure.subplot.right"] = 0.9  # 右余白 0.05
    plt.rcParams["figure.subplot.top"] = 0.9  # 上余白 0.10
    plt.rcParams["font.size"] = 24

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_subplot(111)
    plt.scatter(y_true, y_pred, label=plt_label, c="blue", s=60)

    if (y_true_2 is not None) and (y_pred_2 is not None):
        plt.scatter(y_true_2, y_pred_2, label=plt_label_2, c="green")

    # 主軸の調整

    max_y = max(y_true)
    min_y = min(y_true)

    tmp_min_x, tmp_max_x = plt.xlim()
    if ax_max is None:
        max_x = tmp_max_x
    if ax_min is None:
        min_x = tmp_min_x

    if ticks is None:
        ticks = ax.get_xticks()

    additional_line = np.array([min_x, max_x])
    plt.plot(additional_line, additional_line, c="grey", lw=3, linestyle="dotted")

    plt.xlim(min_x, max_x)
    plt.ylim(min_x, max_x)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_aspect("equal", adjustable="box")

    if score_dict is not None:
        if "r2" in score_dict.keys():
            r2 = score_dict["r2"]
            r2 = round(r2, 2)
        else:
            r2 = np.nan
        if "mae" in score_dict.keys():
            mae = score_dict["mae"]
            mae = round_significant_digits(mae, display_digits=4)
        else:
            mae = np.nan
    else:
        r2 = np.nan
        mae = np.nan

    plt.title(plt_title)
    plt.xlabel("実測値")
    plt.ylabel("予測値")
    plt.legend(loc=4)
    print(f"save_path {save_path}")
    plt.savefig(save_path / f"{save_name}_scatter.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()

    plt.figure(figsize=(7.8, 7.8))
    # 誤差の分布を示す
    y_error = y_true.reshape(-1, 1) - y_pred.reshape(-1, 1)
    plt.scatter(y_true, y_error, c="black", s=60)
    plt.title("実測 - 予測誤差")
    plt.xlabel("実測値")
    plt.ylabel("予測誤差")
    plt.savefig(save_path / f"{save_name}_error_scatter.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()

    plt.figure(figsize=(7.8, 7.8))
    # 誤差の分布を示す
    y_error = y_true.reshape(-1, 1) - y_pred.reshape(-1, 1)
    plt.scatter(y_true, np.abs(y_error), c="black", s=60)
    plt.title("実測 - 絶対誤差")
    plt.xlabel("実測値")
    plt.ylabel("絶対誤差")
    plt.savefig(save_path / f"{save_name}_abs_error_scatter.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()

    plt.figure(figsize=(7.8, 7.8))
    # 誤差の分布を示す
    y_error = y_true - y_pred
    shapiro_test = shapiro(y_error)
    p_value = shapiro_test[1]
    p_value = round_significant_digits(p_value, 3)
    plt.hist(y_error, bins=10)
    plt.title(f"予測誤差の分布 - p値 : {p_value} 正規性{p_value>=0.05}")
    plt.savefig(save_path / f"{save_name}_error_histogram.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()


def save_true_predict_result(
    y_true,
    y_pred,
    save_path,
    save_name,
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
            save_path,
            save_name,
            plt_title="",
            plt_label="predict",
            score_dict=None,
            y_true_2=None,
            y_pred_2=None,
            plt_label_2="predict_2",
            score_dict_2=None,
            labels=labels,
        )
    elif sv_type == "regressioin":
        save_scatter(
            y_true,
            y_pred,
            save_path,
            save_name,
            plt_title="",
            plt_label="predict",
            score_dict=None,
            y_true_2=None,
            y_pred_2=None,
            plt_label_2="predict_2",
            score_dict_2=None,
        )
    else:
        save_scatter(
            y_true,
            y_pred,
            save_path,
            save_name,
            plt_title="",
            plt_label="predict",
            score_dict=None,
            y_true_2=None,
            y_pred_2=None,
            plt_label_2="predict_2",
            score_dict_2=None,
        )
