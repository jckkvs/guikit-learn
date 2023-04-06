

import gc
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
import seaborn as sns
import shap
from scipy.stats import shapiro
import numpy as np

# from original library
from sklearn_expansion.metrics import calc_corr

# from this library
#from ._base import add_text_topng, round_significant_digits
from ..utils.base import add_text_topng, round_significant_digits


matplotlib.use("Agg")

def shapiro_wilk(X, y, save_path):
    df = X
    for y_idx, each_y_type in enumerate(y.dtypes):
        if each_y_type != np.dtype('object'):
            df = pd.concat([df,y.iloc[:, y_idx]], axis=1)


    feature_names = df.columns

    df = df.apply(lambda x :pd.Series(shapiro(x), index=['W','P']))
    df = df.T
    df['p_0.05'] = [True if _p > 0.05 else False for _p in df['P']] 
    df['p_0.01'] = [True if _p > 0.01 else False for _p in df['P']] 
    df = df.T

    df.columns = feature_names

    df.to_csv(save_path / 'shapiro_wilk.csv', encoding = 'shift_jisx0213')
    return 


def correlation_coefficient(X, y, save_path, method='pearson'):
    df = pd.concat([X,y], axis=1)

    correlation_df = calc_corr(df, method=method)

    sns.heatmap(correlation_df, cmap= sns.color_palette('coolwarm', 11),annot=False, linewidths = .5, vmin =-1.0 , vmax=1.0, square=True)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.rcParams["figure.autolayout"] = True
    plt.savefig(save_path / f'correlation_coefficient_{method}.png', dpi = 100, bbox_inches="tight")
    plt.close()


    # 相関係数のランキング作成
    correlation_rank_df = pd.DataFrame()
    
    for column_name, item in correlation_df.iteritems():
        item_index = item.index
        n_index = len(item_index)
        #new_item_index = [i + ':' +  column_name for i in _item_index]
        #item.index  = new_item_index
        index_df = pd.DataFrame(item.index)
        index_df.index=item.index
        item = pd.concat([index_df, pd.DataFrame([column_name]*n_index, index=item_index),item],axis=1)

        item.columns = ['v0','v1', 'corr']
        correlation_rank_df = pd.concat([correlation_rank_df, item], axis = 0)


    correlation_rank_df.columns= ['v0', 'v1', 'corr']
    correlation_rank_df['abs_corr'] = correlation_rank_df['corr'].abs()

    # 相関係数の絶対値が0.99を超える組合せを削除

    correlation_rank_df.to_csv(save_path / f'correlation_coefficient_{method}.csv', encoding = 'shift_jisx0213', index=False)

    correlation_rank_df = correlation_rank_df[correlation_rank_df['abs_corr'] <= 0.99]
    correlation_rank_sorted_df = correlation_rank_df.sort_values('abs_corr', ascending = False)
    correlation_rank_sorted_df = correlation_rank_sorted_df[::2]
    correlation_rank_sorted_df.to_csv(save_path / f'correlation_rank_{method}.csv', encoding = 'shift_jisx0213', index=False)
    return 


def pairplot(X, y, save_path):
    df = pd.concat([X,y], axis=1)
    row_size , col_size = df.shape

    if col_size >= 25:
        return


    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"]   = 0.05  # 左余白 0.25 軸表示するため
    plt.rcParams["figure.subplot.bottom"] = 0.05  # 下余白 0.15
    plt.rcParams["figure.subplot.right"]  = 0.95  # 右余白 0.05
    plt.rcParams["figure.subplot.top"]    = 0.95  # 上余白 0.10
    plt.rcParams["font.size"] = 18

    plt.figure(figsize=(15,15))
    sns.pairplot(df)
    plt.savefig(save_path/ 'pairplot.png', dpi=100 ,bbox_inches="tight")
    plt.close()
    plt.rcParams["figure.autolayout"] = True

    return