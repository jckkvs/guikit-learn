import configparser
import os
from pathlib import Path
import pandas as pd
import sys

def get_file_path(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen
        print("frozen")
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        print(".py")
        datadir = os.path.dirname(__file__)
    return Path(datadir) / filename # os.path.join(datadir, filename)

def load_dataset(config_path=get_file_path("default.ini")):
    """設定ファイルからCSVを読込、X,y,groups等を分割する

    Parameters
    ----------
    config_path : str(path)
        setting.ini

    Returns
    -------
    tuple
        tuple of DataFrame
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    csv_path = config["CSV"]["csv_path"]
    method = config["CSV"]["method"]
    encoding = config["CSV"].get("encoding")

    if method == "n_column":
        n_column_timeseries = config["n_column"].getint("timeseries")
        n_column_info = config["n_column"].getint("info")
        n_column_group = config["n_column"].getint("group")
        n_column_evaluate_sample_weight = config["n_column"].getint("evaluate_sample_weight")
        n_column_training_sample_weight = config["n_column"].getint("training_sample_weight")
        n_column_X = config["n_column"].getint("X")
        n_column_y = config["n_column"].getint("y")

        iloc_column_timeseries = range(0, n_column_timeseries)
        iloc_column_info = range(n_column_timeseries, n_column_timeseries+n_column_info)
        iloc_column_group = range(n_column_timeseries+n_column_info, n_column_timeseries+n_column_info + n_column_group)
        iloc_column_evaluate_sample_weight = range(n_column_timeseries+n_column_info + n_column_group,
                                                n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight)
        iloc_column_training_sample_weight = range(n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight,
                                                n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight + n_column_training_sample_weight)
        iloc_column_X = range(n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight + n_column_training_sample_weight,
                            n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight + n_column_training_sample_weight + n_column_X)
        iloc_column_y = range(n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight + n_column_training_sample_weight + n_column_X,
                            n_column_timeseries+n_column_info + n_column_group + n_column_evaluate_sample_weight + n_column_training_sample_weight + n_column_X + n_column_y)

    elif method == "iloc":
        iloc_column_timeseries = args_to_iloc_list(config["iloc"].get("timeseries", []))
        iloc_column_info = args_to_iloc_list(config["iloc"].get("info", []))
        iloc_column_group = args_to_iloc_list(config["iloc"].get("group", []))
        iloc_column_evaluate_sample_weight = args_to_iloc_list(config["iloc"].get("evaluate_sample_weight", []))
        iloc_column_training_sample_weight = args_to_iloc_list(config["iloc"].get("training_sample_weight", []))
        iloc_column_X = args_to_iloc_list(config["iloc"].get("X"))
        iloc_column_y = args_to_iloc_list(config["iloc"].get("y", []))

    raw_df = pd.read_csv(csv_path, encoding=encoding)
    time_raw_df = raw_df.iloc[:, iloc_column_timeseries]
    info_raw_df = raw_df.iloc[:, iloc_column_info]
    group_raw_df = raw_df.iloc[:, iloc_column_group]
    evaluate_sample_weight_raw_df = raw_df.iloc[:, iloc_column_evaluate_sample_weight]
    training_sample_weight_raw_df = raw_df.iloc[:, iloc_column_training_sample_weight]
    X_raw_df = raw_df.iloc[:, iloc_column_X]
    y_raw_df = raw_df.iloc[:, iloc_column_y]

    return (X_raw_df,
        y_raw_df,
        time_raw_df,
        group_raw_df,
        evaluate_sample_weight_raw_df,
        training_sample_weight_raw_df,
        info_raw_df,
        raw_df,
        encoding)

def args_to_iloc_list(string):
    """文字列をilocで利用可能なlistに変換する

    Parameters
    ----------
    string : str
        string specified iloc

    Returns
    -------
    list
        list of iloc

    Examples
    --------
    >>> string = "1:5"
    >>> args_to_iloc_list(string)
    >>> [1,2,3,4]

    >>> string = "1:5, 7"
    >>> args_to_iloc_list(string)
    >>> [1,2,3,4,7]

    """
    if string is None:
        return []

    strings = string.split(",")
    iloc_list = []
    for each_string in strings:
        each_string = each_string.strip()
        if ":" in each_string:
            start_ , end_  = each_string.split(":")
            for i in range(int(start_), int(end_)):
                iloc_list.append(i)
        else:
            iloc_list.append(int(each_string))

    return iloc_list
