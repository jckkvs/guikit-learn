
import copy
import datetime
import os
import math
import sys
import time

import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
from sklearn.preprocessing import FunctionTransformer

def round_significant_digits(value, display_digits=4):

    #if isinstance(value, int) == False and  isinstance(value, float) == False:
    try:
        value = float(value)
    except:
        #print('{} cannot convert to float'.format(value))
        #time.sleep(0.5)
        return value

    value = float(value)

    if value == 0:
        return 0

    elif value < 0:
        minus_flg = True
        value = abs(value)
    else:
        minus_flg = False

    value_digits = -int(math.log10(value)) + display_digits -1
    value_rounded = round(value, value_digits)

    if minus_flg == True:
        value_rounded = (-1) * value_rounded

    return value_rounded


def add_text_topng(original_image_path, theme_path, text_df):

    original_image = Image.open(original_image_path)
    image_widh = original_image.width / 100

    tmp_save_path = theme_path / 'tmp'
    os.makedirs(tmp_save_path, exist_ok=True)
    
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.subplot.left"]   = 0.03
    plt.rcParams["figure.subplot.bottom"] = 0.03  # 下余白 0.15
    plt.rcParams["figure.subplot.right"]  = 0.97  # 右余白 0.05
    plt.rcParams["figure.subplot.top"]    = 0.97  # 上余白 0.10
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = 'IPAexGothic'
    
    plt.tight_layout()
    # refer https://www.geeksforgeeks.org/matplotlib-axes-axes-table-in-python/
    fig, ax = plt.subplots(figsize=(image_widh,2.0))
    ax.axis('off')
    ax.axis('tight')

    #ax.set_axis_off()
    text_df = text_df.T
    table = ax.table(cellText=text_df.values,
                      colLabels=None,
                      loc='center',
                      cellLoc='left',
                      rowLoc='center',
                      colLoc='left',
                      colWidths = None,
                      edges='open',
                      bbox=None)

    #table.auto_set_font_size(True)
    table.set_fontsize(image_widh*2.5)
    table.scale(1,2)
    png_name  = 'score_text'
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    tmp_save_name =  tmp_save_path / f'{png_name}_{now}.png'
    plt.savefig(tmp_save_name)
    plt.close()
    time.sleep(1.5)

    add_text_image = Image.open(tmp_save_name)

    new_image = Image.new('RGB', (original_image.width , original_image.height + add_text_image.height))
    new_image.paste(original_image, (0, 0))
    new_image.paste(add_text_image, (0, original_image.height))
    new_image.save(original_image_path)

    return


def recursive_replace(_string: str, old: str, new: str) -> str:
    before_string = copy.deepcopy(_string)
    new_string = _string.replace(old, new)

    if before_string == new_string:
        return new_string
    else:
        return recursive_replace(new_string, old, new)


def get_estimator(
    model,
    model_type="estimator",
    remove_multioutput=False,
    remove_pipeline=True,
    remove_searcher=True,
):
    if model_type in ["pipeline", "Pipeline", "pipe", "Pipe"]:
        remove_pipeline = False

    if model_type == None:
        return model, "model"

    if (
        model.__class__.__name__
        in [
            "OptunaSearchCV",
            "GridSearchCV",
            "OptunaSearchClustering",
        ]
    ) and (remove_searcher == True):
        # print("SearchCV")
        # print(f"Search {model.estimator} {model_type}")
        if hasattr(model, "best_estimator_"):
            return get_estimator(
                model.best_estimator_,
                model_type,
                remove_multioutput,
                remove_pipeline,
            )
        else:
            return get_estimator(
                model.estimator, model_type, remove_multioutput, remove_pipeline
            )

    if model.__class__.__name__ == "Pipeline":
        if remove_pipeline == False:
            pass

        else:
            indexes = [
                idx
                for idx, (name, class_) in enumerate(model.steps)
                if model_type in str(name)
            ]
            if len(indexes) == 0:
                return FunctionTransformer(), ""
            else:
                # print(f"Search {model.steps[indexes[0]][1]} {model_type}")

                return get_estimator(
                    model.steps[indexes[0]][1],
                    model_type,
                    remove_multioutput,
                    remove_pipeline,
                )

    if model.__class__.__name__ in [
        "MultiOutputRegressor",
        "MultiOutputClassifier",
    ]:
        if remove_multioutput == True:
            return get_estimator(
                model.estimator, model_type, remove_multioutput, remove_pipeline
            )
        elif remove_multioutput == False:
            pass            

    if hasattr(model, "model_name") == True:
        model_name = model.model_name
        
    else:
        if model.__class__.__name__ in [
            "MultiOutputRegressor",
            "MultiOutputClassifier",
        ]:
            model_name = model.estimator.__class__.__name__
        else:
            model_name = model.__class__.__name__

    return model, model_name

