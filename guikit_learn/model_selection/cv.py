from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, GroupKFold
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit 
from sklearn_expansion.model_selection import WalkForwardValidation

def load_splliter_dict(): 
    randint = 42
    splitter_dict = {
        "KFold": {
            "display_name": "KFold",
            "instance": KFold(shuffle=True),
            "default": False,
            "argument": f"",
        },
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
    return splitter_dict