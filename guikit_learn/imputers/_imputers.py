from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer

def load_imputers():
    imputers = []

    # IdentityMapping - No scaling
    imputer = FunctionTransformer()
    imputer.model_name = "IdentityMapping"
    imputer.step_name = "imputer"
    imputer.default = True
    imputers.append(imputer)

    # SimpleImputer
    imputer = SimpleImputer()
    imputer.model_name = "SimpleImputer"
    imputer.step_name = "imputer"
    imputer.default = False
    imputers.append(imputer)

    # KNNImputer
    imputer = KNNImputer()
    imputer.model_name = "KNNImputer"
    imputer.step_name = "imputer"
    imputer.default = False
    imputers.append(imputer)

    # IterativeImputer
    imputer = IterativeImputer()
    imputer.model_name = "IterativeImputer"
    imputer.step_name = "imputer"
    imputer.default = False
    imputers.append(imputer)

    return imputers