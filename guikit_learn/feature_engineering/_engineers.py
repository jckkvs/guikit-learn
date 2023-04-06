from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

def load_engineers():
    engineers = []

    # IdentityMapping - No scaling
    enginner = FunctionTransformer()
    enginner.model_name = "IdentityMapping"
    enginner.step_name = "engineer"
    enginner.default = True
    engineers.append(enginner)

    # PolynomialFeatures
    enginner = PolynomialFeatures()
    enginner.model_name = "PolynomialFeatures"
    enginner.step_name = "engineer"
    enginner.default = False
    engineers.append(enginner)


    return engineers