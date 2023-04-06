
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def load_models():
    """シンプルなload_models関数の例

    Returns
    -------
    models: list of estimator
    """
    models = []
    models.append(LinearRegression())
    models.append(RandomForestRegressor())

    return models

if __name__ == "__main__":
    load_models()