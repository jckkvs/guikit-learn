from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler

def load_scalers():
    scalers = []

    # IdentityMapping - No scaling
    scaler = FunctionTransformer()
    scaler.model_name = "IdentityMapping"
    scaler.step_name = "scaler"
    scaler.default = False
    scalers.append(scaler)

    # StandardScaler
    scaler = StandardScaler()
    scaler.model_name = "StandardScaler"
    scaler.step_name = "scaler"
    scaler.default = True

    scalers.append(scaler)

    # MinMaxScaler
    scaler = MinMaxScaler()
    scaler.model_name = "MinMaxScaler"
    scaler.step_name = "scaler"
    scaler.default = False

    scalers.append(scaler)
    return scalers