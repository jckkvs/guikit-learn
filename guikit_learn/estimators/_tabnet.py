#from pytorch_tabnet.pretraining import TabNetPretrainer
#from pytorch_tabnet.tab_model import TabNetRegressor
#from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn_expansion.tabnet import TabNetRegressor

def load_regression_tabnet():
    models_set = {}

    # tabnet without pretrain

    tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                     n_independent=2, n_shared=2,
                     seed=0, lambda_sparse=1e-3, 
                     optimizer_fn=torch.optim.Adam, 
                     optimizer_params=dict(lr=2e-2),
                     mask_type="entmax",
                     scheduler_params=dict(mode="min",
                                           patience=5,
                                           min_lr=1e-5,
                                           factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10
                    )

    fit_params = dict(max_epochs=5000)

    estimator =  TabNetRegressor(**tabnet_params)
    param_range = {}
    param_grid = []

    model_name = 'tabnet'
    model = {'estimator':estimator,
             'default':True,
             'model_name':model_name ,
             'param_range':param_range,
             'param_grid':param_grid,
             'fit_params':fit_params}


    models_set[model_name] = model


    # tabnet with pretrain


    return models_set

def load_tabnet(ml_type='regression'):
    if ml_type == 'regression':
        models = load_regression_tabnet()
    else:
        raise ValueError('Models other than regression have not yet been defined!')
    return models


