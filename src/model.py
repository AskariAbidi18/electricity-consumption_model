import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def xgBoost():
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',  # regression
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    num_round = 500
    model = xgb.train(
        params,
        dtrain,
        num_round,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=10
    )

    y_pred = model.predict(dtest)

    return model, y_pred



