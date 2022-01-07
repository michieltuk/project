import sklearn.metrics

import numpy as np

def RMSE(y, yhat, per_column=False):
    
    if per_column:
        rmse = []
        for y_series, yhat_series in zip(y.T, yhat.T):
             rmse.append(sklearn.metrics.mean_squared_error(y_series, yhat_series, squared=False))
    else:
        rmse = sklearn.metrics.mean_squared_error(y, yhat, squared=False)

    return rmse
    

def MSE(y, yhat, per_column=False):
    
    if per_column:
        rmse = []
        for y_series, yhat_series in zip(y.T, yhat.T):
             rmse.append(sklearn.metrics.mean_squared_error(y_series, yhat_series, squared=True))
    else:
        rmse = sklearn.metrics.mean_squared_error(y, yhat, squared=True)

    return rmse

def performance_summary(y, yhat):

    y = y[~np.isnan(yhat).any(axis=1),:]
    yhat = yhat[~np.isnan(yhat).any(axis=1),:]

    summary = {}
    summary['rmse'] = RMSE(y, yhat)
    #summary['rmse_per_maturity'] = RMSE(y, yhat, per_column=True)
    summary['mse'] = MSE(y, yhat)
    #summary['mse_per_maturity'] = MSE(y, yhat, per_column=True)

    return summary
