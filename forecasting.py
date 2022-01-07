from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

import numpy as np

class VARForecasting:
    def __init__(self, settings):
        self.lag_order = settings['lag_order']

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None):

        if burn_in is None:
            burn_in = 100

        forecasts = np.empty(factors.shape)
        forecasts[:] = np.NaN

        for t in range(burn_in, len(factors)-forecast_horizon):
            sample = factors[:t, :]
            forecast = self.forecast(sample, forecast_horizon=forecast_horizon)
            forecasts[t+forecast_horizon] = forecast[-1, :]

        return forecasts

    def forecast(self, factors, forecast_horizon=1):
        model = VAR(factors)
        results = model.fit(self.lag_order)
        forecast = results.forecast(factors[-self.lag_order:, :], forecast_horizon)

        return forecast

class ARIMAForecasting:
    def __init__(self, settings):
        self.order = settings['order']

        if (self.order[1] == 0) & (self.order[2] == 0):
            self.method = 'yule_walker'
        else:
            self.method = 'innovations_mle'

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None):

        forecasts = np.empty(factors.shape)
        forecasts[:] = np.NaN

        for t in range(burn_in, len(factors)-forecast_horizon):
            
            print(f'Estimating ARIMA for t={t}/{len(factors)-forecast_horizon}')
            sample = factors[:t, :]
            forecast = self.forecast(sample, forecast_horizon=forecast_horizon)
            forecasts[t+forecast_horizon] = forecast[-1, :]

        return forecasts

    def forecast(self, factors, forecast_horizon=1):

        forecast = np.empty((forecast_horizon, factors.shape[1]))

        for i, f in enumerate(factors.T):

            model = ARIMA(endog=f, order=self.order)
            results = model.fit(method=self.method)
            factor_forecast = results.forecast(steps=forecast_horizon)
            forecast[:, i] = factor_forecast

        return forecast

class RWForecasting:

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None):

        forecasts = np.roll(factors, forecast_horizon, axis=0)
        forecasts[0:forecast_horizon, :] = np.NaN

        return forecasts

    def forecast(self, factors, forecast_horizon=1):

        forecast = np.tile(factors[-1,:], (forecast_horizon, 1))

        return forecast
