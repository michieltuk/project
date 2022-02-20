from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

import numpy as np

class VARForecasting:
    def __init__(self, settings):
        self.lag_order = settings['lag_order']

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None, reestimation_frequency=1, reestimation_window=None):

        if burn_in is None:
            burn_in = 100

        forecasts = np.empty(factors.shape)
        forecasts[:] = np.NaN

        for t in range(burn_in, len(factors)-forecast_horizon):

            sample = factors[max(t-reestimation_window, 0):t, :]

            if ((t-burn_in) % reestimation_frequency) == 0:
                print(f'Re-estimating VAR at t = {t}/{len(factors)-forecast_horizon}')
                model = VAR(sample)
                results = model.fit(self.lag_order)

            forecast = results.forecast(sample[-self.lag_order:, :], forecast_horizon)
            forecasts[t+forecast_horizon] = forecast[-1, :]

            #forecast = self.forecast(sample, forecast_horizon=forecast_horizon)
            #forecasts[t+forecast_horizon] = forecast[-1, :]

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
            self.method = 'statespace'

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None, reestimation_frequency=1, reestimation_window=None):

        forecasts = np.empty(factors.shape)
        forecasts[:] = np.NaN  

        for i, f in enumerate(factors.T):

            for t in range(burn_in, len(factors)-forecast_horizon):

                sample = f[max(t-reestimation_window, 0):t]
                forecast = np.empty(forecast_horizon)

                if ((t-burn_in) % reestimation_frequency) == 0:
                    print(f'Re-estimating ARIMA for factor {i+1} at t = {t}/{len(factors)-forecast_horizon}')
                    model = ARIMA(endog=sample, order=self.order)
                    results = model.fit(method=self.method)

                else:
                    results = results.append([sample[-1]])

                factor_forecast = results.forecast(steps=forecast_horizon)
                forecasts[t+forecast_horizon, i] = factor_forecast[-1]
            
            #forecasts[t+forecast_horizon] = forecast[-1, :]                
            
            #print(f'Estimating ARIMA for t={t}/{len(factors)-forecast_horizon}')
            
            #forecast = self.forecast(sample, forecast_horizon=forecast_horizon)
            

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

    def rolling_forecast(self, factors, forecast_horizon=1, burn_in=None, reestimation_frequency=1, reestimation_window=None):

        forecasts = np.roll(factors, forecast_horizon, axis=0)
        forecasts[0:forecast_horizon, :] = np.NaN

        return forecasts

    def forecast(self, factors, forecast_horizon=1):

        forecast = np.tile(factors[-1,:], (forecast_horizon, 1))

        return forecast
