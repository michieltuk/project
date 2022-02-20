import numpy as np

import sklearn.metrics

# Import classes for forecasting
from forecasting import VARForecasting
from forecasting import ARIMAForecasting
from forecasting import RWForecasting

# Import classes for yield curve factors
from nelson_siegel import NelsonSiegel

class YieldCurveModel:

    def __init__(self, yields, maturities, factor_model, factor_model_settings, forecasting_model, forecasting_model_settings):

        self.yields = yields
        self.maturities = maturities
        self.factor_model = factor_model
        self.factor_model_settings = factor_model_settings
        self.forecasting_model = forecasting_model
        self.forecasting_model_settings = forecasting_model_settings

        self.statistics = {}

        # Link factor models with models
        # Factor classes need to have the following function(s):
        # - factor_estimates(yields)
        # - yield_estimates(factors)

        if self.factor_model == 'nelson-siegel':
            self.factor_model_class = NelsonSiegel(
                self.maturities, self.factor_model_settings['shape_parameter'])

        # Link factor forecasting models 
        # Forecasting classes need to have the following function(s):
        # - rolling_factor_forecast(factors, forecast_horizon, burn_in)

        if self.forecasting_model == 'var':
            self.forecasting_class = VARForecasting(
                self.forecasting_model_settings)

        if self.forecasting_model == 'arima':
            self.forecasting_class = ARIMAForecasting(
                self.forecasting_model_settings)

        if self.forecasting_model == 'rw':
            self.forecasting_class = RWForecasting()

    def rolling_factor_forecast(self, forecast_horizon=1, burn_in=100, reestimation_frequency=1, reestimation_window=None):

        self.factor_forecasts = self.forecasting_class.rolling_forecast(
            self.factor_estimates, forecast_horizon=forecast_horizon, burn_in=burn_in, reestimation_frequency=reestimation_frequency, reestimation_window=reestimation_window)

        self.statistics['factor_forecasts_RMSE'] = sklearn.metrics.mean_squared_error(
            self.factor_estimates[burn_in+forecast_horizon, :], self.factor_forecasts[burn_in+forecast_horizon, :], squared=False)

        return self.factor_forecasts

    def factor_estimates(self):

        self.factor_estimates = self.factor_model_class.factor_estimates(
            self.yields)

        return self.factor_estimates

    def estimate_yields(self):

        self.yield_estimates = self.factor_model_class.yield_estimates(
            self.factor_estimates)

        self.statistics['yield_estimates_RMSE'] = sklearn.metrics.mean_squared_error(
            self.yields, self.yield_estimates, squared=False)

        return self.yield_estimates

    def forecast_yields(self):

        self.yield_forecasts = self.factor_model_class.yield_estimates(
            self.factor_forecasts)

        valid_value = ~np.isnan(self.yield_forecasts)

        self.statistics['yield_forecasts_RMSE'] = sklearn.metrics.mean_squared_error(
            self.yields[valid_value], self.yield_forecasts[valid_value], squared=False)

        return self.yield_forecasts


