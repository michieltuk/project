## Planning
# 1. Extract factors
#   a. Nelson-Siegel
#   b. PCA
#   c. Autoencoder
#   d. Spline
# 2. Forecast factors 
#   - MA
#   - AR
#   - ARMA
#   - VAR
#   - RNN
#   - ARIMA
#   - VAR + exogenous
#   - RNN + exogenous
#   - ARIMA + exogenous
# 3. Reconstruct yield curve
# 4. Compare accuracy

# To-do
# - RW for yields directly
# - Differencing for factors
# - Percentage in RMSE
# - Add MAE
# - Error charts/dashboards

# %% 

# Import libraries

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import test_normality
import copy
from importlib import reload

import yield_curve_model
reload(yield_curve_model)
from yield_curve_model import YieldCurveModel

import diagnostics
reload(diagnostics)

import charts
reload(charts)

from datetime import datetime
from load_data import load_FRED_data


import pickle

# def main():
# %%

# Load data
print('Starting script...')

offline = False
save_data = True

api_key = '78a6c6d6b67c353362f20cb69f158456'

series_ids = ['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10']#, 'DGS20', 'DGS30']
maturities = [1, 2, 3, 5, 7, 10]#, 20, 30]
start_date = None #'2015-01-01'
end_date = '2020-12-31'

params = {
    'series_id': None,
    'api_key': api_key,
    'observation_start': start_date,
    'observation_end': end_date,
    'file_type': 'json',
    'limit': 100000
}

if params['observation_start'] == None:
    params.pop('observation_start')

if offline:
    file = open('data.pkl', 'rb')
    data_clean = pickle.load(file)
    file.close()

else:
    data_raw = load_FRED_data(series_ids, params)
    data_clean = data_raw.dropna()

if save_data:
    file = open('data.pkl', 'wb')
    pickle.dump(data_clean, file)
    file.close()

print('Done loading data')

# %%
# Set up data

validation_holdout = 252 # 252*2

# Split estimation and validation data
data_estimation = data_clean.iloc[0:-validation_holdout]
data_validation = data_clean.iloc[-validation_holdout:]

# Define dates
dates_estimation = np.array([datetime.strptime(x, '%Y-%m-%d') for x in data_estimation.index.values])
dates_validation = np.array([datetime.strptime(x, '%Y-%m-%d') for x in data_validation.index.values])
dates = np.array([datetime.strptime(x, '%Y-%m-%d') for x in data_clean.index.values])

results_summary = []

# %%
# Set up runs

forecast_horizon = 66
reestimation_frequency = 126
reestimation_window = 3 * 252
burn_in = reestimation_window #252

runs = []

run_base = {
    'run_id': 1,
    'run_description': 'Base run',
    'factor_model': 'nelson-siegel',
    'factor_settings': {'shape_parameter': 1.0},
    'forecasting_model': 'var',
    'forecasting_settings': {'lag_order': 1},
    'forecasting_horizon': forecast_horizon,
    'reestimation_frequency': reestimation_frequency,
    'reestimation_window': reestimation_window,
    'burn_in': burn_in,
    'start_date_estimation': dates_estimation[0],
    'end_date_estimation': dates_estimation[-1],
    'start_date_validation': dates_validation[0],
    'end_date_validation': dates_validation[-1]
}

run1 = copy.deepcopy(run_base)
run1['run_id'] = 1
run1['run_description'] = 'VAR(1)'
run1['forecasting_settings']['lag_order'] = 1

run2 = copy.deepcopy(run_base)
run2['run_id'] = 2
run2['run_description'] = 'VAR(2)'
run2['forecasting_settings']['lag_order'] = 2

run3 = copy.deepcopy(run_base)
run3['run_id'] = 3
run3['run_description'] = 'VAR(3)'
run3['forecasting_settings']['lag_order'] = 3

run4 = copy.deepcopy(run_base)
run4['run_id'] = 4
run4['forecasting_model'] = 'arima'
run4['run_description'] = 'ARIMA(1,0,0)'
run4['forecasting_settings']['order'] = (1, 0, 0)

run5 = copy.deepcopy(run_base)
run5['run_id'] = 5
run5['forecasting_model'] = 'arima'
run5['run_description'] = 'ARIMA(2,1,2)'
run5['forecasting_settings']['order'] = (2, 1, 2)

run6 = copy.deepcopy(run_base)
run6['run_id'] = 6
run6['run_description'] = 'Random walk'
run6['forecasting_model'] = 'rw'
run6['forecasting_settings'] = {}

run7 = copy.deepcopy(run_base)
run7['run_id'] = 7
run7['forecasting_model'] = 'rnn'
run7['forecasting_settings']['ltsm_nodes'] = 3
run7['forecasting_settings']['time_steps'] = 22

#runs.append(run1)
#runs.append(run2)
# runs.append(run3)
#runs.append(run4)
runs.append(run5)
runs.append(run6)

# %% Execute all runs

yields = data_estimation.to_numpy()

for r in runs:
    print(f'''Starting run {r['run_id']}, {r['run_description']}''')

    print('Initialising yield curve model...')

    model = YieldCurveModel(yields,
                            maturities,
                            r['factor_model'],
                            r['factor_settings'],
                            r['forecasting_model'],
                            r['forecasting_settings'])
    
    print('Estimating factors...')
    factors = model.factor_estimates()

    print('Reconstructing yields...')
    yields_estimate = model.estimate_yields()
    r['reconstruction_metrics'] = diagnostics.performance_summary(yields, yields_estimate)

    print('Forecasting factors...')
    factors_forecast = model.rolling_factor_forecast(
        forecast_horizon=r['forecasting_horizon'], burn_in=r['burn_in'], reestimation_frequency=r['reestimation_frequency'], reestimation_window=r['reestimation_window'])
    r['factor_forecast_metrics'] = diagnostics.performance_summary(factors, factors_forecast)
    
    print('Forecasting yields...')
    yields_forecast = model.forecast_yields()
    r['yield_forecast_metrics'] = diagnostics.performance_summary(yields, yields_forecast)
    
    print('Plotting factors and factor forecasts...')
    charts.plot_forecasts(factors, factors_forecast,
                          dates_estimation, dates_estimation)

    #r.update(model.statistics)

results_summary.extend(runs)
results_summary_df = pd.DataFrame(results_summary)
print(results_summary_df)

# %%

import charts
reload(charts)

metrics = pd.DataFrame(list(results_summary_df['yield_forecast_metrics'].values))
metrics.set_index(results_summary_df['run_description'], inplace=True)

charts.plot_performance_metrics(metrics)
# %%
print('End of script')
exit()

# if __name__ == "__main__":
#     main()
#     exit()
