import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

import plotly

import plotly.io as pio
#pio.renderers.default = "vscode"
pio.templates.default = "simple_white"

def plot_forecasts(observed_values, forecasted_values, observed_dates=None,  forecasted_dates=None, labels=None, extend_forecast=True):
    
    if observed_dates is None:
        observed_dates = np.arange(len(observed_values))
    
    if forecasted_dates is None:
        if extend_forecast == True:
            forecasted_dates = np.arange(len(observed_values), len(observed_values) + len(forecasted_values))
        
        elif extend_forecast == False:
            forecasted_dates = np.arange(len(forecasted_values))


    cols = plotly.colors.qualitative.Plotly

    fig = go.Figure(layout=go.Layout(title=go.layout.Title(text="Forecasted values")))
    for i, series in enumerate(observed_values.T):
        fig.add_trace(go.Scatter(x=observed_dates, y=series, mode='lines', name=f'Observed {i+1}', line=dict(width=2, color=cols[i])))

    for i, series in enumerate(forecasted_values.T):
        fig.add_trace(go.Scatter(x=forecasted_dates, y=series, mode='lines', name=f'Forecasted {i+1}', line=dict(width=2, color=cols[i], dash='dot')))

    fig.show()

    return fig

def plot_performance_metrics(metrics):

    rows = metrics.shape[1]

    fig = plotly.subplots.make_subplots(rows=rows, cols=1)

    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(x=metrics.index, y=metrics[m], name=metrics.columns[i]),row=i+1, col=1)

    fig.show()

    return fig