data_clean = data_raw/100
data_clean.dropna(inplace=True)
data_array = np.array(data_clean)
dates = list(data_clean.index.values)

data_clean.plot()

from sklearn.linear_model import LinearRegression
import statistics

# Nelson-Siegel
def nelson_siegel(spot_rates, maturities, shape_parameter=1.0):
    x_1 = [np.exp(-x/shape_parameter) for x in maturities]
    x_2 = [x/shape_parameter * np.exp(-x/shape_parameter) for x in maturities]

    X = np.array([x_1, x_2]).T
    y = np.array(spot_rates)    
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    
    factors = np.insert(model.coef_, 0, model.intercept_)
    
    return y_hat, list(factors)

def nelson_siegel_predict(factors, maturities, shape_parameter=1.0):
    x_1 = [np.exp(-x/shape_parameter) for x in maturities]
    x_2 = [x/shape_parameter * np.exp(-x/shape_parameter) for x in maturities]
    
    X = np.array([x_1, x_2]).T
    X = np.insert(X, 0, 1, axis=1)

    yhat = X@factors.T
    
    return yhat

yhat = []
losses = []
factors = []

for y in data_array:
    yield_curve_prediction, factor = nelson_siegel(y, maturities, 1.0)
    yhat.append(yield_curve_prediction)
    losses.append(y - yield_curve_prediction)
    factors.append(factor)

factors = np.array(factors)
losses = np.array(losses)
yhat = np.array(yhat)

total_loss = np.mean([loss**2 for loss in losses])
print(total_loss)

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

for factor in factors.T:
    results = adfuller(factor)
    print(f'p-value of ADF test: {results[1]}')
    
factors_diff = np.diff(factors, axis=0)
print(factors_diff)

model = VAR(factors)
results = model.fit(1)
results.summary()
forecast = results.forecast(factors, 100)
results.plot_forecast(360)
yields_forecast = nelson_siegel_predict(forecast, maturities)
plt.figure()
full_series = np.append(data_array[-100:-1,0], yields_forecast[0])
plt.plot(full_series)

plt.figure()
plt.plot(factors)

# Print worst fit

worst_yield_curve_index = np.unravel_index(np.argmax(losses), losses.shape)[0]

plt.figure()
plt.title(dates[worst_yield_curve_index])
plt.plot(maturities,yhat[worst_yield_curve_index]*100, label="Predicted")
plt.plot(maturities, data_array[worst_yield_curve_index]*100, label="Observed")
plt.legend() 

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pprint

X = data_array
y = data_array

input_maturities = len(maturities)
hidden_nodes = 5
hidden_nodes_grid = [2, 3, 4, 5, 6]
factors = 4
K_fold_splits = 3
epochs = 5

folded_dataset = KFold(n_splits=K_fold_splits, random_state=1, shuffle=True)

loss_function = keras.losses.MeanSquaredError()

fold_nr = 1

results_summary = []
result = {}

results_summary_per_fold = []
result_per_fold = {}

for nr_hidden_nodes in hidden_nodes_grid:
    print(f'Number of hidden nodes: : {nr_hidden_nodes}')
    
    
    encoder = keras.models.Sequential([layers.Dense(hidden_nodes, input_shape=[input_maturities]),
                                  layers.Dense(factors)])

    decoder = keras.models.Sequential([layers.Dense(hidden_nodes, input_shape=[factors]),
                                      layers.Dense(input_maturities)])

    model = keras.Sequential([encoder, decoder])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    for train, validation in folded_dataset.split(X, y):

            result['nr_hidden_nodes'] = nr_hidden_nodes
            
            print(f'Fold number: {fold_nr}')
            result['fold'] = fold_nr

            output = model.fit(X[train], X[train], epochs=epochs, validation_data=(X[validation], X[validation]))

            result['train_loss'] = output.history['loss']
            result['validation_loss'] = output.history['val_loss']

            results_summary.append(result.copy())

            fold_nr = fold_nr + 1

    average_training_loss = statistics.mean([x['train_loss'][-1] for x in results_summary])
    average_validation_loss = statistics.mean([x['validation_loss'][-1] for x in results_summary])
         
    print(f'Average training loss: {average_training_loss}')
    print(f'Average validation loss: {average_validation_loss}')
    

pprint.pprint(results_summary[0])

average_training_loss = statistics.mean([x['train_loss'][-1] for x in results_summary])


average_training_loss

yhat = model.predict(data_array)
yhat

worst_yield_curve_index = np.unravel_index(np.argmax(losses), losses.shape)[0]

plt.figure()
plt.title(dates[worst_yield_curve_index])
plt.plot(maturities,yhat[worst_yield_curve_index]*100, label="Predicted")
plt.plot(maturities, data_array[worst_yield_curve_index]*100, label="Observed")
plt.legend()

import torch
import matplotlib.pyplot as plt

class yield_curve_autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        self.encoder = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Sigmoid(), torch.nn.Linear(8, 3))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Sigmoid(), torch.nn.Linear(8, 8))
        
        self.double()
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
model = yield_curve_autoencoder()
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)

loader = torch.utils.data.DataLoader(dataset = np.array(data_clean), batch_size = 32, shuffle = True)


epochs = 10
outputs = []
losses = []
for epoch in range(epochs):
    for yields in loader:

        # Output of Autoencoder
        reconstructed = model(yields)
        
        # Calculating the loss function
        loss = loss_function(reconstructed, yields)

        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss)
        outputs.append((epochs, yields, reconstructed))
  

# Plotting the last 100 values
#losses = losses.detach().numpy()

losses_clean = np.vstack([x.detach().numpy() for x in losses])

plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses_clean)
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(losses_clean[-100:])

outputs[0][2][1].detach()

plt.plot(outputs[-2][1][5])
plt.plot(outputs[-2][2][5].detach())

y = outputs[-2][1][3]
y_hat = outputs[-2][2][3].detach()

plt.plot(maturities, y_hat*100, label="Predicted")
plt.plot(maturities, y*100, label="Observed")
plt.legend()
