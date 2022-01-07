
# %%

# Machine learning libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Utils
import matplotlib.pyplot as plt
import datetime
import pickle


from nelson_siegel import NelsonSiegel


# %%

# Load data

file = open('data.pkl', 'rb')
data_clean = pickle.load(file)
file.close()

maturities = [1, 2, 3, 5, 7, 10]#, 20, 30]
nelsonsiegel = NelsonSiegel(maturities, 1.0)
factors = nelsonsiegel.factor_estimates(data_clean.values)
yield_estimates = nelsonsiegel.yield_estimates(factors)

# %%

#train_factors = factors[-1200:-200, :]
#validation_factors = factors[-200:, :]

train_factors = factors[:-1000, :]
validation_factors = factors[-1000:,:]

# Pre-process data

scaler = StandardScaler()
scaled_train_factors = scaler.fit_transform(train_factors)
scaled_validation_factors = scaler.fit_transform(validation_factors)

plt.figure()
plt.plot(scaled_train_factors)
plt.figure()
plt.plot(scaled_validation_factors)

# %%

# Structure data for NN

forecasting_horizon = 50
time_steps = 100
#batch_size = 32

X_train = []
y_train = []

for t in range(time_steps, len(scaled_train_factors)):
    X_train.append(scaled_train_factors[t-time_steps:t, :])
    y_train.append(scaled_train_factors[t+forecasting_horizon, :])

X_train, y_train = np.array(X_train), np.array(y_train)

X_validation = []
y_validation = []

for t in range(time_steps, len(scaled_validation_factors)-forecasting_horizon):
    X_validation.append(scaled_validation_factors[t-time_steps:t, :])
    y_validation.append(scaled_validation_factors[t+forecasting_horizon, :])

X_validation, y_validation = np.array(X_validation), np.array(y_validation)

# X_train = np.split(X_train, np.arange(batch_size, len(X_train), batch_size))
# X_train.pop()

# y_train = np.split(y_train, np.arange(batch_size, len(y_train), batch_size))
# y_train.pop()


# %%

# Create RNN

features = X_train.shape[2]
hidden_units = 3
LTSM_units = 5
LTSM_dropout = 0
batch_nr = len(X_train)

model = keras.Sequential()
model.add(keras.layers.LSTM(units = LTSM_units, dropout=LTSM_dropout, stateful=False, return_sequences=False, input_shape = (time_steps, features)))
#model.add(keras.layers.Dense(units = hidden_units, activation='linear'))
model.add(keras.layers.Dense(units = features, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=5, batch_size=1, callbacks=[tensorboard_callback])

# %%

yhat_train = model.predict(X_train)
yhat_validation = model.predict(X_validation)

# %%

from importlib import reload
import charts
reload(charts)
import charts

charts.plot_forecasts(y_train, yhat_train, extend_forecast=False)
charts.plot_forecasts(y_validation, yhat_validation, extend_forecast=False)


# %%

class Nelson_Siegel(keras.layers.Layer):
    def __init__(self, maturities, shape_parameter):
        super(Nelson_Siegel, self).__init__()

        #self.output_shape = (6,)

        w_1 = [np.exp(-x/shape_parameter) for x in maturities]
        w_2 = [x/shape_parameter*np.exp(-x/shape_parameter) for x in maturities]

        W = np.array([w_1, w_2]).T
        W = np.insert(W, 0, 1, axis=1)

        W = tf.convert_to_tensor(W, dtype="float32")

        self.w = tf.Variable(initial_value=W, trainable=False )
        #self.b = tf.Variable(initial_value=tf.ones(len(maturities)), trainable=False )


    def call(self, inputs):
        inputs = tf.reshape(inputs, (3, 1))
        outputs = tf.matmul(self.w, inputs)
        return tf.reshape(outputs, (-1,6))

nelson_siegel_layer = Nelson_Siegel(maturities, 1.0)

#test_factors = tf.reshape(tf.convert_to_tensor([1.0, 1.0, 1.0]), (3, 1))

test_factors = tf.reshape(tf.convert_to_tensor(factors[0], dtype='float32'), (3,1))

test_yields = nelson_siegel_layer(test_factors)

plt.plot(test_yields)

# tf.convert_to_tensor(factors[0])
# %%

X_yields = data_clean.to_numpy()

batch_size = 1
hidden_factors = 3
input_shape = (len(maturities),)

autoencoder = keras.Sequential()
autoencoder.add(keras.layers.Dense(units=hidden_factors, input_shape=input_shape, activation='linear'))
autoencoder.add(nelson_siegel_layer)
autoencoder.add(keras.layers.Dense(6))

autoencoder.compile(loss='mean_squared_error', optimizer='adam')

autoencoder.fit(X_yields, X_yields, epochs=3, batch_size=batch_size)

# %%

#yhat = autoencoder.predict(tf.reshape(tf.convert_to_tensor(X_yields[0]), (1,6)))

#inputs = tf.reshape(tf.convert_to_tensor(X_yields, dtype='float32'), (-1, 6))
inputs = X_yields

yhat = autoencoder.predict(inputs, batch_size=1)

#yhat = autoencoder.predict_on_batch(inputs)

print(autoencoder.summary())


# %%
# Structure data for NN

forecasting_horizon = 50
time_steps = 50
#batch_size = 32
validation_window = 1000

X = data_clean.to_numpy()
X_raw_train = data_clean.iloc[:-validation_window, :].to_numpy()
X_raw_validation = data_clean.iloc[-validation_window:, :].to_numpy()

X_train = []
y_train = []

for t in range(time_steps, len(X_raw_train)-forecasting_horizon):
    X_train.append(X_raw_train[t-time_steps:t, :])
    y_train.append(X_raw_train[t+forecasting_horizon, :])

X_train, y_train = np.array(X_train), np.array(y_train)

X_validation = []
y_validation = []

for t in range(time_steps, len(X_raw_validation)-forecasting_horizon):
    X_validation.append(X_raw_validation[t-time_steps:t, :])
    y_validation.append(X_raw_validation[t+forecasting_horizon, :])

X_validation, y_validation = np.array(X_validation), np.array(y_validation)

# %%

LTSM_units = 3
hidden_factors = 3
features = len(maturities)
time_steps = X_train.shape[1]

#nelson_siegel_layer = Nelson_Siegel(maturities, 1.0)

model = keras.Sequential()
model.add(keras.layers.LSTM(units=LTSM_units, input_shape=(time_steps, features), stateful=False, return_sequences=False))
model.add(keras.layers.Dense(hidden_factors, activation='linear'))
model.add(Nelson_Siegel(maturities, 1.0))

model.compile(loss='mean_squared_error', optimizer='adam')

log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S-LTSM_{LTSM_units}")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=3, batch_size=1, callbacks=[tensorboard_callback])

yhat_train = model.predict(X_train, batch_size=1)
yhat_validation = model.predict(X_validation, batch_size=1)

# %%

from importlib import reload
import charts
reload(charts)
import charts

maturity_to_plot = 4

charts.plot_forecasts(y_train[:, 1:3], yhat_train[:, 1:3], extend_forecast=False)
charts.plot_forecasts(y_validation[:, 1:3], yhat_validation[:, 1:3], extend_forecast=False)


# %%
