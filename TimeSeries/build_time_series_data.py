# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GjiigBZ_sEM7c2Teqqop7V111MmUGtgF

## Xây dựng dataset
"""

import tensorflow as tf

dataset = tf.data.Dataset.range(10)

dataset

x = list(dataset.as_numpy_iterator())

x

dataset.take(1)

"""Sử dụng window trượt qua 5 phần tử, mỗi lần trượt này cách nhau 1."""

dataset = dataset.window(5, shift=1, drop_remainder=True)

for window in dataset:
  for val in window:
    print(val.numpy(), end=" ")
  print()

dataset = dataset.flat_map(lambda window: window.batch(5))

for window in dataset:
  print(window.numpy())

"""Chia feature và nhãn"""

dataset = dataset.map(lambda window: (window[:-1], window[-1]))

for window in dataset:
  print(window[0], window[1])

list(dataset.take(1))[0][0].shape

list(dataset.take(1))[0][1].shape

dataset = dataset.shuffle(buffer_size=10)

for window in dataset:
  print(window)

dataset = dataset.batch(2)

dataset

list(dataset.take(1))

# for window in dataset:
#   print(window)

"""## Mock up dữ liệu"""

import numpy as np
import matplotlib.pyplot as plt
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

plot_series(time, series)

time

series

window_size = 20
batch_size = 32
shuffle_size = 1000

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

# input: t=1 -> t=20
# output: t=2 -> t=21
# window_size = 21


# input: t=1 -> t=20
# output: t=21 -> t=40
# window_size = 40


# input: t=1 -> t=16
# output: t=10 -> t=40
# window_size = 40

window_size = 40

def windowed_ds(series, window_size, batch_size, shuffle_size):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  
  dataset = dataset.window(window_size, shift=1, drop_remainder=True)
  
  dataset = dataset.flat_map(lambda window: window.batch(window_size))
  
  dataset = dataset.shuffle(shuffle_size).map(lambda window: (window[:-1], window[-1]))
  
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

series

dataset2 = windowed_ds(series, window_size, batch_size, shuffle_size)

dataset2

list(dataset2.take(1))

list(dataset2.take(1))[0][0].shape

list(dataset2.take(1))[0][1].shape

"""## Sử dụng mạng nơ ron

Input: (batch_size, 16) 

Ouput: (batch_size, 30)
"""

model = tf.keras.models.Sequential([
                                    
])

dense1 = tf.keras.layers.Dense(10, input_shape=[19,], activation='relu')
dense2 = tf.keras.layers.Dense(10, activation='relu')
dense3 = tf.keras.layers.Dense(1)

# model = tf.keras.models.Sequential([
#     dense1,
#     dense2,
#     dense3,                    
# ])

model.add(dense1)
model.add(dense2)
model.add(dense3)

model.summary()

model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), metrics=['mae'])
model.fit(dataset2, epochs=100, verbose=1)

# forecast = []
# for time in range(len(series) - window_size):
#   forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# forecast = forecast[split_time-window_size:]
# results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)



split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda window: window.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda window: (window[:-1], window[1:]))
  ds = ds.batch(batch_size).prefetch(1)
  return ds

def model_forecast(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size, shift=1 , drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size))
  ds = ds.batch(32).prefetch(1)
  forecast = model.predict(ds)
  return forecast

"""## Sử dụng model RNN"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Bidirectional
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint

# clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)

# window_size = 64
# batch_size = 256

# train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# print(train_set)
# print(x_train.shape)

# def model(input_shape):
#     input = Input(shape=input_shape)
#     x = input
#     x = Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu')(x)
#     x = LSTM(64, return_sequences=True)(x)
#     x = LSTM(64, return_sequences=True)(x)
#     x = Dense(30, activation='relu')(x)
#     x = Dense(10, activation='relu')(x)
#     x = Dense(1)(x)
#     model = Model(input, x)
#     return model
# model = Sequential([
#                     Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]),
#                     LSTM(64, return_sequences=True),
#                     LSTM(64, return_sequences=True),
#                     Dense(30, activation='relu'),
#                     Dense(10, activation='relu'),
#                     Dense(1),
#                     Lambda(lambda x: x * 400)
# ])
# for tensor in model.layers:
#   print(tensor)
# model = model(input_shape=[None, 1])
# model.summary()

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
# model.compile(loss=tf.keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
# history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([1e-8, 1e-4, 0, 60])

checkpoint = ModelCheckpoint(filepath='model.h5', monitor='mae', verbose=0, save_best_only=True)

clear_session()
tf.random.set_seed(0)
np.random.seed(0)

train_set = windowed_dataset(x_train, window_size=60, batch_size=64, shuffle_buffer=shuffle_buffer_size)
model = Sequential([
                    Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=[None, 1]),
                    LSTM(64, return_sequences=True),
                    LSTM(64, return_sequences=True),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1)
])
# optim = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])
history = model.fit(train_set, epochs=100, callbacks=[checkpoint])

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)

import numpy as np

time = np.arange(4 * 365 + 1)

def trend(time, slope):
    return time * slope

series = trend(0.1)

series

import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()

plot_series(time, series)

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)

def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level

noise_level = 40
noisy_series = series + noise(time, noise_level)
plot_series(time, noisy_series)

