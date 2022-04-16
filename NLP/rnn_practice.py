# -*- coding: utf-8 -*-
"""RNN-Practice.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yZKUdHG2890IkMeIquU3wxS8aRkpvM8R
"""

import tensorflow_datasets as tfds

import tensorflow as tf

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

list(train_dataset.take(1))

tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)

len(train_dataset)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

list(train_dataset.take(2))

"""# Simple RNN"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, RNN, Dense, LSTM, Bidirectional, SimpleRNN

tokenizer.vocab_size

model1 = Sequential()
model1.add(Embedding(tokenizer.vocab_size, 64))
model1.add(SimpleRNN(32))

model1.summary()

model1.add(Dense(16, activation='relu'))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model1.fit(train_dataset, epochs=10, validation_data=test_dataset)

"""# LSTM"""

model2 = Sequential()
model2.add(Embedding(tokenizer.vocab_size, 64))
model2.add(LSTM(32))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.summary()

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model2.fit(train_dataset, epochs=10, validation_data=test_dataset)

"""# Bidirectional"""

model3 = Sequential()
model3.add(Embedding(tokenizer.vocab_size, 64))
model3.add(Bidirectional(LSTM(32)))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.summary()

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model3.fit(train_dataset, epochs=10, validation_data=test_dataset)

"""# Deep Bidirectional LSTM"""

model4 = Sequential()
model4.add(Embedding(tokenizer.vocab_size, 64))
model4.add(Bidirectional(LSTM(64, return_sequences=True))) # (64 * 64 * 8 + 64 * 4) * 2
model4.add(Bidirectional(LSTM(64, return_sequences=True))) # (64 * 64 * 8 + 64 * 4) * 2
model4.add(Bidirectional(LSTM(32))) # (128 * 32 * 4 + 32 * 32 * 4 + 32 * 4) * 2
model4.add(Dense(32, activation='relu'))
model4.add(Dense(1, activation='sigmoid'))

model4.summary()

model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model4.fit(train_dataset, epochs=10, validation_data=test_dataset)
