import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


def ffn(d_ff, d_model):
    # TODO: Update document
    return Sequential(
        [
            Dense(units=d_ff, activation = 'relu'),
            Dense(d_model)
        ]
    )