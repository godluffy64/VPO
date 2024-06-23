import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape


def Conv(n_filters, filter_width):
    return Conv2D(n_filters, filter_width, 
                  strides=2, padding="same", activation="relu")

def Deconv(n_filters, filter_width):
    return Conv2DTranspose(n_filters, filter_width, 
                           strides=2, padding="same", activation="linear") # attention ici, mettre 'linear'

def Encoder(inputs):
    X = Conv(32, 5)(inputs)
    X = Conv(64, 5)(X)
    X = Conv(128, 3)(X)
    X = Conv(256, 3)(X)
    X = Flatten()(X)
    Z = Dense(1024, activation="relu", name="encoder_output")(X)
    return Z

"""def Decoder(Z):
    X = Dense(32,  activation="relu", name="decoder_input")(Z)
    X = Dense(64,  activation="relu")(X)
    X = Dense(128, activation="relu")(X)
    X = Dense(64,  activation="relu")(X)
    X = Reshape((2, 2, 16))(X)      # X.shape --> TensorShape([None, 2, 2, 16])
    X = Deconv(16, 3)(X)            # X.shape --> TensorShape([None, 4, 4, 16])
    X = Deconv(16, 3)(X)            # X.shape --> TensorShape([None, 8, 8, 16])
    X = Deconv(16, 5)(X)            # X.shape --> TensorShape([None, 16, 16, 16])
    
    X = Deconv(16, 5)(X)            # X.shape --> TensorShape([None, 32, 32, 16])
    X = Deconv(16, 5)(X)            # X.shape --> TensorShape([None, 64, 64, 16])
    X = Deconv(16, 5)(X)            # X.shape --> TensorShape([None, 128, 128, 16])
    
    X = Conv2D(3, 1)(X)             ## X.shape --> TensorShape([None, 128, 128, 3])
    return X """


def Decoder(Z):
    X = Reshape((2, 2, 256), name="decoder_input")(Z)
    X = Deconv(128, 3)(X)
    X = Deconv(64, 5)(X)
    X = Deconv(32, 5)(X)
    X = Deconv(3, 5)(X)
    X = Conv2D(3, 1)(X) # vous pouvez ici adjoindre 'activation="sigmoid"'a
    return X 

def AutoEncoder():
    X = tf.keras.Input(shape=(8, 8, 3))
    Z = Encoder(X)
    X_pred = Decoder(Z)
    return tf.keras.Model(inputs=X, outputs=X_pred)

