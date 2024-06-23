import os
import sys
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data_manager import DataManager

import matplotlib.pyplot as plt 

from model import Encoder, Decoder

from absl import app
from absl import flags

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
flags.DEFINE_boolean("use_noise", False, "sample noisey images")
FLAGS = flags.FLAGS


def sample(model, n_samples):
    """Passes n random samples through the model and displays X & X_pred"""
    manager = DataManager()
    _, X = manager.get_batch(n_samples, use_noise=FLAGS.use_noise)
    X_pred = model.predict(X)
    """ X_stitched = np.reshape(X.swapaxes(0,1), (x_dim, y_dim*n_samples))
    X_pred_stitched = np.reshape(X_pred.swapaxes(0,1), (x_dim, y_dim*n_samples))
    stitched_img = np.vstack((X_stitched, X_pred_stitched))"""
    fig, axes = plt.subplots(nrows=4, ncols=n_samples, figsize=(12, 12))
    
    for i in range(n_samples):
        # Première rangée : image d'origine 64x64
        axes[0, i].imshow(X[i], cmap='gray')
        axes[0, i].axis('off')
        
        # Deuxième rangée : image redimensionnée naïvement en 128x128
        naive_resized = tf.image.resize(X[i], size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        axes[1, i].imshow(tf.squeeze(naive_resized), cmap='gray')
        axes[1, i].axis('off')

        # Troisième rangée : image redimensionnée en bicubique en 128x128
        bicubic_resized = tf.image.resize(X[i], size=(128, 128), method=tf.image.ResizeMethod.BICUBIC)
        axes[2, i].imshow(tf.squeeze(bicubic_resized), cmap='gray')
        axes[2, i].axis('off')

        # quatrième rangée : image redimensionnée avec ce projet de deep learning
        axes[3, i].imshow(X_pred[i], cmap='gray')  # Supposant que X_pred est déjà en 128x128
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(FLAGS.model)
    model = tf.keras.models.load_model(model_path)

    # holds dimensions of latent vector once we find it
    z_dim = None

    # define encoder
    encoder_in  = tf.keras.Input(shape=(3, 3, 3))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # load encoder weights and get the dimensions of the latent vector
    for i, layer in enumerate(model.layers):
        encoder.layers[i] = layer
        if layer.name == "encoder_output":
            z_dim = (layer.get_weights()[0].shape[-1])
            break

    # define encoder
    decoder_in  = tf.keras.Input(shape=(z_dim,))
    decoder_out = Decoder(decoder_in)
    decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # load decoder weights
    found_decoder_weights = False
    decoder_layer_cnt = 0
    for i, layer in enumerate(model.layers):
        print(layer.name)
        weights = layer.get_weights()
        if len(layer.get_weights()) > 0:
            print(weights[0].shape, weights[1].shape)
        if "decoder_input" == layer.name:
            found_decoder_weights = True
        if found_decoder_weights:
            decoder_layer_cnt += 1
            print("dec:" + decoder.layers[decoder_layer_cnt].name)
            decoder.layers[decoder_layer_cnt].set_weights(weights)

    encoder.summary()
    decoder.summary()

    return encoder, decoder, model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    encoder, decoder, autoencoder = load_model()
    sample(autoencoder, FLAGS.sample_size)

if __name__ == '__main__':
    app.run(main)
