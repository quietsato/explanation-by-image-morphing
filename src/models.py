from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, metrics, losses, Model, Input, Sequential

from config import *


class IDCVAE(Model):
    def __init__(self, latent_dim=16, *args, **kwargs):
        super(IDCVAE, self).__init__(*args, **kwargs)

        self.E = build_encoder(latent_dim)
        self.D = build_decoder(latent_dim)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.rec_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

        self.built = True

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, x) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.E(x)

    def decode(self, z, y) -> tf.Tensor:
        y_onehot = tf.one_hot(y, num_classes)
        D_input = tf.concat([z, y_onehot], axis=1)
        return self.D(D_input)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(x)
            rec = self.decode(z, y)
            rec_loss = reconstruction_loss(x, rec)
            kl_loss = KL_loss(z_mean, z_log_var)
            total_loss = rec_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int) -> Model:
    input = Input((image_size, image_size, image_channels))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(input, [z, z_mean, z_log_var], name="E")


def build_decoder(latent_dim: int) -> Model:
    return Sequential(layers=[
        layers.InputLayer((latent_dim + num_classes, )),
        layers.Dense(7 * 7 * 64),
        layers.Reshape((7, 7, 64)),
        layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
    ], name="D")


def reconstruction_loss(x, rec, foreach=False):
    loss = tf.reduce_sum(
        losses.binary_crossentropy(x, rec), axis=(1, 2)
    )
    if foreach:
        return loss
    else:
        return tf.reduce_mean(loss)


def KL_loss(z_mean, z_log_var):
    return tf.reduce_mean(tf.reduce_sum(
        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
    ))
