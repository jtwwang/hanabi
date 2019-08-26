# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from __future__ import division
from .policy_pred import policy_pred
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Cropping1D, ActivityRegularization
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from .blocks import conv_block, deconv_block
import numpy as np


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class autoencoder(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "autoencoder"
        super(autoencoder, self).__init__(agent_class, self.model_type)

    def create_model(self):
        """
        This function builds a tf.keras.Model instance and initialize the
        variable self.model with such instance.

        The model is a deep convolutional autoencoder with a bottleneck of size
        144. After 140 epochs with 5000 games of data it achieves
        ~97.3% accuracy
        """

        # build encoder
        inputs = Input(shape=(self.input_dim, 1), name='encoder_input')
        x = conv_block(inputs, 64, 3, 1, 3, 3)
        x = conv_block(x, 64, 3, 1, 3, 3)
        x = conv_block(x, 64, 3, 1, 3, 3)
        x = conv_block(x, 16, 3, 1, 3, 3)
        encoded = ActivityRegularization(10e-4, 0.0)(x)

        # build decoder
        x = deconv_block(encoded, 16, 3, 1, 3)
        x = Cropping1D((0, 2))(x)
        x = deconv_block(x, 64, 3, 1, 3)
        x = Cropping1D((0, 1))(x)
        x = deconv_block(x, 64, 3, 1, 3)
        x = Cropping1D((0, 2))(x)
        x = deconv_block(x, 64, 3, 1, 3)
        x = deconv_block(x, 1, 3, 1, 1)
        decoder = Cropping1D((0, 2), name='decoder_output')(x)

        self.model = Model(inputs,
                           outputs=decoder,
                           name='vae_predictor')

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):

        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)

        # reshape the data
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        if self.model is None:
            self.create_model()

        self.model.compile(loss='mean_squared_error',
                           optimizer=adam,
                           metrics=['accuracy'])

        self.model.fit(
            self.X_train,
            self.X_train,
            epochs=epochs,
            verbose=2,
            validation_data=[self.X_test, self.X_test],
            validation_freq=5,
            callbacks=[self.tensorboard],
            workers=0,
            shuffle=True)
