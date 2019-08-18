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

from .policy_pred import policy_pred
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from .blocks import conv_block
import numpy as np
import os


class encoder_pred(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "encoder_pred"
        super(encoder_pred, self).__init__(agent_class, self.model_type)

    def create_model(self):
        # build encoder
        inputs = Input(shape=(self.input_dim, 1))
        x = conv_block(inputs, 64, 3, 1, 3, 3)
        x = conv_block(x, 64, 3, 1, 3, 3)
        x = conv_block(x, 64, 3, 1, 3, 3)
        x = conv_block(x, 16, 3, 1, 3, 3)

        # build predictor
        x = conv_block(x, 64, 3, 1, 3, 3)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.action_space, activation='softmax')(x)

        # create the model
        self.model = Model(
            inputs=inputs,
            outputs=output,
            name="encoder_pred"
        )
        return self.model

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):

        # make sure the model is created
        if self.model is None:
            self.create_model()

        # load the encoder
        self.load_autoencoder()

        # freeze the encoder layers
        for layer in self.model.layers[:17]:
            layer.trainable = False

        # call the fit() method from policy_pred
        super(encoder_pred, self).fit(epochs, batch_size, learning_rate)

    def load_autoencoder(self):
        """
        load the encoder layers from the autoencoder.
        The model has to be in the same folder, and at the moment is moved
        manually in the correct directory.
        """

        # if you want to test other encoders, you need to changed the filename
        # from "regularized" to the new file name
        model_path = os.path.join(self.path, "regularized")

        model = load_model(model_path)
        for i in range(17):
            w = model.layers[i].get_weights()
            self.model.layers[i].set_weights(w)
