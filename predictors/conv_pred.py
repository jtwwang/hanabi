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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout

import numpy as np


class conv_pred(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "conv"
        super(conv_pred, self).__init__(agent_class, self.model_type)

    def create_model(self):
        activation = None
        x = Sequential()
        x.add(Conv1D(filters=16, kernel_size=5, strides=2,
                     input_shape=(self.input_dim, 1), padding='same', activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=3, strides=2))

        x.add(Conv1D(filters=32, kernel_size=3, strides=2,
                     padding="same", activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2, strides=2))

        x.add(Conv1D(filters=64, kernel_size=3, strides=2,
                     padding="same", activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2, strides=2))

        x.add(Conv1D(filters=64, kernel_size=3, strides=2,
                     padding='same', activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2, strides=2))

        x.add(Flatten())
        x.add(Dense(64, activation='relu'))
        x.add(Dropout(0.2))

        x.add(Dense(self.action_space, activation='softmax'))

        self.model = x
        return x

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def extract_data(self, agent_class, val_split=0.3,
                     games=-1, balance=False):
        """
        args:
                agent_class (string)
                val_split (float)
                games (int)
        """
        obs, actions, _ = super(conv_pred, self).extract_data(agent_class,
                                                              val_split, games=games, balance=balance)

        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        self.input_dim = self.X_train.shape[1]
        self.action_space = self.y_train.shape[1]
