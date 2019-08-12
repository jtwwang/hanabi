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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization, concatenate
from tensorflow.keras.layers import Activation, Dropout, Input

from tensorflow.keras.utils import plot_model

from .blocks import conv_block

import numpy as np


class conv_pred(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "conv"
        super(conv_pred, self).__init__(agent_class, self.model_type)


    def create_model_old(self):
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

    def create_model_conv_block(self):
        inputs = Input(shape=(self.input_dim,1))
        x = conv_block(inputs, 16, 5, 2, 3, 2)
        x = conv_block(x, 32, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.action_space, activation='softmax')(x)

        # create the model
        self.model = Model(
            inputs=inputs,
            outputs=output,
            name="big_conv"
        )

        return x

    def create_model(self, img_path='network_image.png'):
        input_shape = (self.input_dim,1)
        inputs = Input(shape=input_shape)
        activation=None
        # TOWER 1
        tower_1 = Conv1D(filters=16, kernel_size=7, strides=2,
            padding="same", activation=activation)(inputs)
        tower_1 = MaxPooling1D(pool_size=3, strides=2) (tower_1)
        tower_1 = BatchNormalization()(tower_1)
        tower_1 = Activation("relu")(tower_1)

        tower_1 = Conv1D(filters=32, kernel_size=3, strides=2,
            padding="same", activation=activation)(inputs)
        tower_1 = MaxPooling1D(pool_size=3, strides=2) (tower_1)
        tower_1 = BatchNormalization()(tower_1)
        tower_1 = Activation("relu")(tower_1)

        # TOWER 2
        tower_2 = Conv1D(filters=16, kernel_size=5, strides=2,
            padding="same", activation=activation)(inputs)
        tower_2 = MaxPooling1D(pool_size=3, strides=2)(tower_2)
        tower_2 = BatchNormalization()(tower_2)
        tower_2 = Activation("relu")(tower_2)

        tower_2 = Conv1D(filters=32, kernel_size=3, strides=2,
            padding="same", activation=activation)(inputs)
        tower_2 = MaxPooling1D(pool_size=3, strides=2) (tower_2)
        tower_2 = BatchNormalization()(tower_2)
        tower_2 = Activation("relu")(tower_2)

        # TOWER 3
        tower_3 = Conv1D(filters=16, kernel_size=3, strides=2,
            padding="same", activation=activation)(inputs)
        tower_3 = MaxPooling1D(pool_size=3, strides=2)(tower_3)
        tower_3 = BatchNormalization()(tower_3)
        tower_3 = Activation("relu")(tower_3)

        tower_3 = Conv1D(filters=32, kernel_size=3, strides=2,
            padding="same", activation=activation)(inputs)
        tower_3 = MaxPooling1D(pool_size=3, strides=2) (tower_3)
        tower_3 = BatchNormalization()(tower_3)
        tower_3 = Activation("relu")(tower_3)

        merged = concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = Flatten()(merged)

        out = Dense(128, activation='relu')(merged)
        out = Dense(self.action_space, activation='softmax')(out)

        model = Model(inputs, out)
        plot_model(model, to_file=img_path)
        self.model = model
        return model


    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X
