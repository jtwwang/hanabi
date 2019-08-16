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

from __future__ import print_function
from data_pipeline.experience import Experience
from data_pipeline.util import one_hot, split_dataset
from .policy_pred import policy_pred

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras import optimizers

from .blocks import conv_block

import numpy as np


class multihead(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "multihead"
        super(multihead, self).__init__(agent_class, self.model_type)

    @staticmethod
    def policy_head(inputs, action_space):
        # fully connected layer
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)

        # fully connected layer with softmax
        x = Dense(action_space,
                  activation='softmax',
                  name='policy_output')(x)
        return x

    @staticmethod
    def value_head(inputs):
        # fully connected layer
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)

        # fully connected layer with softmax
        x = Dense(1, name='value_output')(x)
        return x

    def create_model(self):
        """
        Function to create the model
        """
        inputs = Input(shape=(self.input_dim,1))
        x = conv_block(inputs, 16, 5, 2, 3, 2)
        x = conv_block(x, 32, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = Flatten()(x)

        policy_head = multihead.policy_head(x, self.action_space)
        value_head = multihead.value_head(x)

        self.model = Model(inputs = inputs,
                           outputs=[policy_head, value_head],
                           name="multihead")
        return self.model

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def fit(self, epochs= 100, batch_size=64, learning_rate=0.01):
        """
        args:
            epochs (int): the number of epochs
            batch_size (int): the size of the batch in training and testing
            learning_rate (float): the relative size of the step taken in
                                   the steepest descent direction
        """
        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)

        if self.model == None:
            self.create_model()

        losses={'policy_output': "categorical_crossentropy",
                'value_output': "mean_squared_error"}
        loss_weights={'policy_output': 1.0,
                      'value_output': 1.0}

        self.model.compile(loss=losses,
                           loss_weights=loss_weights,
                           optimizer=adam,
                           metrics=['accuracy'])

        self.model.fit(
            self.X_train,
            {'policy_output': self.moves_train,
            'value_output': self.rewards_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=[self.X_test,
            {'policy_output': self.moves_test,
            'value_output': self.rewards_test}],
            callbacks=[self.tensorboard],
            shuffle=True,
            verbose = 2
        )


    def extract_data(self, agent_class, val_split=0.3,
                     games=-1, balance=False):
        """
        args:
                agent_class (string)
                val_split (float): default 0.3
                games (int): default it loads all
                balance: this class does not support the balance function
        """
        print("Loading Data...", end='')
        replay = Experience(agent_class, load=True)
        moves, rewards, obs, eps = replay.load(games=games)

        # split dataset here
        data_list = [moves, rewards, obs]
        training_list, test_list = split_dataset(data_list, val_split)
        self.moves_train, self.moves_test = training_list[0], test_list[0]
        self.rewards_train, self.rewards_test = training_list[1], test_list[1]
        self.X_train, self.X_test = training_list[2], test_list[2]

        # convert to one-hot encoded tensor
        self.moves_train = one_hot(self.moves_train, replay.n_moves)
        self.moves_test = one_hot(self.moves_test, replay.n_moves)

        # reshape data for the network input layer
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        self.input_dim = self.X_train.shape[1]
        self.action_space = self.moves_train.shape[1]
        print("DONE")
