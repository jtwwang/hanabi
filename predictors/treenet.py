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
from .policy_pred import policy_pred

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras import optimizers

from data_pipeline.util import accuracy

from .blocks import conv_block

import numpy as np


class treenet(policy_pred):
    def __init__(self, agent_class, predictor_name, model_type=None):
        if model_type is None:
            model_type = "treenet"
        super(treenet, self).__init__(agent_class=agent_class,
                                      model_type=model_type,
                                      predictor_name=predictor_name)

        self.n_heads = 3 # more heads don't seem necessary

    def policy_head(self, inputs, output_dim, name_head):
        """
        Architecture for each of the heads

        args:
            inputs: a tensor
            output_dim: the number of classes we are predicting
            name_head: the name to give to the output layer
        """
        x = conv_block(inputs, 64, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(output_dim,
                  activation='softmax',
                  name=name_head)(x)
        return x

    def create_model(self):
        """
        Function to create the model
        """
        inputs = Input(shape=(self.input_dim, 1))
        x = conv_block(inputs, 64, 5, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)

        heads = []
        for i in range(self.n_heads):
            name_head = 'ph' + str(i)
            heads.append(self.policy_head(x, self.action_space, name_head))

        self.model = Model(inputs=inputs,
                           outputs=heads,
                           name="treenet")
        return self.model

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def predict(self, X, batch_size=64):
        predictions = self.model.predict(X,
                                         batch_size=batch_size,
                                         verbose=0)
        return sum(predictions) / self.n_heads

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):
        """
        args:
            epochs (int): the number of epochs
            batch_size (int): the size of the batch in training and testing
            learning_rate (float): the relative size of the step taken in
                                   the steepest descent direction
        """

        # reshape the data
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)

        if self.model is None:
            self.create_model()

        losses = {}
        loss_weights = {}
        labels = {}
        validation_labels = {}
        for i in range(self.n_heads):
            name_head = 'ph' + str(i)
            losses[name_head] = "categorical_crossentropy"
            loss_weights[name_head] = 1.0
            labels[name_head] = self.y_train
            validation_labels[name_head] = self.y_test

        self.model.compile(loss=losses,
                           loss_weights=loss_weights,
                           optimizer=adam,
                           metrics=['accuracy'])

        self.model.fit(
            self.X_train,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=[self.X_test, validation_labels],
            callbacks=[self.tensorboard],
            shuffle=True,
            verbose=2
        )

        pred = self.predict(self.X_test, batch_size)
        acc = accuracy(pred, self.y_test)
        print('Final result: %f' % acc)
