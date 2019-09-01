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
from tensorflow.keras.metrics import CategoricalAccuracy

from .blocks import conv_block

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class treenet(policy_pred):
    def __init__(self, agent_class, predictor_name):
        self.model_type = "treenet"
        super(treenet, self).__init__(agent_class, self.model_type)

    def policy_head(self, inputs, output_dim, name_head):
        # fully connected layer
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

        head_1 = self.policy_head(x, self.action_space, 'ph1')
        head_2 = self.policy_head(x, self.action_space, 'ph2')
        head_3 = self.policy_head(x, self.action_space, 'ph3')

        self.model = Model(inputs=inputs,
                           outputs=[head_1, head_2, head_3],
                           name="multihead")
        return self.model

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

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

        losses = {'ph1': "categorical_crossentropy",
                  'ph2': "categorical_crossentropy",
                  'ph3': "categorical_crossentropy"}
        loss_weights = {'ph1': 1.0, 'ph2': 1.0, "ph3": 1.0}

        self.model.compile(loss=losses,
                           loss_weights=loss_weights,
                           optimizer=adam,
                           metrics=['accuracy'])

        self.model.fit(
            self.X_train,
            {'ph1': self.y_train,
             'ph2': self.y_train,
             'ph3': self.y_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=[self.X_test,
                             {'ph1': self.y_test,
                              'ph2': self.y_test,
                              'ph3': self.y_test}],
            callbacks=[self.tensorboard],
            shuffle=True,
            verbose=2
        )

        acc = CategoricalAccuracy()
        avg_pred = np.zeros(self.y_test.shape)
        (p1, p2, p3) = self.model.predict(self.X_test,
                                          batch_size=batch_size,
                                          verbose=0)
        avg_pred = (p1 + p2 + p3) / 3
        acc.update_state(self.y_test, avg_pred)
        print('Final result: %f' % acc.result().numpy())
