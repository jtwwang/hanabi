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
from tensorflow.keras.layers import Dense

import numpy as np


class dense_pred(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "dense"
        super(dense_pred, self).__init__(agent_class, self.model_type)

    def create_model(self):
        x = Sequential()
        x.add(Dense(32, input_shape=(self.input_dim,)))
        x.add(Dense(64))
        x.add(Dense(128))
        x.add(Dense(128))
        x.add(Dense(self.action_space))
        self.model = x
        return x

    def extract_data(self, agent_class, val_split=0.3, games=-1, balance=False):
        """
        args:
                agent_class (string)
                num_player (int)
        """
        super(dense_pred, self).extract_data(agent_class,
                                             val_split,
                                             games=games,
                                             balance=balance)

        self.input_dim = self.X_train.shape[1]
        self.action_space = self.y_train.shape[1]

    def reshape_data(self, X):
        if X.shape == (self.action_space,):  # Only one sample inputted
            X = np.reshape(X, (1, X))
        return X
