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
from data_pipeline.balance_data import balance_data
from data_pipeline.util import one_hot, split_dataset
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
import os
import math

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HanabiSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


class policy_pred(object):
    def __init__(self, agent_class, model_type=None):
        """
        args:
            agent_class (string): the class of agents that we are predicting
            model_type (string): the string correspondent to the type of model
        """
        self.input_dim = None
        self.action_space = None
        self.model = None

        # Create the directory for this particular model
        self.path = os.path.join("model", agent_class)
        if model_type is not None:
            self.path = os.path.join(self.path, model_type)
            self.make_dir(self.path)
            print("Writing to", self.path)
        else:
            print(self.path, "already exists. Writing to it.")
        self.checkpoint_path = os.path.join(self.path, "checkpoints")

        # create callbacks for tensorboard
        self.tensorboard = TensorBoard(log_dir=self.path)

    def make_dir(self, path):
        """
        Create the directory @path if it doesn't exist already

        args:
            path (string)
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % self.path)
            else:
                print("Successfully created the directory %s" % self.path)

    def reshape_data(self, X):
        pass

    def create_model(self):
        # Override
        pass

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
            self.create_model()  # create the model if not already loaded

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam, metrics=['accuracy'])

        train_sequence = HanabiSequence(self.X_train, self.y_train, batch_size)
        test_sequence = HanabiSequence(self.X_test, self.y_test, batch_size)

        self.model.fit_generator(
            train_sequence,
            epochs=epochs,
            verbose=2,
            validation_data=test_sequence,
            validation_freq=5,
            callbacks=[self.tensorboard],
            workers=0,
            shuffle=True
        )

    def predict(self, X):
        """
        args:
                X (input)
        returns:
                prediction given the model and the input X
        """
        X = self.reshape_data(X)
        if self.model == None:
            self.create_model()
        pred = self.model.predict(X)
        return pred

    def save(self, model_name="predictor.h5"):
        """
        args:
            model_name (string): the name to give to the predictor
        """
        self.make_dir(self.path)
        model_path = os.path.join(self.path, model_name)
        self.model.save(model_path)

    def load(self, model_name="predictor.h5"):
        """
        Function to load the saved model

        args:
            model_name (string): the name of the predictor to load
        """
        model_path = os.path.join(self.path, model_name)

        try:
            self.model = load_model(model_path)
        except IOError:
            print("Create new model.")

    def extract_data(self, agent_class, val_split=0.3, games=-1, balance=False):
        """
        args:
                agent_class (string): "SimpleAgent", "RainbowAgent"
                val_split (float): the split between training and test set
                games (int): how many games we want to load
                balance (bool)
        """
        print("Loading Data...", end='')
        replay = Experience(agent_class, load=True)
        moves, _, obs, eps = replay.load(games=games)

        # split dataset here
        training_set, test_set = split_dataset([obs, moves], val_split)
        X_train, X_test = training_set[0], test_set[0]
        y_train, y_test = training_set[1], test_set[1]

        if balance:
            # make class balanced
            X_train, y_train = balance_data(X_train, y_train)

        # convert to one-hot encoded tensor
        self.y_train = one_hot(y_train, replay.n_moves)
        self.y_test = one_hot(y_test, replay.n_moves)

        self.X_train = X_train
        self.X_test = X_test

        # define model parameters
        self.define_model_dim(self.X_train.shape[1], self.y_train.shape[1])

        print("DONE")
        return X_train, self.y_train, X_test, self.y_test, eps

    def define_model_dim(self, input_dim, action_space):
        """
        args:
            input_dim (int): the size of the input
            action_space (int): the size of the output
        """
        self.input_dim = input_dim
        self.action_space = action_space
