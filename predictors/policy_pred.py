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
import os
import sys
import errno
import math
import pandas as pd
import IPython as ip

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


from data_pipeline.experience import Experience
from data_pipeline.balance_data import balance_data
from data_pipeline.util import one_hot, split_dataset
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import os
import sys
import errno
import math




def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)


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
    def __init__(self, agent_class, model_type=None, predictor_name='predictor'):
        """
        args:
            agent_class (string): the class of agents that we are predicting
            predictor_name (string): A directory will be created for each predictor name.
            model_type (string): the string correspondent to the type of model
        """
        self.input_dim = None
        self.action_space = None
        self.model = None

        # Set up the directory for the saved predictor model
        self.model_dir = 'model'
        if model_type is not None:
            self.model_type=model_type
            self.model_dir = os.path.join(self.model_dir, model_type)
            self.make_dir(self.model_dir)
        else:
            print("Please specify a model to use")
            sys.exit(0)
        print("Writing to", self.model_dir)
        self.predictor_name = predictor_name
        self.predictor_dir = os.path.join(self.model_dir,predictor_name)
        self.make_dir(self.predictor_dir)
        self.tensorboard_dir = os.path.join(self.predictor_dir, 'tensorboard')
        self.make_dir(self.tensorboard_dir)
        self.checkpoint_dir = os.path.join(self.predictor_dir, "checkpoints")
        self.make_dir(self.checkpoint_dir)

        # create callbacks for tensorboard
        self.tensorboard = TensorBoard(log_dir=self.tensorboard_dir)
        # create logs
        self.logpath = os.path.join(self.model_dir, 'logs.csv')
        self.logs = self.get_logs()

    def get_logs(self):

        if os.path.exists(self.logpath):
            logs = pd.read_csv(self.logpath)
        else:
            feature_names = ['pred_name', 'episodes', 'epochs', 'batch_size',
                'acc', 'loss', 'val_acc', 'val_loss']
            logs = pd.DataFrame(columns=feature_names)
        return logs

    def load_interactive(self):
        print("Below are the models for {}:".format(self.model_type))
        print(self.logs)
        pred_idx = raw_input("Select the index of the model you wish to load:")
        self.predictor_name = self.logs.loc[pred_idx]['pred_name']
        self.predictor_dir = os.path.join(self.model_dir,self.predictor_name)
        self.load()

    def load(self):
        """
        Load the saved model

        args:
            self.predictor_path (string): the name of the predictor to load
        """
        self.predictor_path = os.path.join(self.predictor_dir,'predictor.h5')
        try:
            self.model=load_model(self.predictor_path)
            print("Model loaded!")
        except IOError:
            print("Create a new model before loading.")
            sys.exit(0)



    def make_dir(self, path):
        """
        Create the directory @path if it doesn't exist already

        args:
            path (string)
        """
        try:
            os.makedirs(path)
            print("Successfully created the directory %s", path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                # print("%s already exists." % path)
                pass
        # if not os.path.exists(dir):
        #     try:
        #         os.makedirs(dir)
        #     except OSError:
        #         print("Creation of the directory %s failed" % self.model_dir)
        #     else:
        #         print("Successfully created the directory %s" % self.model_dir)

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

        self.model.compile(loss=cosine_proximity,
                           optimizer=adam, metrics=['accuracy'])

        train_sequence = HanabiSequence(self.X_train, self.y_train, batch_size)
        test_sequence = HanabiSequence(self.X_test, self.y_test, batch_size)

        history = self.model.fit_generator(
            train_sequence,
            epochs=epochs,
            verbose=2,
            validation_data=test_sequence,
            callbacks=[self.tensorboard],
            workers=0,
            shuffle=True,
            use_multiprocessing=True
        )
        log = history.history
        log['pred_name'] = self.predictor_name
        log['epochs'] = epochs
        log['batch_size'] = batch_size
        log['episodes'] = 'TODO'
        self.log = pd.DataFrame(log)


    def predict(self, X):
        """
        args:
                X (input)
        returns:
                prediction given the model and the input X
        """
        X = self.reshape_data(X)
        if self.model is None:
            self.create_model()
        pred = self.model.predict(X)
        return pred

    def save(self):
        """
        args:
            self.predictor_path (string): the name to give to the predictor
        """
        self.predictor_path = os.path.join(self.predictor_dir, 'predictor.h5')
        self.model.save(self.predictor_path)

        self.logs=self.logs.append(self.log,sort=False,ignore_index=True)
        ip.embed()


        self.logs.to_csv(self.logpath)

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
