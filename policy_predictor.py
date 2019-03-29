from __future__ import print_function
from experience import Experience
import tensorflow.keras as keras
from keras.layers import Dense, ReLU, Dropout
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.models import load_model
import numpy as np
import getopt
import sys
import os
import math
# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class policy_net():

    def __init__(self, input_dim, action_space, agent_class):
        self.input_dim = input_dim
        self.action_space = action_space
        self.model = self.create_dense()
        self.path = os.path.join("model", agent_class)

    def create_dense(self):
        x = Sequential()
        x.add(Dense(128, input_dim=self.input_dim))
        x.add(Dropout(0.1))
        x.add(Dense(64, activation='relu'))
        x.add(Dropout(0.1))
        x.add(Dense(32, activation='relu'))
        x.add(Dropout(0.1))
        x.add(Dense(self.action_space))
        return x

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.001):
        """
        args:
                X (int arr): vectorized features
                y (int arr): one-hot encoding with dimensions(sample_size,action_space)
        """
        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)
        self.model.compile(loss='cosine_proximity',
                           optimizer=adam, metrics=['accuracy'])
        print()
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size)
        self.save()

    def save(self):

        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print("Creation of the directory %s failed" % self.path)
            else:
                print("Successfully created the directory %s" % self.path)

        try:
            self.model.save(os.path.join(self.path, "predictor.h5"))
        except:
            print("something wrong")

    def predict(self, X):
        """
        args:
            X (input)
        return
            prediction given the model and the input X
        """
        pred = self.model.predict(X)
        return pred

    def load(self):
        """
        function to load the saved model
        """
        try:
            self.model = load_model(os.path.join(self.path, "predictor.h5"))
        except:
            print("Create new model")


def extract_data(agent_class):
    """
    args:
        agent_class (string)
        num_player (int)
    """
    print("Loading Data...", end='')
    replay = Experience(agent_class, load=True)
    replay.load()
    X = replay._obs()
    Y = replay._one_hot_moves()
    eps = replay.eps
    assert X.shape[0] == Y.shape[0]

    print("LOADED")
    return X, Y, eps


def cross_validation(k, max_ep):
    global flags
    mean = 0

    X,Y,eps = extract_data(flags['agent_class'])
    max_ep = min(max_ep, math.floor(len(eps)/k))

    for i in range(k):
        # split the data
        train_id = range(eps[i * max_ep][0], eps[(i + 1)*max_ep - 1][1])
        test_id = range(0,eps[i * max_ep][0]) + range(eps[(i + 1)*max_ep][0], eps[-1][1])
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = Y[train_id], Y[test_id]

        # initialize the predictor (again)
        pp = policy_net(X.shape[1], Y.shape[1], flags['agent_class'])

        pp.fit(X_train, y_train,
               epochs=flags['epochs'],
               batch_size=flags['batch_size'],
               learning_rate=flags['lr'])

        # calculate accuracy and add it to the mean
        score = pp.model.evaluate(X_test, y_test, verbose=0)
        mean += score[1]

    # calculate the mean
    mean = mean/k
    return mean


if __name__ == '__main__':

    flags = {'epochs': 400,
             'batch_size': 32,
             'lr': 0.001,
             'agent_class': 'SimpleAgent',
             'cv': False}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['epochs=',
                                        'batch_size=',
                                        'lr=',
                                        'agent_class=',
                                        'cv='])
    if arguments:
        sys.exit()
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)
   
    if (flags['cv']):
        # do cross validation
        k = 5
        max_episodes = 100
        mean = cross_validation(k, max_episodes)
        print('Average: ', end='')
        print(mean)
    else:
        # data
        X, Y, _ = extract_data(flags['agent_class'])

        pp = policy_net(X.shape[1], Y.shape[1], flags['agent_class'])
        pp.load()
        pp.fit(X, Y,
               epochs=flags['epochs'],
               batch_size=flags['batch_size'],
               learning_rate=flags['lr'])
