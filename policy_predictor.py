from __future__ import print_function
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
from experience import Experience


class policy_predictor():

    path = "predictor.h5"

    def __init__(self, input_dim, action_space):
        self.input_dim = input_dim
        self.action_space = action_space
        self.model = self.create_dense()

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

    def fit(self, X, y, X_test, Y_test, epochs=100, batch_size=16, learning_rate=0.001):
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
            batch_size=batch_size,
            validation_data=(X_test, Y_test))
        self.model.save(self.path)

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
            self.model = load_model(self.path)
        except:
            print("Create new model")


if __name__ == '__main__':

    flags = {'epochs': 400,
             'batch_size': 16,
             'lr': 0.001
             }

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['epochs=',
                                        'batch_size=',
                                        'lr='])

    if arguments:
        sys.exit()
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    print("Loading Data...", end='')
    replay = Experience(2)
    replay.load()
    X = replay._obs()
    Y = replay._one_hot_moves()

    assert X.shape[0] == Y.shape[0]
    n_entries = X.shape[0]

    # divide in training and test set
    divider = int(0.1 * n_entries)
    X_train = X[:divider]
    Y_train = Y[:divider]
    X_test = X[divider:]
    Y_test = Y[divider:]

    # randomize
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    Y_train = Y_train[p]

    print("LOADED")

    pp = policy_predictor(X.shape[1], Y.shape[1])
    pp.load()

    print("init done")
    pp.fit(X_train, Y_train, X_test, Y_test,
           epochs=flags['epochs'],
           batch_size=flags['batch_size'],
           learning_rate=flags['lr'])
