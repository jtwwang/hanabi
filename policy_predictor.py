from __future__ import print_function
import tensorflow.keras as keras
from keras.layers import Dense
from keras.layers import ReLU
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.models import load_model
import numpy as np
from experience import Experience

class policy_predictor():
    def __init__(self, input_dim, action_space):
        self.input_dim = input_dim
        self.action_space = action_space
        self.model = self.create_dense()

    def create_dense(self):
        x = Sequential()
        x.add(Dense(32, input_dim=self.input_dim))
        x.add(Dense(24, activation='relu'))
        x.add(Dense(16, activation='relu'))
        x.add(Dense(self.action_space, activation='softmax'))
        return x

    def fit(self, X, y, X_test, Y_test, epochs=100, batch_size=1):
        """
        args:
                X (int arr): vectorized features
                y (int arr): one-hot encoding with dimensions(sample_size,action_space)
        """
        adam = optimizers.Adam(
                lr=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam, metrics=['accuracy'])
        print()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

    def predict(self, X):
        pred = self.model.predict(X)
        return pred


if __name__ == '__main__':

    path = "model.h5"

    print("Loading Data...", end='')
    replay = Experience(2)
    replay.load()
    X = replay._obs()
    Y = replay._one_hot_moves()

    # randomize
    assert X.shape[0] == Y.shape[0]
    n_entries = X.shape[0]
    p = np.random.permutation(n_entries)
    X = X[p]
    Y = Y[p]

    # divide in training and test set
    divider = int(0.7 * X.shape[0])
    X_train = X[:divider]
    Y_train = Y[:divider]
    X_test = X[divider:]
    Y_test = Y[divider:]

    print("LOADED")

    pp = policy_predictor(X.shape[1], Y.shape[1])

    try:
        pp.model = load_model(path)
    except:
        print("create new model")
   
    print("init done")
    pp.fit(X_train, Y_train, X_test, Y_test, epochs = 400)
    pp.model.save(path)

