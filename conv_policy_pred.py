from __future__ import print_function
from experience import Experience
import tensorflow.keras as keras
from keras.layers import Dense, ReLU, Dropout, LSTM, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.layers import Activation
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
import matplotlib.pyplot as plt

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PlotAcc(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        plt.show(block=False);

    #def on_train_end(self, logs={}):


class policy_net():

    def __init__(self, input_dim, action_space, agent_class, modelname=None):
        self.input_dim = input_dim
        self.action_space = action_space
        self.model = self.create_dense()
        self.path = os.path.join("model", agent_class)
        if modelname != None:
            self.path = os.path.join(self.path, modelname)
            print("Writing to", self.path)

    def create_dense(self):
        activation=None
        x = Sequential()
        x.add(Conv1D(filters=16, kernel_size=5, strides=2,
            input_shape=(self.input_dim,1), padding='same', activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2,strides=2))


        x.add(Conv1D(filters=32,kernel_size=3,strides=2,padding="same",activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2,strides=2))


        x.add(Conv1D(filters=64,kernel_size=3,strides=2,padding="same",activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2,strides=2))


        x.add(Conv1D(filters=64, kernel_size=3, strides=2, padding='same', activation=activation))
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2,strides=2))
        """
        x.add(Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation=activation))
        #cosine (feature space, label), dense, crossentropy+softmax. loss=Sum(cosine,crossentropy)
        x.add(BatchNormalization())
        x.add(Activation("relu"))
        x.add(MaxPooling1D(pool_size=2,strides=2))
        """

        x.add(Flatten())
        x.add(Dense(64, activation='relu'))
        x.add(Dropout(0.15))
        """

        x.add(Dense(64, activation='relu'))
        x.add(Dropout(0.2))
        x.add(Dense(32, activation='relu'))
        x.add(Dropout(0.2))
        """

        x.add(Dense(self.action_space))
        print(x.summary())
        return x

    def fit(self, X, y, epochs=100, batch_size=16, learning_rate=0.01):
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

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.path)

        # IF CONV
        print(X.shape)
        X = np.reshape(X,(X.shape[0],X.shape[1],1))

        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks = [tensorboard],
            validation_split=0.3,
            shuffle=True
            )


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
    y = replay._one_hot_moves()
    eps = replay.eps
    assert X.shape[0] == y.shape[0]

    action_distribution = np.sum(y, axis=0)
    plot_hist(action_distribution, agent_class)

    print("LOADED")
    return X, y, eps

def plot_hist(action_distribution, agent_class):
    y_pos = np.arange(len(action_distribution))
    plt.bar(y_pos, action_distribution, align="center")
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title(str(agent_class) + ' action distribution')

    plt.savefig(str(agent_class) + '_hist.png')

if __name__ == '__main__':

    flags = {'epochs': 400,
             'batch_size': 32,
             'lr': 0.001,
             'agent_class': 'SimpleAgent',
             'cv': False,
             'load': False,
             'modelname': None}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['epochs=',
                                        'batch_size=',
                                        'lr=',
                                        'agent_class=',
                                        'cv=',
                                        'load=',
                                        'modelname='])
    if arguments:
        sys.exit()
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        if flags[flag] != None:
            flags[flag] = type(flags[flag])(value)




    X, Y, _ = extract_data(flags['agent_class'])
"""
    pp = policy_net(X.shape[1], Y.shape[1], flags['agent_class'], flags['modelname'])
    if flags['load']:
        pp.load()
    pp.fit(X, Y,
           epochs=flags['epochs'],
           batch_size=flags['batch_size'],
           learning_rate=flags['lr'])
    pp.save()
"""