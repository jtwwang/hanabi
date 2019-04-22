# centralized nn that runs all models
# Last edited: JW 4-22

from __future__ import print_function
from experience import Experience
from tensorflow import keras
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class policy_nn():
	def __init__(self, input_dim, action_space, agent_class, model_name=None):
		self.input_dim = input_dim
		self.action_space = action_space
		self.model = self.create_model()
		self.path = os.path.join("model", agent_class)
		self.model_type = None
		if model_name != None:
			self.path = os.path.join(self.path, model_name)
			self.make_dir(self.path)
			print("Writing to", self.path)
		else:
			print(self.path, "already exists. Writing to it.")

	def make_dir(self, path):
		if not os.path.exists(path):
			try:
				os.makedirs(path)
			except OSError:
				print("Creation of the directory %s failed" % self.path)
			else:
				print("Successfully created the directory %s" % self.path)

	def create_model(self):
		# Override
		# TODO: set default dense
		return None

	def fit(self, X, y, epochs=100, batch_size=5, learning_rate=0.01):
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

		self.make_dir(self.path)
		model_path = os.path.join(self.path, "predictor.h5")

		try:
			self.model.save(model_path)
		except:
			print("Unable to write model to", model_path)

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



class dense_nn(policy_nn):
	def __init__(self):

		self.model_type = "dense"

class conv_nn(policy_nn):
	def __init__(self, input_dim, action_space, agent_class, model_name=None):
		super().__init__(input_dim, action_space, agent_class, model_name)
		self.model_type = "conv"

	def create_model(self):
		activation=None
		x = Sequential()
		x.add(Conv1D(filters=16, kernel_size=5, strides=2,
			input_shape=(self.input_dim,1), padding='same', activation=activation))
		x.add(BatchNormalization())
		x.add(Activation("relu"))
		x.add(MaxPooling1D(pool_size=3,strides=2))


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

		x.add(Flatten())
		x.add(Dense(64, activation='relu'))
		x.add(Dropout(0.2))

		x.add(Dense(self.action_space))
		print(x.summary())
		return x

class lstm_nn(policy_nn):
	def __init__(self):
		self.model_type = "lstm"

if __name__ == "__main__":
	nn = conv_nn(10000,5,"SimpoleAgent",model_name="dense1")

