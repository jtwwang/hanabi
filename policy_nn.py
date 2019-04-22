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
	def __init__(self, input_dim, action_space, agent_class, model_type="dense", model_name=None):
		self.input_dim = input_dim
		self.action_space = action_space
		self.model = self.create_dense()
		self.path = os.path.join("model", agent_class)
		self.model_type = model_type
		if model_name != None:
			self.path = os.path.join(self.path, model_name)
			print("Writing to", self.path)

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




class conv(policy_nn):
	pass

class dense(policy_nn):
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

class lstm(policy_nn):
	pass

