from .policy_pred import policy_pred

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization
from keras.layers import Activation, ReLU, Dropout

import numpy as np
import IPython as ip


class conv_pred(policy_pred):
	def __init__(self, agent_class, model_name=None):
		super().__init__(agent_class, model_name)
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

		self.model = x
		return x

	def reshape_data(self, X_raw):
		#ip.embed()
		if X_raw.shape == (self.action_space,): # If only one sample is put in
			X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))
		X = np.reshape(X_raw,(X_raw.shape[0],X_raw.shape[1],1)) # Add an additional dimension for filters
		return X

	def extract_data(self, agent_class):
		"""
		args:
			agent_class (string)
			num_player (int)
		"""
		obs, actions, eps = super().extract_data(agent_class)
		X = self.reshape_data(obs)
		y = actions
	
		self.X = X
		self.y = y
		self.input_dim = X.shape[1]
		self.action_space = y.shape[1]

		return X, y, eps