from .policy_pred import policy_pred

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

class dense_pred(policy_pred):
	def __init__(self, agent_class, model_name=None):
		super(dense_pred, self).__init__(agent_class, model_name)
		self.model_type = "dense"

	def create_model(self):
		activation=None
		x = Sequential()
		x.add(Dense(32, input_shape=(self.input_dim,)))
		x.add(Dense(64))
		x.add(Dense(128))
		x.add(Dense(128))
		x.add(Dense(self.action_space))
		self.model = x
		return x

	def extract_data(self, agent_class, games = -1):
		"""
		args:
			agent_class (string)
			num_player (int)
		"""
		obs, actions, eps = super(dense_pred, self).extract_data(agent_class, games = games)
		X = obs
		y = actions

		self.X = X
		self.y = y
		self.input_dim = X.shape[1]
		self.action_space = y.shape[1]

		return X, y, eps

	def reshape_data(self, X):
		if X.shape == (self.action_space,): # Only one sample inputted
			X = np.reshape(X,(1,X))
		return X
