# centralized nn that runs all models
# Last edited: JW 4-22

from __future__ import print_function
from experience import Experience
from tensorflow import keras
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.layers import Activation, ReLU, Dropout
from keras.layers import LSTM, Embedding, TimeDistributed
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

# Debugging
import IPython as ip

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EvalAcc(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		print("acc: {} - val_acc: {} - loss: {} - eval_loss: {}".format(
			logs.get('acc'), logs.get('val_acc'), logs.get('loss'), logs.get('val_loss')))

class policy_pred():
	def __init__(self, agent_class, model_name=None):
		self.X = None	# nn input
		self.y = None	# nn output
		self.input_dim = None
		self.action_space = None

		self.model = None

		# Create the directory for this particular model
		self.path = os.path.join("model", agent_class)
		self.model_type = None
		if model_name != None:
			self.path = os.path.join(self.path, model_name)
			self.make_dir(self.path)
			print("Writing to", self.path)
		else:
			print(self.path, "already exists. Writing to it.")
		self.checkpoint_path = os.path.join(self.path, "checkpoints")

	def make_dir(self, path):
		# Create the directory @path if it doesn't exist already
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

	def create_validation_split(self, val_split):
		test_idx = np.random.choice(self.X.shape[0], int(self.X.shape[0]*val_split), replace=False)
		train_idx = np.setdiff1d(range(self.X.shape[0]), test_idx)
		X_train, X_test = self.X[train_idx], self.X[test_idx]
		y_train, y_test = self.y[train_idx], self.y[test_idx]
		return X_train, X_test, y_train, y_test

	def fit(self, epochs=100, batch_size=5, learning_rate=0.01, val_split=None):
		"""
		args:
			val_split(float, between 0 and 1): Fraction of the data to be used as validation data
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

		if val_split == None:
			self.model.fit(
				self.X,
				self.y,
				epochs=epochs,
				batch_size=batch_size,
				callbacks = [tensorboard],
				validation_split=0.3, # Uses 30% of training data as validation
				shuffle=True
				)
		else:
			X_train, X_test, y_train, y_test = self.create_validation_split(val_split)
			self.model.fit(
				X_train,
				y_train,
				epochs=epochs,
				batch_size=batch_size,
				callbacks = [tensorboard],
				validation_data = (X_test, y_test),
				shuffle=True)
			#ip.embed()
		# DEBUGGING
		#self.predict(self.X[0])



	def predict(self, X):
		"""
		args:
			X (input)
		returns:
			prediction given the model and the input X
		"""
		# If just reading in one sample, add a "sample" dimension
		
		X = self.reshape_data(X)
		#ip.embed()
		pred = self.model.predict(X)
		return pred

	def save(self):

		self.make_dir(self.path)
		model_path = os.path.join(self.path, "predictor.h5")

		try:
			self.model.save(model_path)
		except:
			print("Unable to write model to", model_path)

	def load(self):
		"""
		function to load the saved model
		"""
		try:
			self.model = load_model(os.path.join(self.path, "predictor.h5"))
		except:
			print("Create new model")


	def extract_data(self, agent_class):
		"""
		args:
			agent_class (string): "SimpleAgent", "RainbowAgent"
			num_player (int)
		"""
		print("Loading Data...", end='')
		replay = Experience(agent_class, load=True)
		replay.load()
		obs = replay._obs()
		actions = replay._one_hot_moves()
		eps = replay.eps
		assert obs.shape[0] == actions.shape[0]

		print("Experience Loaded!")
		return obs, actions, eps

   

