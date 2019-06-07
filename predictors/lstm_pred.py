from .policy_pred import policy_pred

from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Embedding, TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import os
from os import path

class lstm_pred(policy_pred):
	def __init__(self, agent_class):
            self.model_type = "lstm"
	    super(lstm_pred, self).__init__(agent_class, self.model_type)

	def create_model(self):
		activation=None
		x = Sequential()
		#x.add(TimeDistributed(Conv1D(filters=16, kernel_size=5, strides=2,
		#	input_shape=(self.input_dim,1), padding='same', activation=activation)))
		x.add(LSTM(64, input_shape=(None,self.input_dim), return_sequences=True))
		x.add(LSTM(64, return_sequences=True))
		x.add(LSTM(128, return_sequences=True))
		x.add(TimeDistributed(Dense(self.action_space)))
		self.model = x
		return x

	def reshape_data(self, X_raw):

		if X_raw.shape == (self.action_space,):
			X_raw = np.reshape(X_raw,(1,X_raw.shape[0]))

		X = np.reshape(X_raw, (1,X_raw.shape[0], X_raw.shape[1]))
		return X

	def seperate_games(self, obs, actions, eps):
		X, y = [], []
		for ep in eps:
			X.append(obs[range(ep[0],ep[1])])
			y.append(actions[range(ep[0],ep[1])])
		return (X,y)

	

	def extract_data(self, agent_class, games = -1):
		"""
		args:
			agent_class (string)
			num_player (int)
		"""
		obs, actions, eps = super(lstm_pred,self).extract_data(agent_class, games = games)
		# Dimensions: (episodes, moves_per_game, action_space)
		X, y = self.seperate_games(obs, actions, eps)
		
		self.X = np.array(X)
		self.y = np.array(y)
		self.input_dim = obs.shape[1]
		self.action_space = actions.shape[1]
		#import IPython; IPython.embed()

		return X, y, eps

	def separate_player_obs(self, X, y, players=2):
		""" Seperates observations into what each player sees
		Returns:
			X_sep_all (np arr): obs belonging to a single agent
			y_sep_all (np arr): labels belonging to a single agent
		"""
		X_sep_all = []
		y_sep_all = []
		for player in range(players):
			X_sep_all.append(np.asarray(X[player::players]))
			y_sep_all.append(np.asarray(y[player::players]))
			#print("X_sep: {}".format(X_sep_all[player].shape))
		return X_sep_all, y_sep_all

	def generate_conv1d(self, X, y):
		i = 0
		while True:
			X_sep_all, y_sep_all = self.separate_player_obs(X[i], y[i]) 
			for X_sep, y_sep in zip(X_sep_all, y_sep_all):
				#print("X_sep: {}".format(X_sep.shape))
				X_train = self.reshape_data(X_sep)
				y_train = self.reshape_data(y_sep)
				# X_train = np.reshape(X_sep,(1,X_sep.shape[0],X_sep.shape[1],1))
				# y_train = np.reshape(y_sep,(1,y_sep.shape[0],y_sep.shape[1],1))
				#ip.embed()
				yield X_train, y_train
			i = (i + 1) % len(X)

	def generate(self, X, y):
		"""
		Generate trainable matrices per game_episode.
		Necessary since we need to train game by game (instead of move by move)...
		but games have varying length.
		"""
		i = 0
		while True:
			X_sep_all, y_sep_all = self.separate_player_obs(X[i], y[i]) 
			for X_sep, y_sep in zip(X_sep_all, y_sep_all):
				#print("X_sep: {}".format(X_sep.shape))
				X_train = self.reshape_data(X_sep)
				y_train = self.reshape_data(y_sep)
				# X_train = np.reshape(X_sep,(1,X_sep.shape[0],X_sep.shape[1]))
				# y_train = np.reshape(y_sep,(1,y_sep.shape[0],y_sep.shape[1]))
				#ip.embed()
				yield X_train, y_train
			i = (i + 1) % len(X)

	def fit(self, epochs=100, batch_size=1, learning_rate=0.001, val_split=None):

		adam = optimizers.Adam(
			lr=learning_rate,
			beta_1=0.9,
			beta_2=0.999,
			epsilon=None,
			decay=0.0,
			amsgrad=False)

		self.model.compile(loss='cosine_proximity',
						   optimizer=adam, metrics=['accuracy'])

		# Create checkpoint directory
		if not os.path.exists(self.checkpoint_path):
			try:
				os.makedirs(self.checkpoint_path)
			except OSError:
				print("Creation of the directory %s failed" % self.checkpoint_path)

		checkpoints = ModelCheckpoint(
						os.path.join(self.checkpoint_path, 'weights{epoch:08d}.h5'), 
						save_weights_only=True, period=50)
		#plot_acc = PlotAcc()
		tensorboard = TensorBoard(log_dir="./logs")
		
		if val_split == None:
			#ip.embed()
			self.model.fit_generator(
				self.generate(self.X, self.y),
				steps_per_epoch = self.X.shape[0]/batch_size,
				epochs = epochs,
				callbacks = [checkpoints])
		else:
			X_train, X_test, y_train, y_test = self.create_validation_split(val_split)
			#ip.embed()
			self.model.fit_generator(
					self.generate(X_train, y_train),
					steps_per_epoch = X_train.shape[0]/batch_size,
					epochs = epochs,
					validation_data=self.generate(X_test, y_test),
					validation_steps=X_test.shape[0]/batch_size,
					callbacks = [checkpoints, tensorboard])
		
		# Test predict
		#print(self.predict(self.X[0]))

	# def predict(self,X):
	# 	X = self.reshape_data(X)
	# 	self.model.predict()

	# CURRENTLY UNUSED. TODO: REWRITE
	def perform_lstm_cross_validation(k, max_ep):
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

