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



	def predict(self, X):
		"""
		args:
			X (input)
		return
			prediction given the model and the input X
		"""
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






class lstm_nn(policy_pred):
	def __init__(self, input_dim, action_space, agent_class, model_name=None):
		super().__init__(input_dim, action_space, agent_class, model_name)
		self.model_type = "lstm"

	def create_model(self):
		x = Sequential()
		x.add(LSTM(64, input_shape=(None,self.input_dim), return_sequences=True))
		x.add(LSTM(128, return_sequences=True))
		x.add(TimeDistributed(Dense(self.action_space)))
		print(x.summary())
		return x

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


	def train_generator(self, X, y):
		i = 0
		while True:
			X_sep_all, y_sep_all = self.separate_player_obs(X[i], y[i])
			for X_sep, y_sep in zip(X_sep_all, y_sep_all):
				#print("X_sep: {}".format(X_sep.shape))
				X_train = np.reshape(X_sep,(1,X_sep.shape[0],X_sep.shape[1]))
				y_train = np.reshape(y_sep,(1,y_sep.shape[0],y_sep.shape[1]))
				yield X_train, y_train
			i = (i + 1) % len(X)

	def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=100, batch_size=32, learning_rate=0.001):
		"""
		args:
				X (int arr): vectorized features
				y (int arr): one-hot encoding with dimensions(sample_size,action_space)
		"""
		
		X_train = np.asarray(X_train)
		y_train = np.asarray(y_train)
		X_test = np.asarray(X_test)
		y_test = np.asarray(y_test)
		#print(type(self.X))
		#print(X[0])


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

		checkpoints = keras.callbacks.ModelCheckpoint(
									os.path.join(self.checkpoint_path, 'weights{epoch:08d}.h5'), 
									save_weights_only=True, period=50)
		#plot_acc = PlotAcc()
		tensorboard = keras.callbacks.TensorBoard(log_dir="./logs")
		evaluationAcc = EvalAcc()
		
		if X_test == None:
			self.model.fit_generator(
				self.train_generator(X_train, y_train),
				steps_per_epoch = X_train.shape[0]/batch_size,
				epochs = epochs,
				callbacks = [checkpoints, tensorboard, evaluationAcc])
		else:
			self.model.fit_generator(
					self.train_generator(X_train, y_train),
					steps_per_epoch = X_train.shape[0],
					epochs = epochs,
					validation_data=self.train_generator(X_test, y_test),
					validation_steps=X_test.shape[0],
					#validation_freq=2,
					callbacks = [checkpoints, tensorboard])

		self.save()

	

#TODO: Put this inside nn class
def extract_data(agent_class):
	"""
	args:
		agent_class (string)
		num_player (int)
	"""
	print("Loading Data...", end='')
	replay = Experience(agent_class, load=True)
	replay.load()
	r = replay._obs()
	m = replay._one_hot_moves()
	eps = replay.eps
	assert r.shape[0] == m.shape[0]

	# reshape the inputs for the lstm layer
	X,y = [],[]
	for e in eps:
		X.append(r[range(e[0],e[1])])
		y.append(m[range(e[0],e[1])])

	print("LOADED")
	return X, y, eps


if __name__ == "__main__":
	flags = {'epochs': 40,
			 'batch_size': 1,
			 'lr': 0.001,
			 'agent_class': 'SimpleAgent',
			 'cv': -1,
			 'load': False}

	options, arguments = getopt.getopt(sys.argv[1:], '',
									   ['epochs=',
										'batch_size=',
										'lr=',
										'agent_class=',
										'load='])
	if arguments:
		sys.exit()
	for flag, value in options:
		flag = flag[2:]  # Strip leading --.
		flags[flag] = type(flags[flag])(value)

	X, Y, _ = extract_data(flags['agent_class'])

	#TODO: Automate input/output dim generation? Generate based on lstm, cnn, dense... in class.
	input_dim = X[0].shape[1]
	output_dim = Y[0].shape[1]

	pp = dense_pred(input_dim, output_dim, flags['agent_class'])
	
	if flags['load']:
		pp.load()

	pp.fit(X, Y,
		   epochs=flags['epochs'],
		   batch_size=flags['batch_size'],
		   learning_rate=flags['lr'])
   
