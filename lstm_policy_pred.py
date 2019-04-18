from __future__ import print_function
from experience import Experience
import tensorflow.keras as keras
from keras.layers import Dense, ReLU, Dropout, LSTM
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
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from IPython.display import clear_output
from time import time

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
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

class policy_net():

	def __init__(self, input_dim, action_space, agent_class):

		self.input_dim = input_dim
		self.action_space = action_space
		self.model = self.create_lstm()
		self.path = os.path.join("model", agent_class)
		self.checkpoint_path = os.path.join(self.path,'checkpoints')
		#self.moves = 0 # total number of moves across all games

	def create_lstm(self):
		x = Sequential()
		x.add(LSTM(64, input_shape=(None,self.input_dim), return_sequences=True,
			  recurrent_regularizer=regularizers.l2(0.001),
			  kernel_regularizer=regularizers.l2(0.001)))
		x.add(Dropout(rate=0.1))
		x.add(Dense(64, activation='relu'))
		x.add(Dropout(rate=0.1))
		x.add(Dense(32, activation='relu'))
		x.add(Dropout(rate=0.1))
		x.add(Dense(self.action_space))
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
		
		if X_test == None:
			self.model.fit_generator(
				self.train_generator(X_train, y_train),
				steps_per_epoch = X_train.shape[0]/batch_size,
				epochs = epochs,
				callbacks = [checkpoints])
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

	def evaluate(self, X, y):
		score = self.model.evaluate_generator(self.train_generator(X,y), steps = X.shape[0])
		return score

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


def cross_validation(k):
	global flags
	mean = 0

	X,Y,eps = extract_data(flags['agent_class'])
	kf = KFold(n_splits=k)
	i = 0
	for training_indices, testing_indices in kf.split(eps):
		print("\n==========\nSplit {}\n==========\n".format(i))
		i += 1

		# split the data
		X_train, X_test = np.asarray(X)[training_indices], np.asarray(X)[testing_indices]
		y_train, y_test = np.asarray(Y)[training_indices], np.asarray(Y)[testing_indices]

		# initialize the predictor (again)
		input_dim = X[0].shape[1]
		output_dim = Y[0].shape[1]
		pp = policy_net(input_dim, output_dim, flags['agent_class'])

		pp.fit(X_train, y_train,
			   X_test, y_test,
			   epochs=flags['epochs'],
			   batch_size=flags['batch_size'],
			   learning_rate=flags['lr'])

		# calculate accuracy and add it to the mean
		score = pp.evaluate(X_test, y_test)
		mean += score[1]

	# calculate the mean score
	mean = mean/k
	return mean


if __name__ == '__main__':

	flags = {'epochs': 400,
			 'batch_size': 32,
			 'lr': 0.001,
			 'agent_class': 'SimpleAgent',
			 'cv': -1}

	options, arguments = getopt.getopt(sys.argv[1:], '',
									   ['epochs=',
										'batch_size=',
										'lr=',
										'agent_class=',
										'cv='])
	if arguments:
		sys.exit()
	for flag, value in options:
		flag = flag[2:]  # Strip leading --.
		flags[flag] = type(flags[flag])(value)
   
	if (flags['cv'] > 0):
		# do cross validation
		mean = cross_validation(flags['cv'])
		print('Average score: ', end='')
		print(mean)
	else:
		# data
		X, Y, _ = extract_data(flags['agent_class'])
		input_dim = X[0].shape[1]
		output_dim = Y[0].shape[1]
		pp = policy_net(input_dim, output_dim, flags['agent_class'])
		pp.load()
		pp.fit(X, Y,
			   epochs=flags['epochs'],
			   batch_size=flags['batch_size'],
			   learning_rate=flags['lr'])
