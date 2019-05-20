import getopt
import sys
import numpy as np

from experience import Experience

from predictors.conv_pred import conv_pred
from predictors.dense_pred import dense_pred



#DEBUGGING
import IPython as ip

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

	# # X, Y, _ = extract_data(flags['agent_class'])
	# # X = np.array(X)
	# # Y= np.array(Y)

	# #TODO: Automate input/output dim generation? Generate based on lstm, cnn, dense... in class.
	# input_dim = len(X)
	# output_dim = len(Y)
	agent_class = flags['agent_class']

	pp = conv_pred(agent_class)
	pp.extract_data(agent_class)
	pp.create_model() # Add Model_name here to create different models

	#ip.embed()
	
	if flags['load']:
		pp.load()

	pp.fit(epochs=flags['epochs'],
		   batch_size=flags['batch_size'],
		   learning_rate=flags['lr'])