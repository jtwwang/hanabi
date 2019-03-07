import tensorflow.keras as keras
from keras.layers import Dense
from keras.layers import ReLU
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Sequential
import numpy as np

class policy_predictor():
	def __init__(self, input_dim, action_space):
		self.input_dim=input_dim
		self.action_space = action_space
		self.model = self.create_dense()

	def create_dense(self):
		x = Sequential()
		x.add(Dense(16, input_dim=self.input_dim))
		x.add(Dense(16))
		x.add(Dense(16, activation='relu'))
		x.add(Dense(self.action_space, activation='softmax'))
		return x

	def fit(self, X, y, epochs=100, batch_size=1):
		"""
		args:
			X (int arr): vectorized features
			y (int arr): one-hot encoding with dimensions(sample_size,action_space)
		"""
		self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		print()
		self.model.fit(X,y,epochs=epochs,batch_size=batch_size)
	
	def predict(self, X):
		pred = self.model.predict(X)
		return pred


if __name__ == '__main__':
	pp = policy_predictor(2, 2)
	X = np.asarray([[1,2],
		[3,4]
	])
	y = np.asarray([[1,0],
		[0,1]
	])
	print("init done")
	pp.fit(X,y)
	pred = pp.predict(X)
	print(pred)


