''' observation breakdown

own card knowledge
	- `n` * `hand_size` * 35 bits for card knowledge
		- stack own hand, concat other players.

other players' hands:
	- `n-1 * hand_size` cards, at 25 bits/card; 

	- 1 bit per player to show if player is missing a card (stack 5x5 of ones)

============ Consider 0 to 1, one neuron
deck+ tokens, thermometer encoding:
	- 1 bit/ card in deck; ( 39 1's one 0 after drawing)
	- 8 bits for info token 
	- 3 bits for life tokens
=============

discard:
	- 50 bits for discard, 5 blocks of color [1,1,1,2,2,3,3,4,4,5] R1R2: 100100...


player actions:
	- last actions, `n` bits for active player, one-hot
	- 4 bits for movetype, one-hot
	- `n` bits for last move target, one-hot
	- 1 bit: whether or not it was successful (
	- 1 bit: whether or not we added an information token

	- 5 bits for color revealed, one-hot (seperate)
	- 5 bits for rank revealed, one-hot (seperate)
	- `hand_size` bits for which card was hinted at, one-hot (seperate)
	- `hand_size` bits for which card was just played, one-hot (seperate)

board (may concat with discard):	
	- 25 bits for board, 5 blocks of color (5 inverse therm) R1: 10000. R2B1:1100010000
	- 25 bits for card just played/discarded, one-hot (pyt with player hands)


activity regularizer after concat to avoid overfit.

'''




'''
Assuming `n` player game:
- `n-1 * hand_size` cards, at 25 bits/card; 1 bit per player to show if player is missing a card
- boards 1 bit/ card in deck;
- 25 bits for board, 5 blocks of color


- 8 bits for info token
- 3 bits for life tokens


- 50 bits for discard, 5 blocks of color [1,1,1,2,2,3,3,4,4,5]


- last actions, `n` bits for active player, one-hot
- 4 bits for movetype, one-hot
- `n` bits for last move target, one-hot
- 5 bits for color revealed, one-hot
- 5 bits for rank revealed, one-hot
- `hand_size` bits for which card was hinted at, one-hot
- `hand_size` bits for which card was just played, one-hot
- 25 bits for card just played/discarded, one-hot (new layer)

- `n` * `hand_size` * 35 bits for card knowledge
- 1 bit: whether or not it was successful
- 1 bit: whether or not we added an information token

- all cardknowledge: player_count * handSize * 35










'''

# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from .policy_pred import policy_pred

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization, concatenate
from tensorflow.keras.layers import Activation, Dropout, Input

from tensorflow.keras.utils import plot_model

from .blocks import conv_block

import numpy as np

import IPython as ip

class split_input_pred(policy_pred):
	def __init__(self, agent_class, predictor_name='predictor'):
		self.model_type = "split_input"
		super(split_input_pred, self).__init__(agent_class, 
			model_type=self.model_type, 
			predictor_name=predictor_name)


	def create_model_long(self, img_path='multihead_conv_long.png'):
		input_shape = (self.input_dim,1)
		inputs = Input(shape=input_shape)
		activation=None
		# TOWER 1
		tower_1 = Conv1D(filters=16, kernel_size=7, strides=2,
			padding="same", activation=activation)(inputs)
		tower_1 = MaxPooling1D(pool_size=3, strides=2) (tower_1)
		tower_1 = BatchNormalization()(tower_1)
		tower_1 = Activation("relu")(tower_1)

		tower_1 = Conv1D(filters=32, kernel_size=3, strides=2,
			padding="same", activation=activation)(inputs)
		tower_1 = MaxPooling1D(pool_size=3, strides=2) (tower_1)
		tower_1 = BatchNormalization()(tower_1)
		tower_1 = Activation("relu")(tower_1)

		# TOWER 2
		tower_2 = Conv1D(filters=16, kernel_size=5, strides=2,
			padding="same", activation=activation)(inputs)
		tower_2 = MaxPooling1D(pool_size=3, strides=2)(tower_2)
		tower_2 = BatchNormalization()(tower_2)
		tower_2 = Activation("relu")(tower_2)

		tower_2 = Conv1D(filters=32, kernel_size=3, strides=2,
			padding="same", activation=activation)(inputs)
		tower_2 = MaxPooling1D(pool_size=3, strides=2) (tower_2)
		tower_2 = BatchNormalization()(tower_2)
		tower_2 = Activation("relu")(tower_2)

		# TOWER 3
		tower_3 = Conv1D(filters=16, kernel_size=3, strides=2,
			padding="same", activation=activation)(inputs)
		tower_3 = MaxPooling1D(pool_size=3, strides=2)(tower_3)
		tower_3 = BatchNormalization()(tower_3)
		tower_3 = Activation("relu")(tower_3)

		tower_3 = Conv1D(filters=32, kernel_size=3, strides=2,
			padding="same", activation=activation)(inputs)
		tower_3 = MaxPooling1D(pool_size=3, strides=2) (tower_3)
		tower_3 = BatchNormalization()(tower_3)
		tower_3 = Activation("relu")(tower_3)

		merged = concatenate([tower_1, tower_2, tower_3], axis=1)
		merged = Flatten()(merged)

		out = Dense(128, activation='relu')(merged)
		out = Dense(self.action_space, activation='softmax')(out)

		model = Model(inputs, out)
		plot_model(model, to_file=img_path)
		self.model = model
		return model

	def create_model(self, img_path='multihead_conv.png'):
		input_shape = (self.input_dim,1)
		inputs = Input(shape=input_shape)
		activation=None
		# TOWER 1
		tower_1 = Conv1D(filters=32, kernel_size=7, strides=2,
			padding="same", activation=activation)(inputs)
		tower_1 = MaxPooling1D(pool_size=3, strides=2) (tower_1)
		tower_1 = BatchNormalization()(tower_1)
		tower_1 = Activation("relu")(tower_1)

		# TOWER 2
		tower_2 = Conv1D(filters=32, kernel_size=5, strides=2,
			padding="same", activation=activation)(inputs)
		tower_2 = MaxPooling1D(pool_size=3, strides=2)(tower_2)
		tower_2 = BatchNormalization()(tower_2)
		tower_2 = Activation("relu")(tower_2)

		# TOWER 3
		tower_3 = Conv1D(filters=32, kernel_size=3, strides=2,
			padding="same", activation=activation)(inputs)
		tower_3 = MaxPooling1D(pool_size=3, strides=2)(tower_3)
		tower_3 = BatchNormalization()(tower_3)
		tower_3 = Activation("relu")(tower_3)

		merged = concatenate([tower_1, tower_2, tower_3], axis=1)
		merged = Flatten()(merged)

		out = Dense(128, activation='relu')(merged)
		out = Dense(self.action_space, activation='softmax')(out)
		out = Dropout(0.2)(out)

		model = Model(inputs, out)
		plot_model(model, to_file=img_path)
		self.model = model
		return model


	def reshape_data(self, X_raw):
		if X_raw.shape == (self.action_space,):  # If only one sample is put in
			X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

		# Add an additional dimension for filters
		#X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
		#ip.embed()
		input_splice(X_raw)
		return X

class input_splice:
	def __init__(self, X_raw, player_count=2):
		# X_raw (2d nparr): (num_samples, vectorized_len) Raw inputs.
		# player_count (int): Number of players
		self.X_raw = X_raw
		self.player_count = player_count
		self.X_spliced = []
		self.raw_slice()

	def decode_vector(self, stateVector):
		if(self.player_count == 2 or self.player_count == 3):
			handSize = 5
		elif(self.player_count == 4 or self.player_count == 5):
			handSize = 4
		else:
			raise ValueError('player_count is invalid number')

		prevIndex = 0
		self.handSize = handSize

		numCardsSeen = (self.player_count - 1) * handSize
		self.handSpace = stateVector[prevIndex:(prevIndex+numCardsSeen*25)]
		prevIndex += numCardsSeen*25

		self.playerMissingCards = stateVector[prevIndex:(
			prevIndex+self.player_count)]
		prevIndex += self.player_count

		# Assumes 50 cards in game total.
		deckSize = 50 - (self.player_count * handSize)
		self.currentDeck = stateVector[prevIndex:(prevIndex+deckSize)]
		prevIndex += deckSize

		maxBoardSpace = 25
		self.boardSpace = stateVector[prevIndex:(prevIndex+maxBoardSpace)]
		prevIndex += maxBoardSpace

		numInfoTokens = 8
		self.infoTokens = stateVector[prevIndex:(prevIndex+numInfoTokens)]
		prevIndex += numInfoTokens

		numLifeTokens = 3
		self.lifeTokens = stateVector[prevIndex:(prevIndex + numLifeTokens)]
		prevIndex += numLifeTokens

		self.discardSpace = stateVector[prevIndex:(prevIndex + 50)]
		prevIndex += 50

		self.lastActivePlayer = stateVector[prevIndex:(
			prevIndex + self.player_count)]
		prevIndex += self.player_count

		self.lastMoveType = stateVector[prevIndex:(prevIndex + 4)]
		prevIndex += 4

		self.lastMoveTarget = stateVector[prevIndex:(
			prevIndex + self.player_count)]
		prevIndex += self.player_count

		self.colorRevealed = stateVector[prevIndex:(prevIndex + 5)]
		prevIndex += 5

		self.rankRevealed = stateVector[prevIndex:(prevIndex + 5)]
		prevIndex += 5

		self.cardRevealed = stateVector[prevIndex:(prevIndex + handSize)]
		prevIndex += handSize

		# in the context of the next two arrays, 'played' means played or discarded
		self.positionPlayed = stateVector[prevIndex:(prevIndex + handSize)]
		prevIndex += handSize

		self.cardPlayed = stateVector[prevIndex:(prevIndex + 25)]
		prevIndex += 25

		self.prevPlay = stateVector[prevIndex:(prevIndex + 2)]
		prevIndex += 2

		self.cardKnowledge = stateVector[prevIndex:(
			prevIndex + self.player_count * handSize * 35)]

	def get_own_card_knowledge(self):
		'''
		own card knowledge
			- `n` * `hand_size` * 35 bits for card knowledge
			- stack own hand, concat other players.
		'''
		own_card_knowledge = np.array(self.cardKnowledge)
		stacked_shape = (self.player_count, self.handSize, 35)
		own_card_knowledge_stacked = np.reshape(own_card_knowledge,stacked_shape)
		return own_card_knowledge_stacked
		

	def get_other_players_hands(self):
		'''
		other players' hands:
			- `n-1 * hand_size` * 25 bits/card; 

			- 1 bit per player to show if player is missing a card (stack 5x5 of ones)
		'''
		other_players_hands = np.array(self.handSpace)
		stacked_shape = (self.player_count-1,self.handSize,25)
		other_players_hands_stacked = np.reshape(other_players_hands,stacked_shape)
		#TODO: Concat the one bit for missing card in a plane
		return other_players_hands_stacked

	def get_deck_and_tokens(self):
		'''
		deck+ tokens, thermometer encoding:
			- 1 bit/ card in deck; ( 39 1's one 0 after drawing)
			- 8 bits for info token 
			- 3 bits for life tokens
		'''
		# Currently, % and not integers. Should not matter.
		deck_card_count = float(np.count_nonzero(self.currentDeck)) / len(self.currentDeck)
		info_token_count = float(np.count_nonzero(self.infoTokens)) / len(self.infoTokens)
		life_token_count = float(np.count_nonzero(self.lifeTokens)) / len(self.lifeTokens)
		deck_and_tokens = np.array([deck_card_count, info_token_count, life_token_count])
		ip.embed()
		return deck_and_tokens

	def get_discard(self):
		discard = self.discardSpace
		stacked_shape = (5,10) # color, bool->[1,1,1,2,2,3,3,4,4,5]
		discard_stacked = np.reshape(self.discardSpace, stacked_shape)
		ip.embed()
		return discard_stacked

	def get_player_actions(self):
		last_active_player = self.lastActivePlayer # player_count
		last_move_target = self.lastMoveTarget # player_count
		last_move_type = self.lastMoveType # 4
		color_revealed = self.colorRevealed # 5
		rank_revealed = self.rankRevealed # 5
		card_revealed = self.cardRevealed # hand_size
		position_played = self.positionPlayed # hand_size
		card_played = self.cardPlayed # 25
		prev_play_success = self.prevPlay # 2 ; prev play success + was info token added. TODO: Split.
		return [] # STACK ALL?
	
	def get_board(self):
		board = np.array(self.boardSpace)
		stacked_shape = (5,5) # color, rank (one-hot)
		board_stacked = np.reshape(board, stacked_shape) # Is thermometer or one-hot better?
		ip.embed()
		return board_stacked

	def raw_slice(self):
		# Reads the vectorized input into 
		X_raw_spliced = []
		own_card_knowledge = []
		for i, state in enumerate(self.X_raw):
			self.decode_vector(state)
			own_card_knowledge = self.get_own_card_knowledge()
			other_players_hands = self.get_other_players_hands()
			deck_and_tokens = self.get_deck_and_tokens()
			discard = self.get_discard()
			player_actions = self.get_player_actions()
			board = self.get_board()

			

			
'''
own card knowledge
	- `n` * `hand_size` * 35 bits for card knowledge
		- stack own hand, concat other players.

other players' hands:
	- `n-1 * hand_size` cards, at 25 bits/card; 

	- 1 bit per player to show if player is missing a card (stack 5x5 of ones)

============ Consider 0 to 1, one neuron
deck+ tokens, thermometer encoding:
	- 1 bit/ card in deck; ( 39 1's one 0 after drawing)
	- 8 bits for info token 
	- 3 bits for life tokens
=============

discard:
	- 50 bits for discard, 5 blocks of color [1,1,1,2,2,3,3,4,4,5] R1R2: 100100...


player actions:
	- last actions, `n` bits for active player, one-hot
	- 4 bits for movetype, one-hot
	- `n` bits for last move target, one-hot
	- 1 bit: whether or not it was successful (
	- 1 bit: whether or not we added an information token

	- 5 bits for color revealed, one-hot (seperate)
	- 5 bits for rank revealed, one-hot (seperate)
	- `hand_size` bits for which card was hinted at, one-hot (seperate)
	- `hand_size` bits for which card was just played, one-hot (seperate)


	- 25 bits for card just played/discarded, one-hot (pyt with player hands)

board (may concat with discard):	
	- 25 bits for board, 5 blocks of color (5 inverse therm) R1: 10000. R2B1:1100010000
	'''