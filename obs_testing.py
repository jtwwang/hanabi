from __future__ import print_function

import sys
import getopt
import rl_env
import random
import numpy as np

class Runner(object):
	"""Runner class."""
	def __init__(self, numAgents, numEpisodes):
		self.eps = numEpisodes
		self.players = numAgents
		self.env = rl_env.make(num_players=numAgents)
		

	def run(self):
		# Run a random agent for self.eps episodes and print max reward.
		rewards = []
		for eps in range(self.eps):
			print('Running episode: %d' % eps)

			obs = self.env.reset() # Observation of all players
			action_dim = self.env.num_moves() # Total number of possible actions, illegal or not.

			done = False
			eps_reward = 0

			while not done:
				for player in range(self.players):
					print('Agent:{}\n'.format(obs['current_player']))
					#self.print_obs(obs)

					_, legal_moves, obs_vector = self.parse_obs(obs, action_dim)
					print('Legal Moves: {}\n'.format(legal_moves))
					print('Obs Vector: {}\n'.format(obs_vector))

					ob = obs['player_observations'][player]
					action = random.choice(ob['legal_moves'])
					print('Action: {}\n'.format(action))
					obs, reward, done, _ = self.env.step(action)
					eps_reward += reward
					print('---------------\n')
			rewards.append(eps_reward)
			
		print('Max Reward: %.3f' % max(rewards))

	def print_obs(self, obs):
		# Print observations, given the environment.
		keys = list(obs.keys())
		print('\nObservation entries: {}\n'.format(keys))

		# Player observations
		print('------------ {} -------------'.format(keys[0]))
		val = obs[keys[0]]
		for item in val:
			for k,v in item.items():
				print('{} {}: {}\n'.format(k,type(v),v))

		# Current player
		print('------------ {} -------------'.format(keys[1]))
		val = obs[keys[1]]
		print('{}: {}\n\n'.format(type(val),val))

	def format_legal_moves(self, legal_moves, action_dim):
		"""Returns formatted legal moves.

		This function takes a list of actions and converts it into a fixed size vector
		of size action_dim. If an action is legal, its position is set to 0 and -Inf
		otherwise.
		Ex: legal_moves = [0, 1, 3], action_dim = 5
			returns [0, 0, -Inf, 0, -Inf]

		Args:
			legal_moves: list of legal actions.
			action_dim: int, number of actions.

		Returns:
			a vector of size action_dim.
		"""
		new_legal_moves = np.full(action_dim, -float('inf'))
		if legal_moves:
			new_legal_moves[legal_moves] = 0
		return new_legal_moves

	def parse_obs(self, obs, action_dim):
		""" Parses current observation
		Args:
			obs (dict): Full observations
			action_dim (int): Number of total actions, including illegal.

		Returns:
			curr_player (int): Whose turn it is.
			legal_moves (np.array of floats): Illegal moves are -inf, and legal moves are 0.
			obs_vector (list): Vectorized observation for current player.
		"""
		curr_player = obs['current_player']
		curr_player_obs = obs['player_observations'][curr_player]
		legal_moves = curr_player_obs['legal_moves_as_int']
		legal_moves = self.format_legal_moves(legal_moves, action_dim)
		obs_vector = curr_player_obs['vectorized']

		return curr_player, legal_moves, obs_vector


if __name__ == "__main__":
	runner = Runner(3,1)
	runner.run()
	#runner.run()