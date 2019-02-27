from __future__ import print_function

import sys
import getopt
import rl_env
import random

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

			done = False
			eps_reward = 0

			while not done:
				for player in range(self.players):
					self.print_obs(obs)
					ob = obs['player_observations'][player]
					action = random.choice(ob['legal_moves'])
					print('Agent: {} action: {}'.format(obs['current_player'], action))
					obs, reward, done, _ = self.env.step(action)
					eps_reward += reward
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

	def parse_observations(observations, num_actions, obs_stacker):
		""" PULLED FROM rainbow/run_experiment.py, FOR REFERENCE ONLY
		Deconstructs the rich observation data into relevant components.
		Args:
			observations: dict, containing full observations.
			num_actions: int, The number of available actions.
			obs_stacker: Observation stacker object.
		Returns:
			current_player: int, Whose turn it is.
			legal_moves: `np.array` of floats, of length num_actions, whose elements
				are -inf for indices corresponding to illegal moves and 0, for those
				corresponding to legal moves.
			observation_vector: Vectorized observation for the current player.
		"""
		current_player = observations['current_player']
		current_player_observation = (
			observations['player_observations'][current_player])
		legal_moves = current_player_observation['legal_moves_as_int']
		legal_moves = format_legal_moves(legal_moves, num_actions)
		observation_vector = current_player_observation['vectorized']
		obs_stacker.add_observation(observation_vector, current_player)
		observation_vector = obs_stacker.get_observation_stack(current_player)
		return current_player, legal_moves, observation_vector

	def parse_observations(obs):
		""" Parses current observation
		Args:
			obs (dict): Full observations
		Returns:
			curr_player (int): Whose turn it is.
			legal_moves (np.array of floats): Illegal moves are -inf, and legal moves are 0.
			obs_vector (list): Vectorized observation for current player.
		"""
		curr_player = obs['current_player']
		curr_player_obs = obs['player_observations'][current_player]
		legal_moves = curr_player_obs['legal_moves_as_int']
		obs_vector = curr_player_obs['vectorized']

		return curr_player, legal_moves, obs_vector


if __name__ == "__main__":
	runner = Runner(3,1)
	runner.run()
	#runner.run()