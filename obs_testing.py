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
		print('{}: {}'.format(type(val),val))
"""
	def parse_obs(self, obs):
		# Return a vecgtorized version of the observation
		return None
"""
if __name__ == "__main__":
	runner = Runner(2,1)
	runner.run()
	#runner.run()