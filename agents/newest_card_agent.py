"""

Newest Card Player, a rule-based agent

Plays newest hinted cards if receives a hint.
Otherwise, gives hints about playable cards if the newest card affected is playable.
Otherwise, discards oldest card.

Source: https://github.com/chikinn/hanabi/blob/master/players/newest_card_player.py
Ported by: Justin Wang
"""

# Ported code from simple_agent.py
from rl_env import Agent

import IPython as ip


class NewestCardAgent(Agent):
	"""Agent that applies a simple heuristic."""

	def __init__(self, config, *args, **kwargs):
		"""Initialize the agent."""
		self.config = config
		# Extract max info tokens or set default to 8.
		self.max_information_tokens = config.get('information_tokens', 8)
		self.hint_received = True # Was a fresh hint received?

	@staticmethod
	def playable_card(card, fireworks):
		"""A card is playable if it can be placed on the fireworks pile."""
		return card['rank'] == fireworks[card['color']]

	def act(self, observation):
		"""Act based on an observation."""
		if observation['current_player_offset'] != 0:
			return None




		# Check if there are any pending hints and play the card corresponding to
		# the hint.

		### DEBUGGING
		# print("\nObserved Hands\n")
		# print(observation['observed_hands'][1])
		# print('\nCard Knowledge:\n')
		# print(observation['card_knowledge'])
		# ip.embed()

		# Play the first color hinted if a fresh hint was received
		# TODO: Only do this on FRESH HINTS
		self_cn = observation['card_knowledge'][0]
		for card_index, hint in zip(range(len(self_cn)-1,-1,-1),reversed(self_cn)):
			if hint['color'] is not None or hint['rank'] is not None:
				# print("\nPLAYING: " + str(card_index) + '\n')
				#self.hint_received
				return {'action_type': 'PLAY', 'card_index': card_index}

		# Hint the newest playable card
		fireworks = observation['fireworks']
		if observation['information_tokens'] > 0:
			# Check if there are any playable cards in the hands of the opponents.
			for player_offset in range(1, observation['num_players']):
				player_hand = observation['observed_hands'][player_offset]
				player_hints = observation['card_knowledge'][player_offset]
				# Check if the card in the hand of the opponent is playable.
				for card, hint in reversed(zip(player_hand, player_hints)):
					if NewestCardAgent.playable_card(card,
						fireworks) and hint['color'] is None:
						# print("HINTING: " + card['color'] + '\n')
						return {
							'action_type': 'REVEAL_COLOR',
							'color': card['color'],
							'target_offset': player_offset
						}

		# If no card is hintable then discard or play oldest card.
		if observation['information_tokens'] < self.max_information_tokens:
			return {'action_type': 'DISCARD', 'card_index': 0}
		else:
			return {'action_type': 'PLAY', 'card_index': 0}
