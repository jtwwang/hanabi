"""
Newest Card Player, a rule-based agent

Plays newest hinted cards if receives a hint.
Otherwise, gives hints about playable cards if the newest card affected is playable.
Otherwise, discards oldest card.

Source: https://github.com/chikinn/hanabi/blob/master/players/newest_card_player.py
Ported by: Justin Wang
"""
from rl_env import Agent

import IPython as ip
from collections import Counter


class NewestCardAgent(Agent):
	"""Agent that applies a simple heuristic."""

	def __init__(self, config, *args, **kwargs):
		"""Initialize the agent."""
		self.config = config
		# Extract max info tokens or set default to 8.
		self.max_information_tokens = config.get('information_tokens', 8)

	# @staticmethod
	# def playable_card(card, fireworks):
	# 	"""A card is playable if it can be placed on the fireworks pile."""
	# 	return card['rank'] == fireworks[card['color']]

	@staticmethod
	def playable_card(card, fireworks):
		"""A card is playable if it can be placed on the fireworks pile."""
		if card['color'] == None and card['rank'] != None:
			for color in colors:
				if fireworks[color] == card['rank']:
					continue
				else:
					return False
				
			return True
		elif card['color'] == None or card['rank'] == None:
			return False
		else:
			return card['rank'] == fireworks[card['color']]

	def get_recent_actions(self, action_hist, current_player):
		"""Returns actions since the current agent last played. 
		   Does not include current player's last action.

		Args:
			action_hist (list): Nested list of all actions taken.
				Dimensions: (players, moves)
			current_player (int): absolute player id.

		Returns:
			recent_actions (list): Actions since the current agent last played. 
				Most recent move first.
				Dimensions: (players, move)
				Returns 'action_type'
		
		TODO: Make history indexing relative and not absolute?
		(e.g. index 0 will be previous player always)
		"""
		recent_actions = []
		players = self.config['players']
		# Get other player id's, from most recent to least recent
		other_players = []
		for i in range(players-1):
			other_players.append((current_player - i - 1)%players)
		for player_id in other_players:
			# Player_id is absolute, not relative.
			player_moves = action_hist[player_id] # History of a single player's moves.
			if len(player_moves) > 0:
				recent_actions.append(player_moves[-1])
			else:
				recent_actions.append({'action_type': 'None'}) # Did not make a move
		return recent_actions

	def check_hint_target(self, action, i):
		# Checks if @action is a hint and if recipient is current player
		"""
		Args:
			action (dict): Action type, rank/color hinted, index targeted, player offset
			i (int): Corresponds to a player id.

		Returns:
			True if hint target is current player.
		"""
		if action['action_type'].startswith('REVEAL'):
			if action['target_offset'] == i+1:
				return True
		return False

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
		#ip.embed()


		# Play the newest card affected by a hint if a new hint was received since last turn
		action_hist = observation['action_hist']
		recent_actions = self.get_recent_actions(action_hist, observation['current_player'])
		#ip.embed()
		for i, action in enumerate(recent_actions):
			if self.check_hint_target(action, i):
				# Newest card affected by the hint
				newest_card_index = max(action['indices_affected'])
				return {'action_type': 'PLAY', 'card_index': newest_card_index}
		

		# Hint the newest playable card
		

		fireworks = observation['fireworks']
		if observation['information_tokens'] > 0:
			# Check if there are any playable cards in the hands of the opponents.
			for player_offset in range(1, observation['num_players']):
				already_hinted = Counter() # Check if the card is the newest card of this color
				player_hand = observation['observed_hands'][player_offset]
				player_hints = observation['card_knowledge'][player_offset]
				# Check if the card in the hand of the opponent is playable.
				for card, hint in reversed(zip(player_hand, player_hints)):
					color = card['color']
					already_hinted[color] += 1 
					if (
						NewestCardAgent.playable_card(card, fireworks) and 
						hint['color'] is None and
						already_hinted[color] == 1
					   ):
						# print("HINTING: " + card['color'] + '\n')
						return {
							'action_type': 'REVEAL_COLOR',
							'color': color,
							'target_offset': player_offset
						}

		# If no card is hintable then discard or play oldest card.
		if observation['information_tokens'] < self.max_information_tokens:
			return {'action_type': 'DISCARD', 'card_index': 0}
		else:
			return {'action_type': 'PLAY', 'card_index': 0}
