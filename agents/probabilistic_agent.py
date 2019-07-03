# Developed by Lorenzo Mambretti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rl_env import Agent
import bayes

Weights = {
    "indirectHintWeight" : 1.0,
    # First index is if given the turn it was drawn, last index if drawn after that
    "directHintWeight" : [5.0, 1.0],

    # Weight based on location in hand, far left is newest card
    "directHintOrderWeight" : [2.0, 1.5, 1.0, 1.0, 1.0],

    # Weight based on order given by other player (eg, if given one card, very likely playable)
    "directHintInHintOrderWeight" : [5.0, 2.0, 1.0, 1.0, 1.0]
}

class ProbabilisticAgent(Agent):
    """
    Heuristic agent that looks at the probability that a given card is playable
    and play it if above a threshold value that varies based on number of bombs
    played.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def is_likely_playable(self, card, fireworks, life_tokens):
        raise NotImplementedError()

    def scale_probability(self, probability, scale):
        return probability ** (1.0 / scale)

    def playable_card(self, card, fireworks):
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

    def act(self, ob):
        """Act based on an ob."""
        if ob['current_player_offset'] != 0:
            return None

        fireworks = ob['fireworks']
        life_tokens = ob['life_tokens']

        # Check if it's likely that we have a card to play
        for card_index, hint in enumerate(ob['card_knowledge'][0]):
            if self.is_likely_playable(hint, fireworks, life_tokens):
                return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if it's possible to hint a card to your colleagues.
        if ob['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the opponents.
            for player_offset in range(1, ob['num_players']):
                player_hand = ob['observed_hands'][player_offset]
                player_hints = ob['card_knowledge'][player_offset]
                # Check if the card in the hand of the opponent is playable.
                for card, hint in zip(player_hand, player_hints):
                    if self.playable_card(card,fireworks) and hint['color'] is None:
                        return {
                            'action_type': 'REVEAL_COLOR',
                            'color': card['color'],
                            'target_offset': player_offset
                        }
                    elif self.playable_card(card, fireworks) and hint['rank'] is None:
                        return {
                                'action_type': 'REVEAL_RANK',
                                'rank': card['rank'],
                                'target_offset': player_offset
                                }

        # If no card is hintable then discard or play.
        if ob['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': 0}
        else:
            return {'action_type': 'PLAY', 'card_index': 0}

