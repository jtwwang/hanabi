# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
"""A script to collect episodes from simple agent"""

from rl_env import Agent
import numpy as np
import baysian_belief
import random
import policy_prediction
from datetime import datetime
random.seed(datetime.now())


class MCAgent(Agent):
    """Agent that uses monte carlo tree search to find the optimal move"""

    player_id = -1

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent"""
        self.config = config
        self.max_depth = 1
        self.belief = baysian_belief.Belief(config['players'])

        # load the predictor
        pp = policy_predictor(config['observation_size'], config['num_moves'])
        pp.load() #FIXME: LOAD A SPECIFIC MODEL, for the agent we are competing against

    def sample(self,card):
        rand = random.random()
        v = np.zeros(25)
        for r in range(card.shape[0]):
            for c in range(card.shape[1]):
                if (rand - card[r,c] <= 0):
                    v[r * 5 + c] = 1
                    return v
                else:
                    rand -= card[r][c]

    def next_state_from_move(self, move):
        """returns the next state given a move"""

    def encode(self, obs):
        """returns an hashable state from the observation"""

    def next_state_from_predictor(self, obs):
        

    def act(self, obs):
        my_belief = self.belief.encode(obs, self.player_id)
        v_sample = np.zeros(0)
        for card in my_belief:
            v = self.sample(card)
            v_sample = np.append(v_sample,v)

        # create empty dictionary to save stats of all nodes
        stats = {}

        # set initial state, make it hashable
        state = encode(obs)

        # random rollouts
        n_rollouts = 1000
        for r in n_rollouts:
            # my move
            move = random.choice(obs['legal_moves'])
            state = next_state_from_move()

            # other players moves
            for p in self.config['players']:
                # choose random sample

        
        return random.choice(obs['legal_moves']) #FIXME: 

