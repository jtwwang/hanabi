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
from policy_predictor import policy_net
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
        agent_class = 'SimpleAgent' #FIXME: temporary

        pp = policy_net(config['observation_size'], config['num_moves'], agent_class)
        pp.load()

    def sample(self,card):
        """ sample a card from distribution"""
        rand = random.random()
        v = np.zeros(25)
        for r in range(card.shape[0]):
            for c in range(card.shape[1]):
                if (rand - card[r,c] <= 0):
                    v[r * 5 + c] = 1
                    return v
                else:
                    rand -= card[r][c]

    def sampled_obs(self, my_belief, ob):
        """ returns a sample of observations from belief and observations"""
        v_sample = np.zeros(0)
        for card in my_belief:
            v = self.sample(card)
            v_sample = np.append(v_sample,v)

        raise NotImplementedError
        return v_sample

    def next_state_from_move(self, move, state):
        """returns the next state given a move"""
        raise NotImplementedError

    def encode(self, ob):
        """returns an hashable state from the observation"""
        return str(ob)

    def decode(self, state):
        raise NotImplementedError

    def next_state_from_predictor(self, state):
        raise NotImplementedError

    def act(self, obs):
        my_belief = self.belief.encode(obs, self.player_id)
        
        # create empty dictionary to save stats of all nodes
        stats = {}

        # set initial state, make it hashable
        self.root = self.encode(obs['vectorized'])
        state = self.root

        # random rollouts
        rollouts = 1000
        for r in range(rollouts):
            depth = 0
            while depth < self.max_depth:
                # my move
                move = random.choice(obs['legal_moves'])
                state = self.next_state_from_move(move, state)

                # other players moves
                for p in self.config['players']:
                    # choose random sample
                    state = self.next_state_from_predictor(state)

                # increase the depth by 1
                depth += 1
        
        return random.choice(obs['legal_moves']) #FIXME: 

