# Artificial Intelligence Society at UC Davis
#
#   http://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
"""A script that implements a Monte Carlo Tree search"""

from rl_env import Agent
import numpy as np
import baysian_belief
import random
import pyhanabi
from copy import copy, deepcopy
from policy_predictor import policy_net
from datetime import datetime
random.seed(datetime.now())
from state_translate import state_translator

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

    def sampled_vector(self, vectorized):
        """ 
        returns a sample of observations from belief and observations
        Args:
            vectorized (list): the observations in a vectorized form
        """
        v_sample = []
        translated = state_translator(vectorized, self.config['players'])
        print(self.my_belief.shape)
        card = self.my_belief[0]
        for i in range(self.my_belief.shape[0]):
            v = self.sample(card).tolist()
            v_sample = v_sample + v
            print(len(v_sample))
            #self.my_belief = 

        print(len(translated.handSpace))

        return v_sample

    def encode(self, ob):
        """returns an hashable state from the observation"""
        return str(ob)

    def update_visits(self, state):
        """
        This function updates the visits statics for a certain state
        Args: state (string) the encoded representation of a string
        """
        if state not in self.stats.keys():       # update_node
            self.stats[state] = {'visits': 1, 'value' : 0}
        else:
            self.stats[state]['visits'] += 1

    def act(self, obs, env):
        self.stats = {} # create empty dictionary to save stats of all nodes

        self.root = env.state               # set the root from the state of the game
        vectorized = env.observation_encoder.encode(self.root.observation(self.player_id))
        state = self.encode(vectorized)     # make it hashable
        self.update_visits(state)           # increment number of visits
       
        # random rollouts
        rollouts = 100
        for r in range(rollouts):   
            depth = 0                       # reset the depth
            history = []                    # reset the history of the rollout
            game_state = self.root          # reset the state of the rollout

            while depth < self.max_depth:

                # choose random action
                legal_moves = game_state.legal_moves()
                move = random.choice(legal_moves)

                nextHanabiState = game_state.copy() # make a copy
                nextHanabiState.apply_move(move)    # make a move

                # if it's a play or discard move, need to deal the cards
                if nextHanabiState.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                    nextHanabiState.deal_random_card()

                vectorized = env.observation_encoder.encode(nextHanabiState.observation(self.player_id))
                state = self.encode(vectorized)     # make it hashable
                self.update_visits(state)           # increment number of visits
                history.append(state)               # append the state to the history of the rollout

                # set my belief
                self.my_belief = self.belief.encode(obs, self.player_id)

                # other players' moves
                for p in range(self.config['players']):
                    # choose random sample
                    vectorized = self.sampled_vector(vectorized)
                    # predict the move

                    # if move is hint, calculate the score and terminate

                    # else, make the move

                # increase the depth by 1
                depth += 1
            
            for i in range(len(history)):
                pass
        
        return random.choice(obs['legal_moves']) #FIXME: 

