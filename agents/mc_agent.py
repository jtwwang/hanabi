# Artificial Intelligence Society at UC Davis
#
#   http://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from state_translate import state_translator
from rl_env import Agent
import numpy as np
import baysian_belief
import random
import pyhanabi
from copy import copy, deepcopy
from policy_predictor import policy_net
from datetime import datetime
random.seed(datetime.now())


class MCAgent(Agent):
    """Agent that uses monte carlo tree search to find the optimal move"""

    player_id = -1

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent"""
        self.config = config
        self.max_depth = 2
        self.belief = baysian_belief.Belief(config['players'])

        # load the predictor
        agent_class = 'SimpleAgent'  # FIXME: temporary

        self.pp = policy_net(
            config['observation_size'], config['num_moves'], agent_class)
        self.pp.load()

    def sample(self, card):
        """ sample a card from distribution"""
        rand = random.random()
        v = np.zeros(25)
        for r in range(card.shape[0]):
            for c in range(card.shape[1]):
                if (rand - card[r, c] <= 0):
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
        translated_to_sample = state_translator(
            vectorized, self.config['players'])
        for i in range(self.my_belief.shape[0]):
            v = self.sample(self.my_belief[i]).tolist()
            v_sample = v_sample + v

            # update the card knowledge accordingly to the sample
            translated_to_sample.cardKnowledge[i*35:i*35 + 25] = v

            # update the belief accordingly to the new knowledge
            translated_to_sample.encodeVector()
            re_encoded = translated_to_sample.stateVector
            self.my_belief = self.belief.encode(re_encoded, self.player_id)

        # take the original vector, and change the observations
        translated = state_translator(vectorized, self.config['players'])
        translated.handSpace = v_sample
        translated.encodeVector()

        return translated.stateVector

    def encode(self, ob):
        """returns an hashable state from the observation"""
        return str(ob)

    def update_visits(self, state):
        """
        This function updates the visits statics for a certain state
        Args: state (string) the encoded representation of a string
        """
        if state not in self.stats.keys():       # update_node
            self.stats[state] = {'visits': 1, 'value': 0}
        else:
            self.stats[state]['visits'] += 1

    def select_from_prediction(self, obs_input, state):
        prediction = self.pp.predict(obs_input)
        # convert move to correct type
        best_value = 0
        best_move = -1
        for action in range(prediction.shape[1]):
            move = self.env.game.get_move(action)
            for i in range(len(state.legal_moves())):
                if str(state.legal_moves()[i]) == str(move):
                    if prediction[0,action] > best_value:
                        best_value = prediction[0,action]
                        best_move = move
                    break
        if best_move == -1:
            raise ValueError("best_action has invalid value")
        else:
            return best_move


    def act(self, obs, env):
        self.stats = {}  # create empty dictionary to save stats of all nodes

        self.env = env
        self.root = env.state.copy()        # set the root from the state of the game
        vectorized = env.observation_encoder.encode(
            self.root.observation(self.player_id))
        state = self.encode(vectorized)     # make it hashable
        self.update_visits(state)           # increment number of visits

        # random rollouts
        rollouts = 2
        for r in range(rollouts):
            depth = 0                       # reset the depth
            history = []                    # reset the history of the rollout
            game_state = self.root          # reset the state of the rollout

            while depth < self.max_depth and not game_state.is_terminal():

                # choose random action
                legal_moves = game_state.legal_moves()
                move = random.choice(legal_moves)

                game_state.apply_move(move)    # make a move

                # if it's a play or discard move, need to deal the cards
                if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                    game_state.deal_random_card()

                vectorized = env.observation_encoder.encode(
                    game_state.observation(self.player_id))
                state = self.encode(vectorized)     # make it hashable
                # increment number of visits
                self.update_visits(state)
                # append the state to the history of the rollout
                history.append(state)

                # set my belief
                self.my_belief = self.belief.encode(vectorized, self.player_id)

                # other players' moves
                for p in range(self.config['players'] - 1):

                    if game_state.is_terminal():
                        break

                    # choose random sample
                    vectorized = self.sampled_vector(vectorized)
                    obs_input = np.asarray(
                        vectorized, dtype=np.float32).reshape((1, -1))

                    # predict the move
                    move = self.select_from_prediction(obs_input, game_state)
                    game_state.apply_move(move)

                    hint = True
                    if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                        game_state.deal_random_card()
                        hint = False

                    vectorized = env.observation_encoder.encode(
                            game_state.observation(self.player_id))
                    state = self.encode(vectorized)
                    self.update_visits(state)
                    history.append(state)
                    
                    # if move is hint, calculate the score and terminate
                    if hint:
                        break

                # increase the depth by 1
                depth += 1

            # after the rollout has ended
            translated = state_translator(vectorized, self.config['players'])
            score = sum(translated.boardSpace)
            print(score)

            for i in range(len(history)):
                pass

        return random.choice(obs['legal_moves'])  # FIXME:
