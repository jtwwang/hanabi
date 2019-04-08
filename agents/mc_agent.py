# Artificial Intelligence Society at UC Davis
#
#   http://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from math import sqrt, log
from state_translate import state_translator
from rl_env import Agent
import numpy as np
import baysian_belief
import random
import pyhanabi
from policy_predictor import policy_net
from datetime import datetime
import time
random.seed(datetime.now())


class MCAgent(Agent):
    """Agent that uses monte carlo tree search to find the optimal move"""

    player_id = -1

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent"""
        self.config = config
        self.max_depth = 3
        self.belief = baysian_belief.Belief(config['players'])

        # load the predictor

        self.pp = policy_net(
            config['observation_size'], config['num_moves'], config['predictor'])
        self.pp.load()

        self.stats = {}         # stats of all states
        self.pred_moves = {}    # all predicted_moves

    def sample(self, card):
        """ sample a card from distribution"""
        rand = random.random()
        v = np.zeros(25, dtype=np.uint8)
        for (r, c), value in np.ndenumerate(card):
            if (rand - value <= 0):
                v[r * 5 + c] = 1
                return v
            else:
                rand -= value

    def sampled_vector(self, vectorized):
        """
        returns a sample of observations from belief and observations
        Args:
            vectorized (list): the observations in a vectorized form
        """
        v_sample = []
        translated_to_sample = state_translator(
            vectorized, self.config['players'])
        for i in range(translated_to_sample.handSize):
            v = self.sample(self.belief.my_belief[i]).tolist()
            v_sample = v_sample + v

            # update the card knowledge accordingly to the sample
            translated_to_sample.cardKnowledge[i*35:i*35 + 25] = v

            # update the belief accordingly to the new knowledge
            self.belief.calculate_prob(
                self.player_id, translated_to_sample.cardKnowledge)

        # take the original vector, and change the observations
        vectorized[:len(v_sample)] = v_sample
        return vectorized

    def encode(self, vectorized, move):
        """returns an hashable state from the observation"""
        return str(vectorized).join(str(move))

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
        """
        select the action of the agents given the observations
        Args:
            obs_input (list): the vectorized observations
            state     (HanabiState): the state of the game
        """
        nn_state = str(obs_input)
        if nn_state not in self.pred_moves.keys():
            prediction = self.pp.predict(obs_input)[0]
            # convert move to correct type
            best_value = -1
            best_move = -1
            for action in range(prediction.shape[0]):
                move = self.env.game.get_move(action)
                if state.move_is_legal(move):
                    if prediction[action] > best_value:
                        best_value = prediction[action]
                        best_move = move
            if best_move == -1:
                raise ValueError("best_action has invalid value")
            else:
                self.pred_moves[nn_state] = best_move
                return best_move
        else:
            # if already visited this exact state, use the predicted move
            return self.pred_moves[nn_state]

    def UCT(self, one_stat, N):
        """
        Calculate the Upper-Confidence bound for trees.
        Uses the formula retrieved at https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        under the section "Exploration and exploitation"
        Args:
            one_stat (dict): a dictionary with keys 'value' and 'visits'
            N (int): the number of visits of the parent node
        Returns:
            the UCT value (double)
        """
        v = one_stat['value']
        n = one_stat['visits']
        return (v / n) + 1.41 * sqrt(log(N) / n)

    def choose_move(self, vectorized, game_state, previous_state):
        """
        Chooses the best legal move accordingly to the UCT value of the node
        Args:
            vectorized (list): the list of observations in vectorized form
            game_state (HanabiState): the current state of the game
            previous_state (string): the key to the parent state
        Returns:
            move (HanabiMove): the best move
        """
        N = self.stats[previous_state]['visits']
        max_uct = -1
        best_move = -1
        for move in game_state.legal_moves():
            state = self.encode(vectorized, move)
            if state in self.stats.keys():
                # calculate the uct
                uct_value = self.UCT(self.stats[state], N)
                if max_uct < uct_value:
                    max_uct = uct_value
                    best_move = move
            else:
                # the state has not been explored so choose that
                return move
        if best_move == -1:
            raise ValueError("invalid best_move")
        else:
            return best_move

    def act(self, obs, env):

        start = time.time()
        self.pred_moves = {}  # reset memory
        self.stats = {}      # reset memory

        self.env = env
        self.root = env.state        # set the root from the state of the game

        # random rollouts
        rollouts = 1000
        for r in range(rollouts):
            depth = 0                       # reset the depth
            history = []                    # reset the history of the rollout
            game_state = self.root.copy()          # reset the state of the rollout
            vectorized = env.observation_encoder.encode(
                game_state.observation(self.player_id))
            state = "root"
            self.update_visits(state)

            while depth < self.max_depth:
                if game_state.is_terminal():
                    break

                # choose random action
                move = self.choose_move(vectorized, game_state, state)

                state = self.encode(vectorized, move)   # make it hashable
                self.update_visits(state)               # increment visits
                # append the state to the history of the rollout
                history.append(state)

                game_state.apply_move(move)    # make a move

                # if it's a play or discard move, need to deal the cards
                if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                    game_state.deal_random_card()

                vectorized = env.observation_encoder.encode(
                    game_state.observation(self.player_id))

                # set my belief
                self.belief.encode(vectorized, self.player_id)

                # hint of the other players
                hint = True

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

                    state = self.encode(vectorized, move)   # make it hashable
                    self.update_visits(state)               # increment visits
                    # append the state to the history of the rollouts
                    history.append(state)

                    game_state.apply_move(move)  # make move

                    hint = True
                    if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                        game_state.deal_random_card()
                        hint = False

                    vectorized = env.observation_encoder.encode(
                        game_state.observation(self.player_id))

                    # if move is hint, calculate the score and terminate
                    if hint:
                        break

                if hint:
                    break

                # increase the depth by 1
                depth += 1

            # after the rollout has ended
            if not game_state.is_terminal():
                translated = state_translator(
                    vectorized, self.config['players'])
                score = sum(translated.boardSpace)
            else:
                # if the game is terminal
                score = game_state.score()

            for s in history:
                self.stats[s]['value'] += score

        values = []
        n = []
        vectorized = env.observation_encoder.encode(
            self.root.observation(self.player_id))
        legal_moves = self.root.copy().legal_moves()
        for move in legal_moves:
            state = self.encode(vectorized, move)     # make it hashable
            value = float(self.stats[state]['value'])
            visits = float(self.stats[state]['visits'])
            values.append(value/visits)
            n.append(visits)

        best = values.index(max(values))

        end = time.time()
        if self.verbose:
            print("time of execution: %.3f" % (end-start))
            for i in range(len(values)):
                print("%s: %.2f, visits %d" %
                      (legal_moves[i], values[i], n[i]))

        return obs['legal_moves'][best]
