# Artificial Intelligence Society at UC Davis
#
#   http://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from math import sqrt, log
from data_pipeline.state_translate import state_translator
from rl_env import Agent
import numpy as np
from data_pipeline import bayes
import random
import pyhanabi
from predictors.conv_pred import conv_pred
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
        self.belief = bayes.Belief(config['players'])

        # load the predictor

        self.pp = conv_pred(
            config['agent_predicted'])
        self.pp.load(config['model_name'])

        self.stats = {}         # stats of all states

    def sample(self, belief):
        """
        Ranodmly samples a card from distribution
        Args:
            belief (np.array, type np.float32): a probability distribution
                of size 25 with sum 1
        returns:
            v: (np.array, type np.uint8): a one-hot encoded card
        """
        rand = random.random()
        v = np.zeros(25, dtype=np.uint8)
        for x, value in np.ndenumerate(belief):
            if (rand - value <= 0):
                v[x] = 1
                return v
            else:
                rand -= value

    def sampled_vector(self, vec, player):
        """
        returns a sample of observations from belief and observations
        Args:
            vec (list): the observations in a vectorized form
            player(int): the current player
        """

        hand = []
        vec2sample = state_translator(vec, self.config['players'])
        ix = ((self.player_id - player) %
              self.config['players']) * 35 * vec2sample.handSize

        for i in range(vec2sample.handSize):
            # calculate the probability distribution
            my_belief = self.belief.prob(
                vec2sample.cardKnowledge[ix + i * 35: ix + i * 35 + 25])

            # sample a single card
            card = self.sample(my_belief)

            self.belief.add_known(card)  # add the knowledge of the card
            hand += card.tolist()  # add the sampled card to the final vector

        # take the original vector, and change the observations
        vec[:len(hand)] = hand
        return vec

    def encode(self, vec, move):
        """
        Returns a hashable string of the state-action pair
        Args:
            vec (list): a vector that contains the observation of
                the game state before taking a move
            move (HanabiMove): the taken legal move
        Returns:
            a string which is hashable and represents the state-action
            pair, thus uniquely representing the general future state
            without explicitely defining it.
        """
        return str(vec).join(str(move))

    def update_visits(self, state):
        """
        This function updates the visits statics for a certain state
        Args:
            state (string) the encoded representation of a string
        """
        if state not in self.stats.keys():       # update_node
            self.stats[state] = {'visits': 1, 'value': 0}
        else:
            self.stats[state]['visits'] += 1

    def select_from_prediction(self, vec, game_state):
        """
        select the action of the agents given the observations
        Args:
            obs_input       (list): the vectorized observations
            game_state      (HanabiState): the state of the game
        """
        obs_input = np.asarray(
            vec, dtype=np.float32).reshape((1, -1))

        # reshapre prediction to appropiate size
        # prediction is the result of neural network
        # == to a vector of 20 possible actions, and we take the one with the highest probability
        prediction = self.pp.predict(np.reshape(obs_input, (1, -1, 1)))[0]
        
        best_value = -1
        best_move = -1
        for action in range(prediction.shape[0]):
            move = self.env.game.get_move(action)
            if game_state.move_is_legal(move):
                if prediction[action] > best_value:
                    best_value = prediction[action]
                    best_move = move
        if best_move == -1:
            raise ValueError("best_action has invalid value")
        else:
            return best_move

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

    def choose_move(self, vec, game_state, previous_state):
        """
        Chooses the best legal move accordingly to the UCT value of the node
        Args:
            vec (list): the list of observations in vectorized form
            game_state (HanabiState): the current state of the game
            previous_state (string): the key to the parent state
        Returns:
            move (HanabiMove): the best move
        """
        N = self.stats[previous_state]['visits']
        max_uct = -1
        best_move = -1
        for move in game_state.legal_moves():
            state = self.encode(vec, move)
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
        """
        choosing which action we are taking
        """

        start = time.time()         # start the timer
        self.stats.clear()          # reset memory

        self.env = env              # define the environment as object variable
        self.root = env.state       # set the root from the state of the game

        # random rollouts
        rollouts = 1000
        for r in range(rollouts):
            depth = 0                       # reset the depth
            history = []                    # reset the history of the rollout
            game_state = self.root.copy()          # reset the state of the rollout
            vec = env.observation_encoder.encode(
                game_state.observation(self.player_id)) # vectorized version of 658; depends on number of players (essentially just observations)
            state = "root"
            self.update_visits(state)

            while depth < self.max_depth:
                if game_state.is_terminal():
                    break

                # choose random action
                move = self.choose_move(vec, game_state, state)

                state = self.encode(vec, move)   # make it hashable
                self.update_visits(state)               # increment visits
                # append the state to the history of the rollout
                history.append(state)

                game_state.apply_move(move)    # make a move

                # if it's a play or discard move, need to deal the cards
                if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                    game_state.deal_random_card()
                
                #these are our own observations
                vec = env.observation_encoder.encode(
                    game_state.observation(self.player_id))
                # observation of next player; however, we want to eliminate the use of the "other" observations
                other_vec = env.observation_encoder.encode(
                    game_state.observation(game_state.cur_player()))

                # set my belief; creating the probability distribution from the observation that we have (belief encode) 
                self.belief.encode(vec)

                # hint of the other players
                hint = True

                """ **** OTHER PLAYERS ****
                Here we sample a state from a distribution of possible states,
                and we predict what action the other players will take from said
                state. We simulate the action of the other players and we obtain
                another state which can be used to explore the next move our
                agent can take.
                """
                for p in range(self.config['players'] - 1):

                    if game_state.is_terminal():
                        break

                    # sample from random distribution
                    vec = self.sampled_vector(
                        other_vec, game_state.cur_player())

                    # predict the move
                    move = self.select_from_prediction(vec, game_state)

                    state = self.encode(vec, move)   # make it hashable
                    self.update_visits(state)               # increment visits
                    # append the state to the history of the rollouts
                    history.append(state)

                    game_state.apply_move(move)  # make move

                    hint = True
                    if game_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                        game_state.deal_random_card()
                        hint = False

                    vec = env.observation_encoder.encode(
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
                    vec, self.config['players'])
                score = sum(translated.boardSpace)
            else:
                # if the game is terminal
                score = game_state.score()

            for s in history:
                self.stats[s]['value'] += score

        """ **** STATISTICS ACTIONS ****
        Here we are computing which child of the root node has
        the best average value, and we will return the action
        that correspond to said node.

        If the 'verbose' option is selected, we will print the
        statistics of number of visits and average value for
        each of the child of the root node.
        """
        # initialize matrices for statistics of child nodes
        values, n = [],[]

        # recalculate the vector from root node
        vec = env.observation_encoder.encode(
            self.root.observation(self.player_id))
        legal_moves = self.root.copy().legal_moves()

        # calculate the average for each child node of the root
        for move in legal_moves:
            state = self.encode(vec, move)     # make it hashable
            value = float(self.stats[state]['value'])
            visits = float(self.stats[state]['visits'])
            values.append(value / visits)
            n.append(visits)

        # index of the best node
        best = values.index(max(values))

        end = time.time()
        if self.config['debug']:
            print("time of execution: %.3f" % (end - start))
            for i in range(len(values)):
                print("%s: %.2f, visits %d" %
                      (legal_moves[i], values[i], n[i]))

        return obs['legal_moves'][best]
