# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied


from state_transition import state_tr
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


class PMCTS_Agent(Agent):
    """Agent that uses monte carlo tree search to find the optimal move"""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent"""
        self.config = config
        self.max_depth = 4
        self.rollouts = 1000
        self.belief = bayes.Belief(config['players'])

        # load the predictor

        self.pp = conv_pred(
            config['agent_predicted'])
        self.pp.load(config['model_name'])

        self.stats = {}         # stats of all states

    def choose_move(self, state):
        return 1

    def update_visits(self, state, vec):
        """
        This function updates the visits statics for a certain state
        args:
            state (string) the encoded representation of a string
        """
        if state not in self.stats.keys():
            self.stats[state] = {'visits': 1,
                                 'value': 0,
                                 'vec': vec} # create node
        else:
            self.stats[state]['visits'] += 1

    @staticmethod
    def encode(state, move):
        """
        Function to update the state string
        args:
            state (string) the encoded representation of a state
            move (int) the last move done
        """
        state += chr(move + 65)
        return state

    def act(self, ob):

        vec = ob['vectorized']
        self.stats.clear() # clear the dictionary

        for r in range(self.rollouts):
            depth = 0
            history = []
            state = "root"

            while depth < self.max_depth:

                self.update_visits(state, vec) # update stats
                move = self.choose_move(state) # choose move
                vec = state_tr(vec, move, self.config['players']) # take step
                state = PMCTS_Agent.encode(state, move) # encode state
                history.append(state) # append state in history
