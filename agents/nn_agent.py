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

"""Agent that runs a neural network to predict the action."""

from rl_env import Agent
from predictors.policy_pred import policy_pred
from predictors.conv_pred import conv_pred
from predictors.dense_pred import dense_pred
from predictors.lstm_pred import lstm_pred
import numpy as np

from os import path

model_dict = {
    "conv": conv_pred,
    "dense": dense_pred,
    "lstm": lstm_pred
}


class NNAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config

        if config['model_class'] in model_dict.keys():
            model_class = model_dict[config['model_class']]
            self.pp = model_class(config['agent_predicted'])
        else:
            raise ValueError("model type %s not recognized" %
                             config['model_class'])
        if 'relative_path' in config.keys():
            self.pp.path = path.join(config['relative_path'], self.pp.path)

        self.pp.load(config['model_name'])

    def act(self, ob):
        vec = np.asarray([ob['vectorized']])
        prediction = - self.pp.predict(vec)

        # from prediciton select the best move
        moves = np.argsort(prediction.flatten())
        legal_moves = ob['legal_moves']
        indices_lm = ob['legal_moves_as_int']
        action = -1
        for m in moves:
            if m in indices_lm:
                action = legal_moves[indices_lm.index(m)]
                break
        if action == -1:
            raise ValueError("action is incorrect")
        else:
            return action

    def begin_episode(self, current_player, legal_actions, observation):

        vec = np.asarray([observation])

        prediction = self.pp.predict(vec).flatten()
        legal_indices = np.where(legal_actions == 0.0)[0]

        best_value = np.amax(prediction[legal_indices])
        action = np.where(prediction == best_value)[0]

        assert legal_actions[action] == 0.0
        
        return action

    def step(self, reward, current_player, legal_actions, observation):
        return self.begin_episode(current_player, legal_actions, observation)

    def end_episode(self, final_rewards):
        pass
