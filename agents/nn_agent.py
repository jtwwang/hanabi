# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Random Agent."""

from rl_env import Agent
from predictors.policy_pred import policy_pred
from predictors.conv_pred import conv_pred
from predictors.dense_pred import dense_pred
from predictors.lstm_pred import lstm_pred
import numpy as np

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
            self.pp = model_class(config['agent_predicted'],
                                  model_name=config['model_name'])
        else:
            raise ValueError("model type %s not recognized" %
                             config['model_class'])
        self.pp.load()

    def act(self, ob):
        vec = np.asarray([ob['vectorized']])
        prediction = self.pp.predict(vec)

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
