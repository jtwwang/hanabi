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
from conv_policy_pred import policy_net
import numpy as np


class NeuroEvoAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        modelname = config['model_name']
        self.pp = policy_net(658, 20, 'NeuroEvo', modelname)
        self.pp.load()

    def act(self, ob):
        vec = np.asarray([ob['vectorized']])
        prediction = self.pp.predict(vec)

        # from prediciton select the best move
        moves = np.argsort(prediction.flatten())
        legal_moves = ob['legal_moves']
        indeces_lm = ob['legal_moves_as_int']
        action = -1
        for m in moves:
            if m in indices_lm:
                action = legal_moves[indices_lm.index(m)]
                break
        if action == -1:
            raise ValueError("action is incorrect")
        else:
            return action
