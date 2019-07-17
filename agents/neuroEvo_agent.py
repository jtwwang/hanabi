# Developed by Lorenzo Mambretti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeuroEvolutionAgent."""

from rl_env import Agent
from predictors.conv_pred import conv_pred
import numpy as np


class NeuroEvoAgent(Agent):
    """Agent that uses a convolutional neural network trained with a genetic algorithm"""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        modelname = config['model_name']
        self.pp = conv_pred('NeuroEvo_agent')

        if 'initialize' not in config.keys():
            config['initialize'] = False

        self.pp.load(model_name = config['model_name'])
        if self.pp.model == None or config['initialize']:
            self.pp.define_model_dim(config['observation_size'], config['num_moves'])
            self.pp.create_model()

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

    def save(self, model_name = "predicto.h5r"):
        self.pp.save(model_name = model_name)
