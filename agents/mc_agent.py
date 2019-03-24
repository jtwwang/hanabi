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

class MCAgent(Agent):
    """Agent that uses monte carlo tree search to find the optimal move"""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent"""
        self.config = config
        self.max_depth = 2
        self.belief = baysian_belief.Belief(config['players'])

    def act(self, obs):
        self.belief.encode(obs)
        return random.choice(obs['legal_moves'])

