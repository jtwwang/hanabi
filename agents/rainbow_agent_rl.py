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
"""Rainbow Agent."""

from rl_env import Agent

import sys
sys.path.append("./rainbow/")
import rainbow_agent

class RainbowAgent(Agent):
  """Agent that loads and applies a pretrained rainbow model."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    self.agent = rainbow_agent.RainbowAgent(
        observation_size=1,
        num_actions=self.config['num_moves'],
        num_players=self.config['players'])
    self.agent.eval_mode = True
    
  def _parse_legal_moves(self, observation):
    current_player = observations['current_player']
    current_player_observation = (observations['player_observations'][current_player])
    legal_moves = current_player_observation['legal_moves_as_int']
    legal_moves = self.agent.format_legal_moves(legal_moves, self.config['num_moves'])
    
    return legal_moves

  def act(self, observation):
    """Act based on an observation."""
    
    # FIXME: unclear if below two lines needed
    if observation['current_player_offset'] != 0:
      return None
    
    legal_moves = _parse_legal_moves(observation)
    action = self.agent._select_action(observation, legal_moves)
    
    # FIXME: format action to return
    
    return action

    
