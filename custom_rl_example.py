# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
"""A script to collect episodes from simple agent"""

from __future__ import print_function

import sys
import getopt
import rl_env
import experience as exp
import numpy as np
import pyhanabi
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.rainbow_agent_rl import RainbowAgent

AGENT_CLASSES = {
        'SimpleAgent':  SimpleAgent,
        'RandomAgent':  RandomAgent,
        'RainbowAgent': RainbowAgent}

class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.env = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_config = {
                'players': flags['players'],
                'num_moves' : self.env.num_moves(),
                'observation_size': self.env.vectorized_observation_shape()[0]}
        self.agent_class = AGENT_CLASSES[flags['agent_class']]

    def moves_lookup(self,move, ob):
        """returns the int given the dictionary form"""

        int_obs = ob['legal_moves_as_int']
        
        for i in int_obs:
            isEqual = True
            dict_move = self.env.game.get_move(i).to_dict()

            for d in dict_move.keys():
                if d not in move.keys():
                    isEqual = False
                    break
                elif not move[d] == dict_move[d]:
                    isEqual = False
                    break
            
            if isEqual:
                return i
        
        print("ERROR")
        return -1

    def run(self):

        global replay

        rewards = []
        
        if self.flags['agent_class'] == 'RainbowAgent':
            # put 2-5 copies of the same agent in a list, because loading
            # the same tensorflow checkpoint more than once in a session fails
            agent = self.agent_class(self.agent_config)
            agents = [agent for _ in range(self.flags['players'])]
        else:
            agents = [self.agent_class(self.agent_config)
                    for _ in range(self.flags['players'])]
        
        for eps in range(flags['num_episodes']):
            print('Running episode: %d' % eps)

            obs = self.env.reset()  # Observation of all players
            done = False
            eps_reward = 0

            while not done:
                for agent_id, agent in enumerate(agents):
                    ob = obs['player_observations'][agent_id]
                    action = agent.act(ob)

                    move = self.moves_lookup(action, ob)

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)

                    # add the move to the memory
                    replay.add(ob, reward, move)

                    eps_reward += reward

                    if done:
                        break
            rewards.append(eps_reward)

        print('Max Reward: %.3f' % max(rewards))


if __name__ == "__main__":
    
    flags = {'players': 2,
            'num_episodes': 1000,
            'agent_class': 'SimpleAgent',
            'debug': False}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class=',
                                      'debug='])
    if arguments:
        sys.exit('usage: customAgent.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    # initialize the replay memory
    replay = exp.Experience(flags['players'])

    # run the episodes
    runner = Runner(flags)
    runner.run()

    # save the memory to file
    replay.save()
