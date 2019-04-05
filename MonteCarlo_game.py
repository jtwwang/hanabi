# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
from state_translate import state_translator
from agents.mc_agent import MCAgent
from agents.rainbow_agent_rl import RainbowAgent
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
import pyhanabi
import numpy as np
import experience as exp
import rl_env
import getopt
"""A script to collect episodes from simple agent"""

import os
import sys
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


AGENT_CLASSES = {
    'SimpleAgent':  SimpleAgent,
    'RandomAgent':  RandomAgent,
    'RainbowAgent': RainbowAgent,
    'MCAgent': MCAgent}


class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.env = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_config = {
            'players': flags['players'],
            'num_moves': self.env.num_moves(),
            'observation_size': self.env.vectorized_observation_shape()[0]}
        self.agent_class = AGENT_CLASSES[flags['agent_class']]

    def moves_lookup(self, move, ob):
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
            agents = [agent for _ in range(self.flags['players'] - 1)]
        else:
            agents = [self.agent_class(self.agent_config)
                      for _ in range(self.flags['players'])]
            
        # set the name
        for agent in agents:
            agent.name = self.flags['agent_class']

        # configure the last agent to be the Monte Carlo Agent
        self.agent_class = MCAgent
        agents[-1] = self.agent_class(self.agent_config)
        agents[-1].player_id = len(agents) - 1
        agents[-1].verbose = flags['verbose']
        agents[-1].name = 'MonteCarloAgent'
        avg_steps = 0

        for eps in range(flags['num_episodes']):
            print('Running episode: %d' % eps)

            obs = self.env.reset()  # Observation of all players
            done = False
            eps_reward = 0
            n_steps = 0

            while not done:
                for agent_id, agent in enumerate(agents):
                    ob = obs['player_observations'][agent_id]

                    if agent_id == len(agents) - 1:
                        # if is Monte Carlo Tree search, pass also the env
                        action = agent.act(ob, self.env)
                    else:
                        # otherwise, pass the observations only
                        action = agent.act(ob)

                    move = self.moves_lookup(action, ob)
                    n_steps += 1
                    print("Agent %s made move %s" % (agent.name, action))

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)
                    eps_reward += reward

                    if done:
                        break
            rewards.append(eps_reward)
            avg_steps += n_steps

        n_eps = float(flags['num_episodes'])
        print('Average Reward: %.3f' % (float(sum(rewards))/n_eps))
        print('Average steps: %.3f' % (float(avg_steps)/n_eps))


if __name__ == "__main__":

    flags = {'players': 2,
             'num_episodes': 1000,
             'agent_class': 'SimpleAgent',
             'debug': False,
             'verbose': False}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class=',
                                        'debug=',
                                        'verbose='])
    if arguments:
        sys.exit('usage: customAgent.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    # run the episodes
    runner = Runner(flags)
    runner.run()
