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

from agents.load_agents import load_agent
from agents.mc_agent import MCAgent
import data_pipeline.experience as exp
import rl_env
import getopt

import os
import sys
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.env = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_config = {
            'players': flags['players'],
            'num_moves': self.env.num_moves(),
            'observation_size': self.env.vectorized_observation_shape()[0],
            'agent_predicted': flags['agent_predicted'],
            'model_class': flags['model_class'],
            'model_name': flags['model_name'],
            'debug': flags['debug'],
            'checkpoint_dir': flags['checkpoint_dir']}
        self.agent_class = load_agent(flags['agent_class'])
        if flags['agent2'] != "":
            self.agent_2_config = {
                'players': flags['players'],
                'num_moves': self.env.num_moves(),
                'observation_size': self.env.vectorized_observation_shape()[0],
                'checkpoint_dir': flags['checkpoint_dir2']}
            self.agent_2_class = load_agent(flags['agent2'])
        else:
            self.agent_2_class = load_agent(flags['agent_class'])
            self.agent_2_config = self.agent_config # create copy
        
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

        rewards = []

        if self.agent_2_class == self.agent_class:
            agent = self.agent_class(self.agent_config)
            agents = [agent for _ in range(self.flags['players'])]
        elif self.flags['agent_class'] == 'NeuroEvoAgent':
            self.agent_config['model_name'] = self.flags['model_name']
            agent = self.agent_class(self.agent_config)
            agents = [agent for _ in range(self.flags['players'])]
        else:
            agent = self.agent_2_class(self.agent_2_config)
            agents = [agent for _ in range(self.flags['players'] - 1)]
            agents = [self.agent_class(self.agent_config)] + agents

        # one more thing for the MC agent
        if self.flags['agent_class'] == 'MCAgent':
            agents[0].player_id = 0
        if self.flags['agent2'] == 'MCAgent':
            for i in range(1, self.len(agents)):
                agents[i].player_id = i

        avg_steps = 0

        for eps in range(flags['num_episodes']):

            if eps % 10 == 0:
                print('Running episode: %d' % eps)

            obs = self.env.reset()  # Observation of all players
            done = False
            eps_reward = 0
            n_steps = 0

            while not done:
                for agent_id, agent in enumerate(agents):
                    ob = obs['player_observations'][agent_id]
                    if type(agent) == MCAgent:
                        action = agent.act(ob, self.env)
                    else:
                        action = agent.act(ob)

                    move = self.moves_lookup(action, ob)
                    n_steps += 1

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)

                    # add the move to the memory
                    replay.add(ob, reward, move, eps)

                    eps_reward += reward

                    if done:
                        break
            rewards.append(eps_reward)
            avg_steps += n_steps

        n_eps = float(flags['num_episodes'])
        avg_reward = sum(rewards)/n_eps
        avg_steps /= n_eps
        print('Average Reward: %.3f' % avg_reward)
        print('Average steps: %.2f' % avg_steps)

        return avg_reward, avg_steps

def cross_play(flags):

    import numpy as np
    import matplotlib.pyplot as plt


    """
    Function to play the cross_play between all agents
    """
    AgentList = [
            'NewestCardAgent',
            'RandomAgent',
            'SimpleAgent',
            'SecondAgent',
            'ProbabilisticAgent']

    results = []
    for agent in AgentList:
        for agent2 in AgentList:
            flags['agent_class'] = agent
            flags['agent2'] = agent2
            flags['num_episodes'] = 1000

            runner = Runner(flags)
            avg_score, _ = runner.run()
            results.append(avg_score)

    results = np.asarray(results)
    results = np.reshape(results, (len(AgentList), len(AgentList)))

    fig, ax = plt.subplots()
    im = ax.imshow(results)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("score", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(AgentList)))
    ax.set_yticks(np.arange(len(AgentList)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(AgentList)
    ax.set_yticklabels(AgentList)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(results.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(results.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Cross Play")

    plt.show()


if __name__ == "__main__":

    flags = {'players': 2,
             'num_episodes': 1000,
             'agent_class': 'SimpleAgent',
             'debug': False,
             'agent_predicted': "",
             'model_class': "",
             'model_name': "predictor.h5",
             'agent2': "",
             'checkpoint_dir':"",
             'checkpoint_dir2':"",
             'cross_play': False}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class=',
                                        'debug=',
                                        'agent_predicted=',
                                        'model_class=',
                                        'model_name=',
                                        'agent2=',
                                        'checkpoint_dir=',
                                        'checkpoint_dir2=',
                                        'cross_play='])
    if arguments:
        sys.exit(
    'usage: customAgent.py [options]\n'
    '--players                 number of players in the game.\n'
    '--num_episodes            number of game episodes to run.\n'
    '--agent_class             the agent that you want to use.\n'
    '--agent_predicted <str>   necessary if using MCAgent or NNAgent. Use one of the other classes as string.\n'
    '--model_class <str>       network type ["dense", "conv", "lstm"].\n'
    '--model_name <str>        model name of a pre-trained model.\n'
    '--agent2 <str>            to play \'ad hoc\' against another agent.\n' 
    '--checkpoint_dir <str>    path to the checkpoints for RainbowAgent.\n'
    '--checkpoint_dir2 <str>   path to the checkpoints for RainbowAgent as agent2.\n'
    '--cross_play <True/False> cross_play between all agents.\n')
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

        # initialize the replay memory
    if flags['agent2'] == "":
        nameDir = flags['agent_class']
    else:
        nameDir = flags['agent_class'] + flags['agent2']

    global replay
    replay = exp.Experience(nameDir, numAgents=flags['players'])    

    if flags['cross_play']:
        cross_play(flags)
    else:  
        # run the episodes
        runner = Runner(flags)
        runner.run()
        
        # save the memory to file
        replay.save()
