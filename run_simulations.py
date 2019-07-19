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
                
        # create configurations
        self.agent_config, self.agent_2_config = self.generate_config(flags)

        # use configurations to create agent
        self.agent = load_agent(flags['agent_class'])(self.agent_config)
        
        if flags['agent2_class'] != flags['agent_class']:
           # use configurations to create second agent
           self.agent2 = load_agent(flags['agent2_class'])(self.agent_2_config)
        
    def set_agents(self, agent1, agent2):
        self.agent = agent1
        self.agent2 = agent2

    def generate_config(self, flags):
        general_config = {
                'players': flags['players'],
                'num_moves': self.env.num_moves(),
                'observation_size': self.env.vectorized_observation_shape()[0],
                'debug': flags['debug']}
        
        # config first agent
        agent_config = {
            'agent_predicted': flags['agent_predicted'],
            'model_class': flags['model_class'],
            'model_name': flags['model_name'],
            'checkpoint_dir': flags['checkpoint_dir']}
        agent_config.update(general_config) # merge the two dictionaries

        # config second agent
        agent_2_config = {
                'agent_predicted': flags['agent_predicted2'],
                'model_class': flags['model_class2'],
                'model_name': flags['model_name2'],
                'checkpoint_dir': flags['checkpoint_dir2']}
        agent_2_config.update(general_config) # merge the two dictionaries

        return agent_config, agent_2_config

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

    def generate_pool(self):
        """
        Function to generate the pool of agents given one or two agents
        and specified the number of players (size of the pool)
        """

        if self.flags['agent_class'] == self.flags['agent2_class']:
            agents = [self.agent for _ in range(self.flags['players'])]
        else:
            agents = [self.agent2 for _ in range(self.flags['players'] - 1)]
            agents = [self.agent] + agents

        # one more thing for the MC agent
        if self.flags['agent_class'] == 'MCAgent':
            agents[0].player_id = 0
        if self.flags['agent2_class'] == 'MCAgent':
            for i in range(1, self.len(agents)):
                agents[i].player_id = i

        return agents

    def run(self):
        
        agents = self.generate_pool()

        avg_steps = 0
        avg_reward = 0

        for eps in range(flags['num_episodes']):

            if eps % 10 == 0:
                print('Running episode: %d' % eps)

            obs = self.env.reset()  # Observation of all players
            done = False

            while not done:
                for agent_id, agent in enumerate(agents):
                    ob = obs['player_observations'][agent_id]
                    if type(agent) == MCAgent:
                        action = agent.act(ob, self.env)
                    else:
                        action = agent.act(ob)

                    move = self.moves_lookup(action, ob)
                    avg_steps += 1

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)

                    # add the move to the memory
                    replay.add(ob, reward, move, eps)

                    avg_reward += reward

                    if done:
                        break

        n_eps = float(flags['num_episodes'])
        avg_reward /= n_eps
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

    runner = Runner(flags)

    AgentClassList = [
            'NewestCardAgent',
            'RandomAgent',
            'SimpleAgent',
            'SecondAgent',
            'ProbabilisticAgent',
            'RainbowAgent']

    # create the agents
    AgentDict = {}
    for a_class in AgentClassList:
        flags['agent_class'] = a_class
        config1, config2 = runner.generate_config(flags)
        AgentDict[a_class] = load_agent(a_class)(config1)

    results = []
    for agent in AgentClassList:
        for agent2 in AgentClassList:
            flags['agent_class'] = agent
            flags['agent2_class'] = agent2

            runner.set_agents(AgentDict[agent], AgentDict[agent2])
            avg_score, _ = runner.run()
            results.append(avg_score)

    results = np.asarray(results)
    results = np.reshape(results, (len(AgentClassList), len(AgentClassList)))

    fig, ax = plt.subplots()
    im = ax.imshow(results)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("score", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(AgentClassList)))
    ax.set_yticks(np.arange(len(AgentClassList)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(AgentClassList)
    ax.set_yticklabels(AgentClassList)

    # Loop over data dimensions and create text annotations.	
    for i in range(len(AgentClassList)):	
        for j in range(len(AgentClassList)):	
            text = ax.text(j, i, results[i, j],	
                       ha="center", va="center", color="w")

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
             'agent2_class': "",
             'debug': False,
             'agent_predicted': "",
             'agent_predicted2': "",
             'model_class': "",
             'model_class2': "",
             'model_name': "predictor.h5",
             'model_name2': "predictor.h5",
             'checkpoint_dir':"",
             'checkpoint_dir2':"",
             'cross_play': False}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class=',
                                        'agent2_class=',
                                        'debug=',
                                        'agent_predicted=',
                                        'agent_predicted2=',
                                        'model_class=',
                                        'model_class2=',
                                        'model_name=',
                                        'model_name2=',
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
    '--agent2_class <str>            to play \'ad hoc\' against another agent.\n' 
    '--checkpoint_dir <str>    path to the checkpoints for RainbowAgent.\n'
    '--checkpoint_dir2 <str>   path to the checkpoints for RainbowAgent as agent2.\n'
    '--cross_play <True/False> cross_play between all agents.\n')
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

        # initialize the replay memory
    if flags['agent2_class'] == "":
        nameDir = flags['agent_class']
    else:
        nameDir = flags['agent_class'] + flags['agent2_class']

    global replay
    replay = exp.Experience(nameDir, numAgents=flags['players'])    

    # create the agents
    if flags['agent2_class'] == "":
        flags['agent2_class'] = flags['agent_class']
        flags['model_name2'] = flags['model_name']
        flags['checkpoint_dir2'] = flags['checkpoint_dir']
        flags['model_class2'] = flags['model_class']
        flags['agent_predicted2'] = flags['agent_predicted']

    if flags['cross_play']:
        cross_play(flags)
    else:
                
        # run the episodes
        runner = Runner(flags)
        runner.run()
        
        # save the memory to file
        replay.save()
