# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
from agents.second_agent import SecondAgent
from agents.mc_agent import MCAgent
from agents.nn_agent import NNAgent
from agents.rainbow_agent_rl import RainbowAgent
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from agents.neuroEvo_agent import NeuroEvoAgent
from agents.newest_card_agent import NewestCardAgent
import experience as exp
import rl_env
import getopt

import os
import sys
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


AGENT_CLASSES = {
    'SimpleAgent':  SimpleAgent,
    'NewestCardAgent': NewestCardAgent,
    'SecondAgent': SecondAgent,
    'RandomAgent':  RandomAgent,
    'RainbowAgent': RainbowAgent,
    'MCAgent': MCAgent,
    'NNAgent': NNAgent,
    'NeuroEvoAgent': NeuroEvoAgent}

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
        self.agent_class = AGENT_CLASSES[flags['agent_class']]
        if flags['agent2'] != "":
            self.agent_2_config = {
                'players': flags['players'],
                'num_moves': self.env.num_moves(),
                'observation_size': self.env.vectorized_observation_shape()[0],
                'checkpoint_dir': flags['checkpoint_dir2']}
            self.agent_2_class = AGENT_CLASSES[flags['agent2']]
        else:
            self.agent_2_class = AGENT_CLASSES[flags['agent_class']]
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

        global replay

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
        print('Average Reward: %.3f' % (sum(rewards)/n_eps))
        print('Average steps: %.2f' % (avg_steps/float(n_eps)))


if __name__ == "__main__":

    flags = {'players': 2,
             'num_episodes': 1000,
             'agent_class': 'NewestCardAgent',
             'debug': False,
             'agent_predicted': "",
             'model_class': "",
             'model_name': "predictor.h5",
             'agent2': "",
             'checkpoint_dir':"",
             'checkpoint_dir2':""}
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
                                        'checkpoint_dir2='])
    if arguments:
        sys.exit('usage: customAgent.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    # initialize the replay memory
    if flags['agent2'] == "":
        nameDir = flags['agent_class']
    else:
        nameDir = flags['agent_class'] + flags['agent2']

    replay = exp.Experience(nameDir, numAgents=flags['players'])    

    # run the episodes
    runner = Runner(flags)
    runner.run()

    # save the memory to file
    replay.save()
