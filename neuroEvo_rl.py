# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
import experience as exp
import rl_env
import getopt
import numpy as np
import os
import sys
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from conv_policy_pred import policy_net
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))

def model_crossover(model_idx1, model_idx2, modelname):
    global current_pool
    weights1 = current_pool[model_idx1].model.get_weights()
    weights2 = current_pool[model_idx2].model.get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    weightsnew1[0] = weights2[0]
    weightsnew2[0] = weights1[0]
    pp = policy_net(658, 20, 'NeuroEvo', modelname)
    pp.model.set_weights(weightsnew1)
    return pp

def model_mutate(ix):
    global current_pool
    weights = current_pool[ix].model.get_weights()
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0,1) > 0.9:
                change = random.uniform(-0.1, 0.1)
                weights[xi][yi] += change
    current_pool[ix].model.set_weights(weights)

class Runner(object):
    """Runner class."""

    def __init__(self, flags, agent):
        """Initialize runner."""
        self.flags = flags
        self.env = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_config = {
            'players': flags['players'],
            'num_moves': self.env.num_moves(),
            'observation_size': self.env.vectorized_observation_shape()[0]}
        self.agent = agent

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

        agent = self.agent
        avg_steps = 0

        for eps in range(flags['num_episodes']):

            obs = self.env.reset()  # Observation of all players
            done = False
            eps_reward = 0
            n_steps = 0

            agent_id = 0

            while not done:
                ob = obs['player_observations'][agent_id]
                vec = ob['vectorized']
                vec = np.reshape(vec, (1,658,1))
                prediction = agent.predict(vec)

                # from prediciton select the best move
                moves = np.argsort(prediction)[0]
                #print(moves)
                #print(prediction)
                legal_moves = ob['legal_moves']
                indeces_lm = ob['legal_moves_as_int']
                for m in moves:
                    action = -1

                    for ix in range(len(indeces_lm)):
                        if indeces_lm[ix] == m:
                            action = legal_moves[ix]
                            break
                    if action != -1:
                        break

                n_steps += 1

                # for debugging purpose
                if flags['debug']:
                    print('Agent: {} action: {}'.format(
                        obs['current_player'], action))
                obs, reward, done, _ = self.env.step(action)

                eps_reward += reward

                if done:
                   break

                agent_id = (agent_id + 1) % 2
            
            rewards.append(eps_reward)
            avg_steps += n_steps

        n_eps = float(flags['num_episodes'])
        avg_steps = avg_steps/float(n_eps)
        #print('Average Reward: %.3f' % (sum(rewards)/n_eps))
        #print('Average steps: %.2f' % avg_steps)
        return sum(rewards) * 100 + avg_steps

if __name__ == "__main__":

    # Initialize all models
    current_pool = []
    total_models = 30
    for i in range(total_models):
        # create a new newtwork
        new_guy = policy_net(658, 20, "NeuroEvo", modelname = str(i))
        new_guy.load()

        # add the network to the pool
        current_pool.append(new_guy)

    generations = 40
    score = np.zeros(total_models)
    averages = np.zeros(total_models)
    n_visits = np.zeros(total_models)
    for gen in range(generations):

        print("Generation %i " %gen)

        for i in range(total_models):
            # run them and evaluate them

            flags = {'players': 2,
                    'num_episodes': 10,
                    'id_agent': i,
                    'debug': False}

            # run the episodes
            runner = Runner(flags, current_pool[i])
            n_visits[i] = n_visits[i] + 1 #increase n visits
            score[i] = (score[i] + runner.run())
            averages[i] = score[i] /n_visits[i] # average the score

        # sort them
        ranking = np.argsort(averages)
        print("best: %i with score %f" %(ranking[-1], averages[ranking[-1]]))
        print("avg: %f" %(sum(averages)/total_models))

        worst_bunch = ranking[:int(total_models/2)]
        best_bunch = ranking[int(total_models/2):]

        for i in worst_bunch:
            randomA = np.random.choice(best_bunch)
            randomB = np.random.choice(best_bunch)
            current_pool[i] = model_crossover(randomA, randomB, str(i))
            score[i] = 0
            n_visits[i] = 0
            model_mutate(i)

    for agent in current_pool:
        agent.save()
            
