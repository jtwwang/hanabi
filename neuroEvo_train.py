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
#
import rl_env
import numpy as np
import os
import sys
import random
import getopt
import pickle
from agents.neuroEvo_agent import NeuroEvoAgent
from predictors.conv_pred import conv_pred
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


def model_crossover(weights1, weights2):

    new_weights = []
    assert len(weights1) == len(weights2)
    if random.uniform(0, 1) > 0.3:
        print("crossover")
        for layer in range(len(weights1)):
            # alternate odd and even layers
            if layer % 2 == 0:
                new_weights.append(weights1[layer])
            else:
                new_weights.append(weights2[layer])
    else:
        print("no crossover")
        new_weights = weights1

    return new_weights


def mutate_weights(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0, 1) > 0.9:
                change = random.uniform(-0.1, 0.1)
                weights[xi][yi] += change
    return weights


def make_mutation(ix_to_mutate, best_ones):

    p = np.sort(scores)[2:]
    p = p / np.sum(p)

    # select the weights from parents
    randomA = np.random.choice(best_ones, p=p)
    randomB = np.random.choice(best_ones, p=p)
    while randomB == randomA:
        randomB = np.random.choice(best_ones, p=p)
    weights1 = weights[randomA]
    weights2 = weights[randomB]

    # generate new weights
    new_weights = model_crossover(weights1, weights2)
    new_weights = mutate_weights(new_weights)

    # change the weights of the target agent
    weights[ix_to_mutate] = new_weights


def run(ix, initialize=False):

    # initialize env
    env = rl_env.make('Hanabi-Full', num_players=flags['players'])
    agent_config = {
        'players': flags['players'],
        'num_moves': env.num_moves(),
        'observation_size': env.vectorized_observation_shape()[0],
        'model_name': str(ix),
        'initialize': initialize}

    agent = NeuroEvoAgent(agent_config)

    avg_reward = 0
    avg_steps = 0

    for eps in range(flags['num_episodes']):
        obs = env.reset()  # Observation of all players
        done = False
        agent_id = 0

        while not done:
            ob = obs['player_observations'][agent_id]

            try:
                action = agent.act(ob)
            except ValueError:
                print('Something went wrong. Try to reinitialize the agents'
                      'pool by using --initialize True')
                exit()

            obs, reward, done, _ = env.step(action)

            avg_reward += reward
            avg_steps += 1

            if done:
                break

            # change player
            agent_id = (agent_id + 1) % flags['players']

    n_eps = float(flags['num_episodes'])
    avg_steps /= n_eps
    avg_reward /= n_eps

    agent.save(model_name=str(ix))
    scores[ix] = avg_reward * 1000 + avg_steps


if __name__ == "__main__":

    global flags, scores, weights
    flags = {'players': 2,
             'num_episodes': 100,
             'initialize': False,
             'models': 20,
             'generations': 100}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'initialize=',
                                        'models=',
                                        'generations='])
    if arguments:
        sys.exit('usage: neuroEvo.py [options]\n'
                 '--players      number of players in the game.\n'
                 '--num_episodes number of game episodes to run.\n'
                 '--initialize   whether to re-initialize the weights'
                 'for all agents.\n')

    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    # Initialize all models
    current_pool = []
    scores = np.zeros(flags['models'])
    weights = {}
    to_mutate = 0

    # create one agent
    agent = conv_pred("NeuroEvo_agent")

    # load the file
    filepath = os.path.join("model", "NeuroEvo_agent")
    filepath = os.path.join(filepath, "scores.pickle")
    if not flags['initialize']:
        try:
            scores = pickle.load(open(filepath, "rb"))
            loaded = True
        except IOError:
            loaded = False
    else:
        loaded = False

    print("Initialize")
    # do an initial loop to evaluate all models
    for i in range(flags['models']):
        if flags['initialize'] or not loaded:
            run(i, flags['initialize'])
        agent.load(model_name=str(i))
        weights[i] = agent.model.get_weights()

    for gen in range(flags['generations']):

        print("Generation %i " % gen)

        # sort the results
        ranking = np.argsort(scores)
        print("best: %i with score %f" % (ranking[-1], scores[ranking[-1]]))
        print("worst: %i with score %f" % (ranking[0], scores[ranking[0]]))
        print("avg: %f" % (sum(scores)/flags['models']))

        # divide worst from best
        worst_ones = ranking[:2]
        best_ones = ranking[2:]

        # select the one to mutate and the one to use for the simulation
        ix_to_mutate = worst_ones[to_mutate]
        ix_to_simulate = worst_ones[1 - to_mutate]

        run(ix_to_simulate)
        make_mutation(ix_to_mutate, best_ones)

        # update weights of mutated agent
        agent.model.set_weights(weights[ix_to_mutate])
        agent.save(model_name=str(ix_to_mutate))

        # prepare for next generation
        to_mutate = (to_mutate + 1) % 2

    # save the rankings
    pickle.dump(scores, open(filepath, "wb"))
    print("Saved scores.")
