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

from agents.mc_agent import MCAgent
import data_pipeline.experience as exp
from data_pipeline.state_translate import state_translator
from state_transition import state_tr
from run_simulations import Runner, cross_play
import getopt

import os
import sys
import numpy as np

# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


def isWithinProb(prob, exp):
    """
    function to check if the one-hot encoded is contemplated by the probabilistic
    representation of the state

    args:
        prob: a list with probabilities
        exp: a one-hot encoded list
    return
        boolean (True/False): whether the probability can possibly represent
        the one-hot encoded list
    """
    if np.sum(np.multiply(prob, exp)) == 0:
        if np.sum(exp) == 0:
            return True
        else:
            return False
    else:
        return True


def compareVectors(pred, expected, players):
    """
    Function to compare two vectors to find out whether the prediciton
    is correct
    """
    transPred = state_translator(pred, players)
    transExp = state_translator(expected, players)

    if transPred.handSpace != transExp.handSpace:
        failed = False
        if sum(transExp.handSpace) - sum(transPred.handSpace) > 0.00001:
            failed = True
            print("failed HandSpace - sum of probabilities is incorrect")
        elif not isWithinProb(transPred.handSpace, transExp.handSpace):
            print("failed HandSpace - probability is not contained")
            failed = True
        if failed:
            print(transPred.handSpace)
            print(transExp.handSpace)
    if transPred.playerMissingCards != transExp.playerMissingCards:
        print("failed playerMissingCards")
        print(transPred.playerMissingCards)
        print(transExp.playerMissingCards)
    if transPred.currentDeck != transExp.currentDeck:
        print("failed currentDeck")
        print(transPred.currentDeck)
        print(transExp.currentDeck)
    if transPred.boardSpace != transExp.boardSpace:
        if not isWithinProb(transPred.boardSpace, transExp.boardSpace):
            print("failed boardSpace - probability not contained")
            print(transPred.boardSpace)
            print(transExp.boardSpace)
    if transPred.lifeTokens != transExp.lifeTokens:
        failed = False
        for i in range(3):
            if transExp.lifeTokens[i] == 1 and transPred.lifeTokens[i] < 10e-4:
                failed = True
                break
        if failed:
            print("failed lifeTokens")
            print(transPred.lifeTokens)
            print(transExp.lifeTokens)
    if transPred.infoTokens != transExp.infoTokens:
        print("failed infoTokens")
        print(transPred.infoTokens)
        print(transExp.infoTokens)
    if transPred.lastActivePlayer != transExp.lastActivePlayer:
        print("failed lastActivePlayer")
        print(transPred.lastActivePlayer)
        print(transExp.lastActivePlayer)
    if transPred.discardSpace != transExp.discardSpace:
        if not isWithinProb(transPred.discardSpace, transExp.discardSpace):
            print("failed discardSpace")
            print(transPred.discardSpace)
            print(transExp.discardSpace)
    if transPred.lastMoveType != transExp.lastMoveType:
        print("failed lastMoveType")
        print(transPred.lastMoveType)
        print(transExp.lastMoveType)
    if transPred.lastMoveTarget != transExp.lastMoveTarget:
        print("failed lastMoveTarget")
        print(transPred.lastMoveTarget)
        print(transExp.lastMoveTarget)
    if transPred.colorRevealed != transExp.colorRevealed:
        print("failed colorRevealed")
        print(transPred.colorRevealed)
        print(transExp.colorRevealed)
    if transPred.rankRevealed != transExp.rankRevealed:
        print("failed rankRevealed")
        print(transPred.rankRevealed)
        print(transExp.rankRevealed)
    if transPred.cardRevealed != transExp.cardRevealed:
        print("failed cardRevealed")
        print(transPred.cardRevealed)
        print(transExp.cardRevealed)
    if transPred.positionPlayed != transExp.positionPlayed:
        print("failed positionPlayed")
        print(transPred.positionPlayed)
        print(transExp.positionPlayed)
    if transPred.cardPlayed != transExp.cardPlayed:
        failed = False
        if sum(transPred.cardPlayed) < 0.9999:
            print("cardPlayed FAILED - wrong sum of probabilities")
            failed = True
        elif not isWithinProb(transPred.cardPlayed, transExp.cardPlayed):
            print("cardPlayed FAILED - probability not contained")
            failed = True

        if failed:
            print(transPred.cardPlayed)
            print(transExp.cardPlayed)
    if transPred.prevPlay != transExp.prevPlay:
        failed = False
        if transExp.prevPlay[0] == 1 and transExp.prevPlay[0] == 0:
            failed = True
        elif transExp.prevPlay[1] == 1 and transExp.prevPlay[1] == 0:
            failed = True
        if failed:
            print("prevPlay")
            print(transPred.prevPlay)
            print(transExp.prevPlay)
    if transPred.cardKnowledge != transExp.cardKnowledge:
        if not isWithinProb(transPred.cardKnowledge, transExp.cardKnowledge):
            print("failed cardKnowledge - probability is not contained")
            print(transPred.cardKnowledge)
            print(transExp.cardKnowledge)


class TestRunner(Runner):
    """Runner class."""

    def run(self):

        agents = self.generate_pool()

        avg_steps = 0
        avg_reward = 0

        for eps in range(flags['num_episodes']):

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

                    vec = obs['player_observations'][0]['vectorized']

                    move = self.moves_lookup(action, ob)
                    avg_steps += 1

                    # state transition
                    new_obs = state_tr(vec, action, self.flags['players'])

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)

                    # test all attributes and print if it's wrong
                    if agent_id == 0:
                        vec = obs['player_observations'][0]['vectorized']
                        compareVectors(new_obs, vec, self.flags['players'])

                    avg_reward += reward

                    if done:
                        break

        n_eps = float(flags['num_episodes'])
        avg_reward /= n_eps
        avg_steps /= n_eps
        print('Average Reward: %.3f' % avg_reward)
        print('Average steps: %.2f' % avg_steps)


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
             'checkpoint_dir': "",
             'checkpoint_dir2': "",
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
            '--agent_predicted <str>   necessary if using MCAgent or NNAgent.'
            'Use one of the other classes as string.\n'
            '--model_class <str>       network type ["dense", "conv", "lstm"].\n'
            '--model_name <str>        model name of a pre-trained model.\n'
            '--agent2_class <str>      to play \'ad hoc\' against another agent.\n'
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
        runner = TestRunner(flags)
        runner.run()
