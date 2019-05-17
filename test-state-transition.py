# Artificial Intelligence Society at UC Davis
#
#   http://www,aidavis.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
from agents.second_agent import SecondAgent
from agents.mc_agent import MCAgent
from agents.rainbow_agent_rl import RainbowAgent
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
import experience as exp
import rl_env
import getopt
from state_transition import state_tr
from state_translate import state_translator

import os
import sys
# To find local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))


AGENT_CLASSES = {
    'SimpleAgent':  SimpleAgent,
    'SecondAgent': SecondAgent,
    'RandomAgent':  RandomAgent,
    'RainbowAgent': RainbowAgent,
    'MCAgent': MCAgent}

def isWithinProb(prob, exp):
    ix = exp.index(1)
    if prob[ix] == 0:
        return False
    else:
        return True

"""
Function to compare two vectors to find out whether the prediciton is correct
"""
def compareVectors(pred, expected, players):
    transPred = state_translator(pred, players)
    transExp =  state_translator(expected, players)

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
    if transPred.lifeTokens != transExp.lifeTokens:
        failed = False
        for i in range(3):
            if transExp.lifeTokens[i] == 1 and transPred.lifeTokens[i] < 0.0001:
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
    if transPred.lastMoveType != transExp.lastMoveType:
        print("failed lastMoveType")
        print(transPred.lastMoveType)
        print(transExp.lastMoveType)
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
            agents = [agent for _ in range(self.flags['players'])]
        else:
            agents = [self.agent_class(self.agent_config)
                      for _ in range(self.flags['players'])]
            if self.flags['agent_class'] == 'MCAgent':
                for agent in range(len(agents)):
                    agents[agent].player_id = agent

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
                    action = agent.act(ob)

                    vec = obs['player_observations'][0]['vectorized']

                    move = self.moves_lookup(action, ob)
                    n_steps += 1

                    new_obs = state_tr(vec, action, self.flags['players']) #state transition

                    # for debugging purpose
                    if flags['debug']:
                        print('Agent: {} action: {}'.format(
                            obs['current_player'], action))

                    obs, reward, done, _ = self.env.step(action)

                    # test all attributes and print if it's wrong
                    vec = obs['player_observations'][0]['vectorized']
                    compareVectors(new_obs, vec, self.flags['players'])

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
    replay = exp.Experience(flags['agent_class'], numAgents=flags['players'])

    # run the episodes
    runner = Runner(flags)
    runner.run()

    # save the memory to file
    replay.save()
