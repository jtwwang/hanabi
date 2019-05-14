import numpy as np 
from state_translate import state_translator
from bayes import Belief 
#from state_transition.py import state_tr

def discard_move(obs, move, players):
    print("discard")
    translate = state_translator(obs, players)
    #using Bayes to generate the "belief" about the cards 
    belief = Belief(players)  # Belief is the percentage/probabilities 
    #print("belief:")
    #print(belief)
    simDiscard = translate.discardSpace  #simulated discard space
    #print(simDiscard)
    # this is essentially to make sure that we encode the right vector for the specific move
    # that we're looking at

    translate.lastMoveType = [0,1,0,0]
    print(translate.lastMoveType)


    return obs 

def reveal_color(obs, move, players):
    return obs #FIXME: return the new observation


