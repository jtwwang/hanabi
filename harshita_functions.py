import numpy as np 
from state_translate import state_translator
from bayes import Belief 
#from state_transition.py import state_tr

def discard_move(obs, move, players):
    print("same move:?  ", move)
    translate = state_translator(obs, players)
    #using Bayes to generate the "belief" about the cards 
    belief = Belief(players)  # Belief is the percentage/probabilities 
    #print("belief:")
    #print(belief)
    simDiscard = translate.discardSpace  #simulated discard space
    
    print("TRANS sim Discard:" , (simDiscard))
    # this is essentially to make sure that we encode the right vector for the specific move
    # that we're looking at

    translate.lastMoveType = [0,1,0,0]
    print("HK Last move type: ", translate.lastMoveType)
    
    return obs 

def reveal_color(obs, move, players):
    #print(move)
    #not sure if we have to take care of offset 
    translate = state_translator(obs, players)
    if move['color'] == 'R':
        translate.colorRevealed = [1, 0, 0, 0, 0]
        # print(" RED colorRevealed: ",translate.colorRevealed)
    if move['color'] == 'Y':
        translate.colorRevealed =  [0, 1, 0, 0, 0]
        # print(" YELLOW colorRevealed: ", translate.colorRevealed)
    if move['color'] =='G':
        translate.colorRevealed = [0, 0, 1, 0, 0]
        # print(" GREEN colorRevealed: ", translate.colorRevealed)
        # [0, 0, 0, 0, 0]
    if move['color'] =='W':
        translate.colorRevealed = [0, 0, 0, 1, 0]
        # print(" WHITE colorRevealed: ",translate.colorRevealed)
    if move['color'] == 'B':
        translate.colorRevealed = [0, 0, 0, 0, 1]
        # print(" BLUE colorRevealed: ",translate.colorRevealed)
    
    return obs #FIXME: return the new observation


