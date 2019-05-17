import numpy as np 
from state_translate import state_translator
from bayes import Belief 
#from state_transition.py import state_tr

def discard_move(obs, move, players):
   
    translate = state_translator(obs, players)
    #using Bayes to generate the "belief" about the cards 
    belief = Belief(players)  # Belief is the percentage/probabilities 
    #print("belief:")
    #print(belief)
    translate.positionPlayed = [0, 0, 0, 1, 0]
    translate.infoTokens = [1, 1, 1, 1, 1, 1, 1, 1]
    translate.cardPlayed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
      #simulated discard space
    # this is essentially to make sure that we encode the right vector for the specific move
    # that we're looking at

    translate.prevPlay =   [0, 0]
    translate.lastMoveType = [0,1,0,0]
    #translate.cardPlayed
    
    
    return obs 

def reveal_color(obs, move, players):
    
    #all match test
    translate = state_translator(obs, players)
    translate.prevPlay =  [0, 0] # was already [0,0]; but just to be safe
   
    translate.lastMoveTarget =  [0, 1]
    
    translate.cardPlayed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    translate.infoTokens = [1, 1, 1, 1, 1, 1, 1, 0]
    
    
    translate.positionPlayed = [0, 0, 0, 0, 0]

    translate.rankRevealed = [0, 0, 0, 0, 0]
    #took care of last move type 
    translate.lastMoveType = [0, 0, 1, 0]
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


