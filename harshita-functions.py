import numpy as np 
from state_translate import state_translator
from bayes import Belief 
#from state_transition.py import state_tr

def discard_move(obs, move, players):
    translate = state_translator(obs, players)
    #using Bayes to generate the "belief" about the cards 
    belief = Belief(players)  # Belief is the percentage/probabilities 
    print("belief:")
    print(belief)
    simDiscard = translate.discardSpace  #simulated discard space
    print(simDiscard)
    

def reveal_color(obs, move, players):
    return obs #FIXME: return the new observation



# def discard_move(obs, move, players):

#     discardSpace = translate.discardSpace
#     # need to see what the new hand is 
#     ix = move['card_index']
#     belief.encode(obs)

#     cardKnowledge = translate.cardKnowledge
#     hints = [ix*35 : ix*35+25]
   

#     return obs #FIXME: return the new observation

# def reveal_color(obs, move, players):
#     translate = state_translator(obs, players)
#     # rank hint of last move, if last move was a rank hint
#     recColHint = translate.colorRevealed 
#     print(recColHint)
