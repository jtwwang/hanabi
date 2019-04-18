import numpy as np
from state_translate import state_translator
from bayes import Belief
from harshita_functions import discard_move, reveal_color

def state_tr(obs, move, players):

    translate = state_translator(obs, players)
    belief = Belief(players)

    if move['action_type'] == 'PLAY':
        # play
        ix = move['card_index']
        hand = translate.handSpace
        print("hand:")
        print(hand)

        belief.encode(obs)
        print("full know")
        print(belief.full_know)

        cardKnowledge = translate.cardKnowledge
        hints = cardKnowledge[ix*35 : ix*35+25]
        print("hints")
        print(hints)

        card_prob = belief.prob(hints)
        print("card probability")
        print(card_prob)

        return obs

    elif move['action_type'] == 'DISCARD':
        # discard
        return discard_move(obs, move, players)
    elif move['action_type'] == 'REVEAL_COLOR':
        # reveal color
        return reveal_color(obs, move, players)
    else:
        # reveal rank
        return obs

    return new_obs
