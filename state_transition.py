import numpy as np
from state_translate import state_translator
from bayes import Belief
from harshita_functions import discard_move, reveal_color

def state_tr(obs, move, players):

    translate = state_translator(obs, players)
    belief = Belief(players)

    if move['action_type'] == 'PLAY':
        # index of the card played
        ix = move['card_index']

        # encode the belief before we do any operation
        belief.encode(obs)
        
        # handSpace NOT CHANGE
        # playerMissingCards TODO
        
        # currentDeck DONE
        translate.currentDeck = translate.currentDeck[1:] + [0]

        # boardSpace TODO
        cardKnowledge = translate.cardKnowledge
        hints = cardKnowledge[ix*35 : ix*35+25]
        card_prob = belief.prob(hints)
        # HERE TRANSLATE THE FIREWORK by one
        new_card = np.multiply(card_prob, translated_firework)
        translate.boardSpace += new_card

        # infoTokens NOT CHANGE
        # lifeTokens TODO
        # discardSpace NOT CHANGE
        
        # lastActivePlayer DONE
        first = translate.lastActivePlayer[0]
        translate.lastActivePlayer = translate.lastActivePlayer[1:] + [first]

        # lastMoveType DONE
        translate.lastMoveType = [1,0,0,0]

        # lastMoveTarget DONE
        translate.lastMoveTarget = [0,0]   

        # colorRevealed DONE
        translate.colorRevealed = [0 for c in translate.colorRevealed]
        
        # rankRevealed DONE
        translate.rankRevealed = [0 for r in translate.rankRevealed]
        
        # positionPlayed DONE
        translate.positionPlayed = [0 for p in translate.positionPlayed]
        translate.positionPlayed[ix] = 1

        # cardPlayed DONE
        translate.cardPlayed = card_prob

        # prevPlay TODO
        #1st bit: was successful
        #2nd bit: did it add an info token (successful and n.5)

        # cardKnowledge
        no_hint_card = belief.prob(np.ones(25))
        translate.cardKnowledge[ix*35: ix*35 + 25] = no_hint_card

        return obs

    elif move['action_type'] == 'DISCARD':
        # discard
        print("discard")
        return discard_move(obs, move, players)
    elif move['action_type'] == 'REVEAL_COLOR':
        # reveal color
        print("color")
        return reveal_color(obs, move, players)
    else:
        # reveal rank
        print("rank")
        return obs

    return new_obs
