import numpy as np
from state_translate import state_translator
from bayes import Belief
from harshita_functions import discard_move, reveal_color

def state_tr(obs, move, players):

    tr = state_translator(obs, players)
    belief = Belief(players)

    if move['action_type'] == 'PLAY':
        # index of the card played
        ix = move['card_index']

        # encode the belief before we do any operation
        belief.encode(obs)
        
        # handSpace NOT CHANGE
               
        # currentDeck DONE
        tr.currentDeck = tr.currentDeck[1:] + [0]

         # playerMissingCards DONE
        if tr.currentDeck[0] == 0:
             tr.playerMissingCards[0] = 1
        
        """                         **** BoardSpace ****
        The probability of extracting the next piece is updated with the probability
        of the card that we might have in our hand, according to the hints
        """
        cardKnowledge = tr.cardKnowledge
        hints = cardKnowledge[ix*35 : ix*35+25]

        # calculate the probability
        card_prob = belief.prob(hints)

        # create a copy
        fireworks = tr.boardSpace

        # future projection of the fireworks
        future_firework = []

        # empty the fireworks
        tr.boardSpace = [0 for _ in fireworks]
        for color in range(5):
            # take a single color and advance it by one
            firework = fireworks[color * 5: color * 5 + 5]
            firework = [1 - sum(firework)] + firework[:4]
            future_firework += firework
            for rank in range(5):
                fire = firework[rank]
                # update the probability with the possible extraction
                tr.boardSpace[color * 5 + rank] += fire * card_prob[color * 5 + rank]
                if rank > 0:
                    # and update the existing firework piece
                    tr.boardSpace[color * 5 + rank - 1] += fire * (1 - card_prob[color * 5 + rank])

        # infoTokens NOT CHANGE

        # lifeTokens DONE
        # TODO: test two transitions consecutively, to see if it computes right
        chanceToExplode = 1 - (np.multiply(future_firework, card_prob)).sum()
        remaining_prob = 1
        for life in range(2,-1,-1):
            probableLife = tr.lifeTokens[life] - (tr.lifeTokens[life] * chanceToExplode)
            remaining_prob -= tr.lifeTokens[life]
            if life > 0:
                if tr.lifeTokens[life - 1] == 1:
                    if tr.lifeTokens[life] == 1:
                        tr.lifeTokens[life] = probableLife
                        break
                    else:
                        tr.lifeTokens[life] = probableLife
                        tr.lifeTokens[life-1] = 1 - (remaining_prob * chanceToExplode)
                        break
                else:
                    tr.lifeTokens[life] = probableLife

        # discardSpace NOT CHANGE
        
        # lastActivePlayer DONE
        first = tr.lastActivePlayer[0]
        tr.lastActivePlayer = tr.lastActivePlayer[1:] + [first]

        # lastMoveType DONE
        tr.lastMoveType = [1,0,0,0]

        # lastMoveTarget DONE
        tr.lastMoveTarget = [0,0]   

        # colorRevealed DONE
        tr.colorRevealed = [0 for c in tr.colorRevealed]
        
        # rankRevealed DONE
        tr.rankRevealed = [0 for r in tr.rankRevealed]
        
        # positionPlayed DONE
        tr.positionPlayed = [0 for p in tr.positionPlayed]
        tr.positionPlayed[ix] = 1

        # cardPlayed DONE
        tr.cardPlayed = card_prob

        # prevPlay DONE
        #1st bit: was successful
        tr.prevPlay[0] = 1 - chanceToExplode

        #2nd bit: did it add an info token (successful and n.5)
        future_fives = [future_firework[i*5 + 4] for i in range(5)]
        prob_fives = [card_prob[i*5 + 4] for i in range(5)]
        chanceToSucceedAndFive = np.multiply(future_fives, prob_fives).sum()
        tr.prevPlay[1] = chanceToSucceedAndFive

        # cardKnowledge
        no_hint_card = belief.prob(np.ones(25))
        tr.cardKnowledge[ix*35: ix*35 + 25] = no_hint_card

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
