import numpy as np
from data_pipeline.state_translate import state_translator
from data_pipeline.bayes import Belief

"""
function to map the letter of the color to a corresponding number

Args:
    ColorLetter (string): the letter of the color
        valid letters: 'R', 'Y', 'G', 'W', 'B'

Returns:
    the correspondent number (int)
"""
def Color2Num(ColorLetter):
    if ColorLetter == 'R':
        return 0
    elif ColorLetter == 'Y':
        return 1
    elif ColorLetter =='G':
        return 2
    elif ColorLetter =='W':
        return 3
    else:
        return 4

"""
Function to convert a 25 bit representation to a 50 bit representation.
In a 50 bit representation, duplicates are represented

Args:
    card: a list or numpy array of size 25
"""
def convert_25to50(card):

    # emtpy 50 bits card
    new_card = [0 for _ in range(50)]

    for color in range(5):
        for bit in new_card[color*10:(color*10) + 3]:
            bit = card[color*5:color*5 + 1]
        for bit in new_card[color*10 + 3:(color*10) + 5]:
            bit = card[color*5 + 1:color*5 + 2]
        for bit in new_card[color*10 + 5:(color*10) + 7]:
            bit = card[color*5 + 2:color*5 + 3]
        for bit in new_card[color*10 + 7:(color*10) + 9]:
            bit = card[color*5 + 3:color*5 + 4]
        new_card[color*10 + 9:(color*10) + 10] = card[color*5 + 4:color*5 + 5]

    return new_card


def state_tr(obs, move, players):

    print(move)

    tr = state_translator(obs, players)

    # encode the belief before we do any operation
    belief = Belief(players)
    belief.encode(obs)

    # for all actions (play, discard, reveal color/rank)
    # lastActivePlayer changes in this way
    if 1 not in tr.lastActivePlayer:
        # if it's the first move of the game
        tr.lastActivePlayer[0] = 1
    else:
        # otherwise, cycles through players
        first = tr.lastActivePlayer[0]
        tr.lastActivePlayer = tr.lastActivePlayer[1:] + [first]

    if move['action_type'] == 'PLAY':
        # index of the card played
        ix = move['card_index']
      
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

        # discardSpace TODO: elinate possibilities accordingly to boardSpace

        # step 1: select the spaces where possibly the discard can go
        dubDiscard = [0 for _ in tr.discardSpace]
        for color in range(5):
            oneColor = tr.discardSpace[color*10:(color + 1)*10]
            dubOneColor = dubDiscard[color * 10:(color + 1)*10]
            oneApp = 0 #appearance of number 1
            for i in range(0,3):
                if oneColor[i] > 0:
                    oneApp = i + 1
            if oneApp < 3:
                dubOneColor[oneApp] = 1
            # for the three colors that have 2 cards each
            for z in range(3):
                twoApp = 3 + 2 * z
                for i in range(3 + 2 * z,5 + 2 * z):
                    if oneColor[i] > 0:
                        twoApp = i + 1
                if twoApp < 3 + 2 * z:
                    dubOneColor[twoApp] = 1
            # for fives
            if oneColor[9] == 0:
                dubOneColor[9] = 1

            #update the discard copy
            dubDiscard[color * 10:(color + 1)*10] = dubOneColor

        # step 2: convert the representation from 25 bits to 50 bits encoding
        newCardDiscard = convert_25to50(card_prob)
        
        # step 3: sum the possibly discarded card to the existent discardSpace
        selectedDiscard = np.multiply(dubDiscard, newCardDiscard)
        tr.discardSpace = (selectedDiscard + np.asarray(tr.discardSpace)).tolist()
        
        # lastMoveType DONE
        tr.lastMoveType = [1,0,0,0]

        # lastMoveTarget DONE
        tr.lastMoveTarget = [0,0]   

        # colorRevealed DONE
        tr.colorRevealed = [0 for c in tr.colorRevealed]
        
        # rankRevealed DONE
        tr.rankRevealed = [0 for r in tr.rankRevealed]
        
        # cardRevealed DONE
        tr.cardRevealed = [0 for _ in tr.cardRevealed]

        # positionPlayed DONE
        tr.positionPlayed = [0 for p in tr.positionPlayed]
        tr.positionPlayed[ix] = 1

        # cardPlayed DONE
        tr.cardPlayed = card_prob.tolist()

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
        # TODO are all cards with no hints full of 1s? or if the card is on the table there is a 0?
        tr.cardKnowledge[ix*35: ix*35 + 25] = no_hint_card

    elif move['action_type'] == 'DISCARD':

        ix = move['card_index']
        cardKnowledge = tr.cardKnowledge
        hints = cardKnowledge[ix*35 : ix*35+25]

        # calculate the probability
        card_prob = belief.prob(hints)

        # handSpace DOESN'T CHANGE

        # playerMissingCards DONE
        if tr.currentDeck[0] == 0:
             tr.playerMissingCards[0] = 1
        
        # currentDeck DONE
        # we take out the last card from the deck
        tr.currentDeck = tr.currentDeck[1:] + [0]

        # boardSpace DOESN'T CHANGE

        # infoTokens DONE
        # add one token every time
        tr.infoTokens = [1] + tr.infoTokens[:7]

        # lifeTokens NOT CHANGE
        
        # discardSpace DONE
        # step 1: select the spaces where possibly the discard can go
        dubDiscard = [0 for _ in tr.discardSpace]
        for color in range(5):
            oneColor = tr.discardSpace[color*10:(color + 1)*10]
            dubOneColor = dubDiscard[color * 10:(color + 1)*10]
            oneApp = 0 #appearance of number 1
            for i in range(0,3):
                if oneColor[i] > 0:
                    oneApp = i + 1
            if oneApp < 3:
                dubOneColor[oneApp] = 1
            # for the three colors that have 2 cards each
            for z in range(3):
                twoApp = 3 + 2 * z
                for i in range(3 + 2 * z,5 + 2 * z):
                    if oneColor[i] > 0:
                        twoApp = i + 1
                if twoApp < 3 + 2 * z:
                    dubOneColor[twoApp] = 1
            # for fives
            if oneColor[9] == 0:
                dubOneColor[9] = 1

            #update the discard copy
            dubDiscard[color * 10:(color + 1)*10] = dubOneColor

        # step 2: convert the representation from 25 bits to 50 bits encoding
        newCardDiscard = convert_25to50(card_prob)
        
        # step 3: sum the possibly discarded card to the existent discardSpace
        selectedDiscard = np.multiply(dubDiscard, newCardDiscard)
        tr.discardSpace = (selectedDiscard + np.asarray(tr.discardSpace)).tolist()

        # retrieve the knowledge
        cardKnowledge = tr.cardKnowledge
        hints = cardKnowledge[ix*35 : ix*35+25]

        # calculate the probability
        card_prob = belief.prob(hints)

        # simulated discard space
        # this is essentially to make sure that we encode the right vector for the specific move
        # that we're looking at
        tr.lastMoveType = [0,1,0,0]

        # lastMoveTarget DONE
        tr.lastMoveTarget = [0 for _ in tr.lastMoveTarget]

        # colorRevealed DONE
        tr.colorRevealed = [0 for _ in tr.colorRevealed]

        # rankRevealed DONE
        tr.rankRevealed = [0 for _ in tr.rankRevealed]

        # cardRevealed DONE
        tr.cardRevealed = [0 for _ in tr.cardRevealed]

        # positionPlayed
        tr.positionPlayed = [0 for _ in tr.positionPlayed]
        tr.positionPlayed[ix] = 1
    
        # cardPlayed DONE
        # the card played is probabilistic
        tr.cardPlayed = card_prob.tolist()

        # prevPlay
        tr.prevPlay = [0, 0]

        # cardKnowledge TODO
        
    else:
        # if this is a hint move

        target_offset = move['target_offset']
        
        # handSpace NOT CHANGE
        # currentDeck NOT CHANGE
        # boardSpace NOT CHANGE

        # infoTokens DONE
        tr.infoTokens = tr.infoTokens[1:] + [0]
        
        # lifeTokens NOT CHANGE
        # discardSpace NOT CHANGE

        # lastMoveTarget
        # the target is the last active player + target offset
        
        tr.lastMoveTarget = [0 for i in tr.lastMoveTarget] # reset the list
        moveTarget = (tr.lastActivePlayer.index(1) + target_offset) % players
        tr.lastMoveTarget[moveTarget] = 1 # one-hot encoded

        # positionPlayed
        # reset all position played - we are not playing but giving a hint
        tr.positionPlayed = [0 for p in tr.positionPlayed]

        # cardPlayed
        # reset all cards played - we are giving a hint not playing
        tr.cardPlayed = [0 for c in tr.cardPlayed]

        # prevPlay
        # no action is played, so there is no statistic for prevPlay
        tr.prevPlay = [0,0]

        # cardKnoweldge TODO

        # cardRevealed (1/2)
        tr.cardRevealed = [0 for _ in tr.cardRevealed]
        bitHandSize = 25 * tr.handSize
        start_hand = (target_offset - 1)* bitHandSize
        selectedHand = tr.handSpace[start_hand: start_hand + bitHandSize]

        if move['action_type'] == 'REVEAL_RANK':
            # lastMoveType
            tr.lastMoveType = [0,0,0,1]

            # colorRevealed
            # reset all color revealed - no color has been revealed this turn
            tr.colorRevealed = [0 for c in tr.colorRevealed]

            # rankRevealed
            # select the correct rank to reveal from the move dict
            tr.rankRevealed = [0 for r in tr.rankRevealed]
            tr.rankRevealed[move['rank']] = 1

            # cardRevelaed (2/2)
            for i in range(tr.handSize):
                rankCard = selectedHand[i*25:(i+1)*25].index(1) % 5
                if rankCard == move['rank']:
                    tr.cardRevealed[i] = 1
            # TODO check if rank card knowledge is already in cardKnowledge
            # if it is cardKnowledge, then get rid of that card in cardRevealed
            # MAYBE ?? NOT SURE IF THIS IS THE PROBLEM

        else:
            # else if 'REVEAL_COLOR' (left it implicit)
            # lastMoveType
            tr.lastMoveType = [0,0,1,0]

            # colorRevealed DONE
            tr.colorRevealed = [0 for _ in tr.colorRevealed]
            colorNum = Color2Num(move['color'])
            tr.colorRevealed[colorNum] = 1
            
            # rankRevealed DONE
            tr.rankRevealed = [0 for _ in tr.rankRevealed]

            # cardRevealed (2/2)
            for i in range(tr.handSize):
                #print(selectedHand[i*25:(i+1)*25])
                colorCard = int(selectedHand[i*25:(i + 1)*25].index(1) / 5)
                if colorCard == colorNum:
                    tr.cardRevealed[i] = 1
            #print(tr.cardRevealed)
            # CAN'T UNDERSTAND WHY THIS IS NOT WORKING

    tr.encodeVector()
    new_obs = tr.stateVector

    return new_obs
