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

import numpy as np
from data_pipeline.state_translate import state_translator
from data_pipeline.bayes import Belief


Color2Num = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}


def convert_25to50(card):
    """
    Function to convert a 25 bit representation to a 50 bit representation.
    In a 50 bit representation, duplicates are represented

    args:
        card: a list or numpy array of size 25
    """

    new_card = [0 for _ in range(50)]  # emtpy 50 bits card

    for color in range(5):
        for bit in range(color*10, (color*10) + 3):
            new_card[bit] = card[color*5]
        for bit in range(color*10 + 3, (color*10) + 5):
            new_card[bit] = card[color*5 + 1]
        for bit in range(color*10 + 5, (color*10) + 7):
            new_card[bit] = card[color*5 + 2]
        for bit in range(color*10 + 7, (color*10) + 9):
            new_card[bit] = card[color*5 + 3]
        new_card[color*10 + 9] = card[color*5 + 4]

    return new_card


def update_discard(discardSpace, card):
    # step 1: select the spaces where possibly the discard can go
    dubDiscard = [0 for _ in discardSpace]
    for color in range(5):
        oneColor = discardSpace[color*10:(color + 1)*10]
        dubOneColor = [0 for _ in range(10)]
        oneApp = 0  # appearance of number 1
        for i in range(0, 3):
            if oneColor[i] == 1:
                oneApp = i + 1
        if oneApp < 3:
            dubOneColor[oneApp] = 1
        # for the three colors that have 2 cards each
        for z in range(3):
            twoApp = 3 + 2 * z
            for i in range(3 + (2 * z), 5 + (2 * z)):
                if oneColor[i] == 1:
                    twoApp = i + 1
                else:
                    break
            if twoApp < 5 + (2 * z):
                dubOneColor[twoApp] = 1
        # for fives
        if oneColor[9] < 1:
            dubOneColor[9] = 1

        # update the discard copy
        dubDiscard[color * 10:(color + 1)*10] = dubOneColor

    # step 2: convert the representation from 25 bits to 50 bits encoding
    newCardDiscard = convert_25to50(card)

    # step 3: sum the possibly discarded card to the existent discardSpace
    selectedDiscard = np.multiply(dubDiscard, newCardDiscard)
    discardSpace = (selectedDiscard + np.asarray(discardSpace)).tolist()

    # variable to control whether the bits overflow or not
    overFlow = True

    # if any bit has sum over 1, it moves the remainder to the following spot
    for color in range(5):
        oneColor = discardSpace[color*10:(color+1)*10]
        if oneColor[0] > 1:
            oneColor[1] += (oneColor[0] - 1)
        if oneColor[1] > 1:
            oneColor[2] += (oneColor[1] - 1)
        if oneColor[2] > 1:
            overFlow = True
        for z in range(3):
            if oneColor[3 + 2*z] > 1:
                oneColor[4 + 2*z] = oneColor[3 + 2*z]
            if oneColor[4+2*z] > 1:
                overFlow = True
        if oneColor[9] > 1:
            overFlow = True

    return discardSpace, overFlow


def state_tr(obs, move, players):

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

    if move['action_type'] == 'PLAY' or move['action_type'] == 'DISCARD':

        ix = move['card_index']  # index of card played or discarded
        hints = tr.cardKnowledge[ix*35: ix*35+25]

        # calculate the probability
        card_prob = belief.prob(hints)

        # the cardPlayed is probabilistic
        tr.cardPlayed = card_prob.tolist()

        # playerMissingCard
        if tr.currentDeck[0] == 0:
            tr.playerMissingCards[0] = 1

        # we take the last card from the deck
        tr.currentDeck = tr.currentDeck[1:] + [0]

        # the following are empty
        tr.lastMoveTarget = [0 for _ in tr.lastMoveTarget]
        tr.colorRevealed = [0 for _ in tr.colorRevealed]
        tr.rankRevealed = [0 for _ in tr.rankRevealed]
        tr.cardRevealed = [0 for _ in tr.cardRevealed]

        # cardKnowledge
        belief.add_known(card_prob)  # add the knowledge of the discarded card
        no_hint_card = belief.prob(np.ones(25))  # compute the new card
        tr.cardKnowledge[ix*35: ix*35 + 25] = no_hint_card

        # TODO: change the last 10 bits of the new card
        # TODO: do the case in which a card is missing

        if move['action_type'] == 'PLAY':

            """                         **** BoardSpace ****
            The probability of extracting the next piece is updated with the
            probability of the card that we might have in our hand,
            according to the hints
            """

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
                    tr.boardSpace[color * 5 + rank] += fire * \
                        card_prob[color * 5 + rank]
                    if rank > 0:
                        # and update the existing firework piece
                        tr.boardSpace[color * 5 + rank - 1] += fire * \
                            (1 - card_prob[color * 5 + rank])

            # lifeTokens DONE
            # TODO: test two transitions consecutively, to see if it computes right
            chanceToExplode = 1 - \
                (np.multiply(future_firework, card_prob)).sum()
            remaining_prob = 1
            for life in range(2, -1, -1):
                probableLife = tr.lifeTokens[life] - \
                    (tr.lifeTokens[life] * chanceToExplode)
                remaining_prob -= tr.lifeTokens[life]
                if life > 0:
                    if tr.lifeTokens[life - 1] == 1:
                        if tr.lifeTokens[life] == 1:
                            tr.lifeTokens[life] = probableLife
                            break
                        else:
                            tr.lifeTokens[life] = probableLife
                            tr.lifeTokens[life-1] = 1 - \
                                (remaining_prob * chanceToExplode)
                            break
                    else:
                        tr.lifeTokens[life] = probableLife

            # update discard space
            discardCard = card_prob - np.multiply(future_firework, card_prob)
            tr.discardSpace, overFlow = \
                update_discard(tr.discardSpace, discardCard)

            tr.lastMoveType = [1, 0, 0, 0]

            # positionPlayed DONE
            tr.positionPlayed = [0 for p in tr.positionPlayed]
            tr.positionPlayed[ix] = 1

            # prevPlay DONE
            # 1st bit: was successful
            tr.prevPlay[0] = 1 - chanceToExplode

            # 2nd bit: did it add an info token (successful and n.5)
            future_fives = [future_firework[i*5 + 4] for i in range(5)]
            prob_fives = [card_prob[i*5 + 4] for i in range(5)]
            chanceToSucceedAndFive = np.multiply(
                future_fives, prob_fives).sum()
            tr.prevPlay[1] = chanceToSucceedAndFive

            # infoTokens
            for token in range(len(tr.infoTokens)):
                if tr.infoTokens[token] < 1:
                    new_token = tr.infoTokens[token] + chanceToSucceedAndFive
                    if new_token <= 1:
                        tr.infoTokens[token] = new_token
                    else:
                        tr.infoTokens[token] = 1
                        new_token -= 1
                        if token + 1 > len(tr.infoTokens):
                            overFlow = True
                        else:
                            tr.infoTokens[token + 1] = new_token
                    break

        elif move['action_type'] == 'DISCARD':

            # boardSpace NO CHANGE
            # lifeTokens NO CHANGE

            # infoTokens: add one token every time
            tr.infoTokens = [1] + tr.infoTokens[:7]

            # discardSpace DONE
            tr.discardSpace, overFlow = update_discard(
                tr.discardSpace, card_prob)

            # lastMoveType is static
            tr.lastMoveType = [0, 1, 0, 0]

            # positionPlayed
            tr.positionPlayed = [0 for _ in tr.positionPlayed]
            tr.positionPlayed[ix] = 1

            # prevPlay
            tr.prevPlay = [0, 0]

    else:
        # if this is a hint move
        target_offset = move['target_offset']

        # handSpace NOT CHANGE
        # currentDeck NOT CHANGE
        # boardSpace NOT CHANGE
        # lifeTokens NOT CHANGE
        # discardSpace NOT CHANGE

        # infoTokens DONE
        tr.infoTokens = tr.infoTokens[1:] + [0]

        # lastMoveTarget
        # the target is the last active player + target offset
        tr.lastMoveTarget = [0 for i in tr.lastMoveTarget]  # reset the list
        moveTarget = (tr.lastActivePlayer.index(1) + target_offset) % players
        tr.lastMoveTarget[moveTarget] = 1  # one-hot encoded

        # positionPlayed
        # reset all position played - we are not playing but giving a hint
        tr.positionPlayed = [0 for p in tr.positionPlayed]

        # cardPlayed
        # reset all cards played - we are giving a hint not playing
        tr.cardPlayed = [0 for c in tr.cardPlayed]

        # prevPlay
        # no action is played, so there is no statistic for prevPlay
        tr.prevPlay = [0, 0]

        # cardRevealed (1/2)
        tr.cardRevealed = [0 for _ in tr.cardRevealed]
        bitHandSize = 25 * tr.handSize
        bitKnowSize = 35 * tr.handSize
        start_hand = (target_offset - 1) * bitHandSize
        start_know = (target_offset - 1) * bitKnowSize
        selectedHand = tr.handSpace[start_hand: start_hand + bitHandSize]
        selectedKnowledge = tr.cardKnowledge[start_know: start_know + bitKnowSize]

        handsize = tr.handSize - tr.playerMissingCards[target_offset]

        if move['action_type'] == 'REVEAL_RANK':
            # lastMoveType
            tr.lastMoveType = [0, 0, 0, 1]

            rank = move['rank']

            # colorRevealed
            # reset all color revealed - no color has been revealed this turn
            tr.colorRevealed = [0 for c in tr.colorRevealed]

            # rankRevealed
            # select the correct rank to reveal from the move dict
            tr.rankRevealed = [0 for r in tr.rankRevealed]
            tr.rankRevealed[rank] = 1

            # create a mask for cardKnowledge
            mask = np.zeros(25)
            for color in range(5):
                mask[color*5 + rank] = 1
            mask_inverse = 1 - mask

            for i in range(handsize):
                rankCard = selectedHand[i*25:(i+1)*25].index(1) % 5
                know_card = selectedKnowledge[i*35:i*35 + 25]
                if rankCard == move['rank']:
                    # cardRevealed
                    tr.cardRevealed[i] = 1
                    selectedKnowledge[i*35:i*35 + 25] = np.multiply(mask, know_card)
                    selectedKnowledge[i*35+30:(i+1)*35] = tr.rankRevealed
                else:
                    selectedKnowledge[i*35:i*35 + 25] = np.multiply(mask_inverse, know_card)
                    selectedKnowledge[i*35+30+rank] = 0

            # replace with the update knowledge
            tr.cardKnowledge[start_know: start_know + bitKnowSize] = selectedKnowledge

            # TODO case in which is probabilistic - or maybe not doing it

        else:
            # else if 'REVEAL_COLOR'

            # colorRevealed DONE
            tr.colorRevealed = [0 for _ in tr.colorRevealed]
            colorNum = Color2Num[move['color']]
            tr.colorRevealed[colorNum] = 1

            # rankRevealed DONE
            tr.rankRevealed = [0 for _ in tr.rankRevealed]

            # create a mask for cardKnowledge
            mask = np.zeros(25)
            mask[colorNum*5:(colorNum+1)*5] = 1
            mask_inverse = 1 - mask

            # cardRevealed
            for i in range(handsize):
                colorCard = int(selectedHand[i*25:(i + 1)*25].index(1) / 5)
                know_card = selectedKnowledge[i*35:i*35 + 25]
                if colorCard == colorNum:
                    tr.cardRevealed[i] = 1
                    selectedKnowledge[i*35:i*35 + 25] = np.multiply(mask, know_card)
                    selectedKnowledge[i*35+25:i*35 + 30] = tr.colorRevealed
                else:
                    selectedKnowledge[i*35:i*35 + 25] = np.multiply(mask_inverse, know_card)
                    selectedKnowledge[i*35+25+colorNum] = 0

            # replace with the update knowledge
            tr.cardKnowledge[start_know: start_know + bitKnowSize] = selectedKnowledge

            # TODO case in which is probabilistic

            # lastMoveType
            tr.lastMoveType = [0, 0, 1, 0]

    # re-encode the updated vector
    tr.encodeVector()
    new_obs = tr.stateVector

    return new_obs
