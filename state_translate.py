class state_translator:
    """
    Manages an observation vector.
    Assumes 5 colors and 5 ranks in a game. Assumes 3:2:2:2:1 rank split.
    Based on the vector encoding logic established in hanabi_lib/canonical_encoders.cc
    """
    stateVector = [] #entire state vector
    playerCount
    stateVectorSize

    handSpace = [] #information about each card visible in players hands
    playerMissingCards = [] #a single bit indicating if a player has a handsize below max

    currentDeck = [] #a single bit for each card in the deck indicating if it is drawn
    boardSpace = [] #5 blocks of color for each card played successfully on the board
    infoTokens = [] #a single bit for each information token
    lifeTokens = [] #a single bit for each life token

    discardSpace = [] #tracks all discarded cards

    lastActivePlayer = [] #last player to move
    lastMoveType = [] #last move type
    lastMoveTarget = [] #target of last move relative to last active player
    colorRevealed = [] #color hint of last move, if last move was a color hint
    rankRevealed = [] #rank hint of last move, if last move was a rank hint
    cardRevealed = [] #each card in the target hand that was just hinted at
    positionPlayed = [] #which position in hand was just played from
    cardPlayed = [] #which card was just played
    prevPlay = [] #did previous play succeed, and did it generate a hint token
    cardKnowledge = [] #encodes what each player knows about their hand


    def __init__(self, givenStateVector, givenPlayerCount):
        stateVector = givenStateVector
        playerCount = givenPlayerCount
        stateVectorSize = len(stateVector)
        decodeVector(stateVector)

    def encodeVector():
        stateVector = handSpace + playerMissingCards + currentDeck + boardSpace + infoTokens + lifeTokens + discardSpace + lastActivePlayer + lastMoveType + lastMoveTarget + colorRevealed + rankRevealed + cardRevealed + positionPlayed + cardPlayed + prevPlay + cardKnowledge

        if (len(stateVector) != stateVectorSize):
            raise ValueError('stateVector size has changed since last encodeVector() call.')

    def decodeVector(stateVector):
        if(playerCount == 2 or playerCount == 3):
            handSize = 5
        elif(playerCount == 4 or playerCount == 5):
            handSize = 4
        else
            raise ValueError('playerCount is invalid number')

        prevIndex = 0

        numCardsSeen = (playerCount - 1) * handSize
        handSpace = stateVector[prevIndex:(prevIndex+numCardsSeen)]
        prevIndex += numCardsSeen

        playerMissingCards = stateVector[prevIndex:(prevIndex+playerCount)]
        prevIndex += playerCount

        deckSize = 50 - numCardsSeen #Assumes 50 cards in game total.
        currentDeck = stateVector[prevIndex:(prevIndex+deckSize)]
        prevIndex += deckSize

        maxBoardSpace = 25
        boardSpace = stateVector[prevIndex:(prevIndex+maxBoardSpace)]
        prevIndex += maxBoardSpace

        numInfoTokens = 8
        infoTokens = stateVector[prevIndex:(prevIndex+numInfoTokens)]
        prevIndex += numInfoTokens

        numLifeTokens = 3
        lifeTokens = stateVector[prevIndex:(prevIndex + numLifeTokens)]
        prevIndex += numLifeTokens

        discardSpace = stateVector[prevIndex:(prevIndex + 50)]
        prevIndex += 50

        lastActivePlayer = stateVector[prevIndex:(prevIndex + playerCount)]
        prevIndex += playerCount

        lastMoveType = stateVector[prevIndex:(prevIndex + 4)]
        prevIndex += 4

        lastMoveTarget = stateVector[prevIndex:(prevIndex + playerCount)]
        prevIndex += playerCount

        colorRevealed = stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        rankRevealed = stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        cardRevealed = stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        #in the context of the next two arrays, 'played' means played or discarded
        positionPlayed = stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        cardPlayed = stateVector[prevIndex:(prevIndex + 25)]
        prevIndex += 25

        prevPlay = stateVector[prevIndex:(prevIndex + 2)]
        prevIndex += 2

        cardKnowledge = stateVector[prevIndex:(prevIndex + playerCount * handSize * 35)]



    def getStateVector():
        return stateVector
