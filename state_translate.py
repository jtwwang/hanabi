class state_translator:
    """
    Manages an observation vector.
    Assumes 5 colors and 5 ranks in a game. Assumes 3:2:2:2:1 rank split.
    Based on the vector encoding logic established in hanabi_lib/canonical_encoders.cc
    """

    def __init__(self, givenStateVector, givenPlayerCount):
        self.stateVector = givenStateVector #entire state vector
        self.playerCount = givenPlayerCount
        self.stateVectorSize = len(self.stateVector)

        self.handSpace = [] #information about each card visible in players hands
        self.playerMissingCards = [] #a single bit indicating if a player has a handsize below max

        self.currentDeck = [] #a single bit for each card in the deck indicating if it is drawn
        self.boardSpace = [] #5 blocks of color for each card played successfully on the board
        self.infoTokens = [] #a single bit for each information token
        self.lifeTokens = [] #a single bit for each life token

        self.discardSpace = [] #tracks all discarded cards

        self.lastActivePlayer = [] #last player to move
        self.lastMoveType = [] #last move type
        self.lastMoveTarget = [] #target of last move relative to last active player
        self.colorRevealed = [] #color hint of last move, if last move was a color hint
        self.rankRevealed = [] #rank hint of last move, if last move was a rank hint
        self.cardRevealed = [] #each card in the target hand that was just hinted at
        self.positionPlayed = [] #which position in hand was just played from
        self.cardPlayed = [] #which card was just played
        self.prevPlay = [] #did previous play succeed, and did it generate a hint token
        self.cardKnowledge = [] #encodes what each player knows about their hand


        self.decodeVector(self.stateVector)

    def encodeVector(self):
        self.stateVector = self.handSpace + self.playerMissingCards + self.currentDeck + self.boardSpace + self.infoTokens + self.lifeTokens + self.discardSpace + self.lastActivePlayer + self.lastMoveType + self.lastMoveTarget + self.colorRevealed + self.rankRevealed + self.cardRevealed + self.positionPlayed + self.cardPlayed + self.prevPlay + self.cardKnowledge
        if (len(self.stateVector) != self.stateVectorSize):
            raise ValueError('stateVector size has changed since last encodeVector() call.')

    def decodeVector(self, stateVector):
        if(self.playerCount == 2 or self.playerCount == 3):
            handSize = 5
        elif(self.playerCount == 4 or self.playerCount == 5):
            handSize = 4
        else:
            raise ValueError('playerCount is invalid number')

        prevIndex = 0

        numCardsSeen = (self.playerCount - 1) * handSize
        self.handSpace = stateVector[prevIndex:(prevIndex+numCardsSeen*25)]
        prevIndex += numCardsSeen*25

        self.playerMissingCards = stateVector[prevIndex:(prevIndex+self.playerCount)]
        prevIndex += self.playerCount

        deckSize = 50 - (self.playerCount * handSize) #Assumes 50 cards in game total.
        self.currentDeck = stateVector[prevIndex:(prevIndex+deckSize)]
        prevIndex += deckSize

        maxBoardSpace = 25
        self.boardSpace = stateVector[prevIndex:(prevIndex+maxBoardSpace)]
        prevIndex += maxBoardSpace

        numInfoTokens = 8
        self.infoTokens = stateVector[prevIndex:(prevIndex+numInfoTokens)]
        prevIndex += numInfoTokens

        numLifeTokens = 3
        self.lifeTokens = stateVector[prevIndex:(prevIndex + numLifeTokens)]
        prevIndex += numLifeTokens

        self.discardSpace = stateVector[prevIndex:(prevIndex + 50)]
        prevIndex += 50

        self.lastActivePlayer = stateVector[prevIndex:(prevIndex + self.playerCount)]
        prevIndex += self.playerCount

        self.lastMoveType = stateVector[prevIndex:(prevIndex + 4)]
        prevIndex += 4

        self.lastMoveTarget = stateVector[prevIndex:(prevIndex + self.playerCount)]
        prevIndex += self.playerCount

        self.colorRevealed = stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        self.rankRevealed = stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        self.cardRevealed = stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        #in the context of the next two arrays, 'played' means played or discarded
        self.positionPlayed = stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        self.cardPlayed = stateVector[prevIndex:(prevIndex + 25)]
        prevIndex += 25

        self.prevPlay = stateVector[prevIndex:(prevIndex + 2)]
        prevIndex += 2

        self.cardKnowledge = stateVector[prevIndex:(prevIndex + self.playerCount * handSize * 35)]

    def getStateVector(self):
        return self.stateVector



