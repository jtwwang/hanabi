class state_translator:
    """
    Manages an observation vector.
    Assumes 5 colors and 5 ranks in a game. Assumes 3:2:2:2:1 rank split.
    Based on the vector encoding logic established in hanabi_lib/canonical_encoders.cc
    """
    stateVector = [] #entire state vector

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
        self.stateVector = givenStateVector
        self.playerCount = givenPlayerCount
        self.stateVectorSize = len(self.stateVector)
        self.decodeVector()

    def encodeVector(self):
        self.stateVector = self.handSpace + self.playerMissingCards + self.currentDeck + self.boardSpace + self.infoTokens + self.lifeTokens + self.discardSpace + self.lastActivePlayer + self.lastMoveType + self.lastMoveTarget + self.colorRevealed + self.rankRevealed + self.cardRevealed + self.positionPlayed + self.cardPlayed + self.prevPlay + self.cardKnowledge

        if (len(self.stateVector) != self.stateVectorSize):
            raise ValueError('stateVector size has changed since last encodeVector() call.')

    def decodeVector(self):
        if(self.playerCount == 2 or self.playerCount == 3):
            handSize = 5
        elif(self.playerCount == 4 or self.playerCount == 5):
            handSize = 4
        else:
            raise ValueError('self.playerCount is invalid number')

        prevIndex = 0

        numCardsSeen = (self.playerCount - 1) * handSize
        self.handSpace = self.stateVector[prevIndex:(prevIndex+numCardsSeen)]
        prevIndex += numCardsSeen

        self.playerMissingCards = self.stateVector[prevIndex:(prevIndex+self.playerCount)]
        prevIndex += self.playerCount

        deckSize = 50 - numCardsSeen #Assumes 50 cards in game total.
        self.currentDeck = self.stateVector[prevIndex:(prevIndex+deckSize)]
        prevIndex += deckSize

        maxBoardSpace = 25
        self.boardSpace = self.stateVector[prevIndex:(prevIndex+maxBoardSpace)]
        prevIndex += maxBoardSpace

        numInfoTokens = 8
        self.infoTokens = self.stateVector[prevIndex:(prevIndex+numInfoTokens)]
        prevIndex += numInfoTokens

        numLifeTokens = 3
        self.lifeTokens = self.stateVector[prevIndex:(prevIndex + numLifeTokens)]
        prevIndex += numLifeTokens

        self.discardSpace = self.stateVector[prevIndex:(prevIndex + 50)]
        prevIndex += 50

        self.lastActivePlayer = self.stateVector[prevIndex:(prevIndex + self.playerCount)]
        prevIndex += self.playerCount

        self.lastMoveType = self.stateVector[prevIndex:(prevIndex + 4)]
        prevIndex += 4

        self.lastMoveTarget = self.stateVector[prevIndex:(prevIndex + self.playerCount)]
        prevIndex += self.playerCount

        self.colorRevealed = self.stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        self.rankRevealed = self.stateVector[prevIndex:(prevIndex + 5)]
        prevIndex += 5

        self.cardRevealed = self.stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        #in the context of the next two arrays, 'played' means played or discarded
        self.positionPlayed = self.stateVector[prevIndex:(prevIndex + handSize)]
        prevIndex += handSize

        self.cardPlayed = self.stateVector[prevIndex:(prevIndex + 25)]
        prevIndex += 25

        self.prevPlay = self.stateVector[prevIndex:(prevIndex + 2)]
        prevIndex += 2

        self.cardKnowledge = self.stateVector[prevIndex:(prevIndex + self.playerCount * handSize * 35)]

    def decrese_infoToken(self):
        # decrease the number of hint tokens
        for token in range(len(self.infoTokens)):
            if self.infoTokens[token] == 0:
                self.infoTokens[token] = 1
                break

    def increse_infoToken(self):
        # increase the number of hint tokens
        for token in range(len(self.infoTokens)):
            if self.infoTokens[token] == 1:
                self.infoTokens[token] = 0
                break

    def getStateVector(self):
        return self.stateVector

    def update_lastPlayer(self, player_id):
        """
        Updates the lastActivePlayer
        Args:
            player_id (int): the id of the player that just made a move
        """
        # FIXME: I don't think this is correct
        # update the last player
        for i in range(len(self.lastActivePlayer)):
            if i == player_id:  self.lastActivePlayer[i] = 1
            else:               self.lastActivePlayer[i] = 0

    def update_lastMove(self,move):
        # update the move type
        if move['action_type'] == 'REVEAL_COLOR':
            self.lastMoveType = [0,0,1,0]
        elif move['action_type'] == 'REVEAL_RANK':
            self.lastMoveType = [0,0,0,1]
        elif move['action_type'] == 'PLAY':
            self.lastMoveType = [1,0,0,0]
        elif move['action_type'] == 'DISCARD':
            self.lastMoveType = [0,1,0,0]
        else:
            raise ValueError("Invalid action_type")

        # color hinted
        try:    color = move['color']
        except: color = -1
        for c in range(len(self.colorRevealed)):
            if c == color:  self.colorRevealed[c] = 1
            else:           self.colorRevealed[c] = 0

        # rank hinted
        try:    rank = move['rank']
        except: rank = -1
        for r in range(len(self.rankRevealed)):
            if r == rank:   self.rankRevealed[r] = 1
            else:           self.rankRevealed[r] = 0
    
        # update the move target
        try:    target = move['target_offset']
        except: target = -1
        for t in range(len(self.lastMoveTarget)):
            if t == target: self.lastMoveTarget[t] = 1
            else:           self.lastMoveTarget[t] = 0        

