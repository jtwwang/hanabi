import numpy as np
from state_translate import state_translator


class Belief():

    def __init__(self, players):
        """
        Args:
            players: (int) the number of players in the game
        """
        if players <= 3:
            self.n_cards = 5
        else:
            self.n_cards = 4
        self.players = players

        # initialize matrix for known cards
        self.full_know = np.zeros(25, dtype=np.uint8)

        # set the matrix of available cards
        self.build_availability()

    def build_availability(self):
        """
        Create the array that stores how many cards are available at the beginning of the game
        This function is usually called just once at the beginning
        """

        self.available = np.empty(25, dtype=np.uint8)
        av = np.array([3, 2, 2, 2, 1], dtype=np.uint8)
        for color in range(5):
            self.available[color * 5: (color + 1) * 5] = av

    def add_known(self, card):
        """
        Add the information of one card on the full knowledge
        Args:
            card (list): information of a one-hot encoded card
        """
        self.full_know -= card

    def prob(self, my_hints):
        """
        this function calculates the probability distribution of a card
        over the possible colors and possible ranks
        Args:
            my_hints (list): the hint I received about that card
        returns:
            my_belief(np.array, float32): the probability distirbution over that card
        """

        # filter by hint
        my_belief = np.multiply(self.full_know, my_hints, dtype=np.float32)

        # divide by the number of remaining cards
        my_belief /= my_belief.sum()

        return my_belief

    def encode(self, vectorized):
        """
        Encodes the observations and the knowledge into the belief
        Args:
            vectorized(list): the observations in a vectorized form
        """

        # since an action may have been performed, we delete all the previous
        # belief we had
        self.full_know[:] = self.available

        tr = state_translator(vectorized, self.players)

        # update the fireworks knowledge
        self.full_know -= np.asarray(tr.boardSpace, dtype=np.uint8)

        # update the discard knowledge
        discard = np.asarray(tr.discardSpace)
        for c in range(5):
            color = discard[c * 10: (c + 1) * 10]
            self.full_know[c * 5: (c + 1) * 5][0] -= color[0] + \
                color[1] + color[2]
            self.full_know[c * 5: (c + 1) * 5][1] -= color[3] + color[4]
            self.full_know[c * 5: (c + 1) * 5][2] -= color[5] + color[6]
            self.full_know[c * 5: (c + 1) * 5][3] -= color[7] + color[8]
            self.full_know[c * 5: (c + 1) * 5][4] -= color[9]

        # update the observed hand knowledge
        for i in range(self.n_cards * (self.players - 1)):
            self.full_know -= np.asarray(
                tr.handSpace[i * 25:(i + 1) * 25], dtype=np.uint8)
