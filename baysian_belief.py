import numpy as np
import pyhanabi
from state_translate import state_translator

class Belief():

    color_map = {
        'Y': 0,
        'B': 1,
        'R': 2,
        'W': 3,
        'G': 4
    }

    def __init__(self, players):
        """
        players: (int) the number of players in the game
        """
        if players <= 3:
            self.n_cards = 5
        else:
            self.n_cards = 4
        self.players = players

        self.belief = np.zeros((players * self.n_cards + 1, 5, 5))
        self.discard_belief = np.zeros((5, 10))

    def reset(self):
        """
        this function sets all the matrices back to 0
        """
        self.belief[:, :, :] = 0
        self.discard_belief[:, :] = 0
    
    def calculate_prob(self, player_id, vectorized):
        """
        this function calculates the probability distribution of each card
        over the possible colors and possible ranks

        player (int): 
        card_knowledge (object)

        """
        full_known = np.sum(self.belief, axis=0)
        availability = [3, 2, 2, 2, 1]
        index = player_id * self.n_cards

        # count the full availability of cards
        for color in range(5):
            for rank in range(5):
                self.belief[:self.n_cards, color, rank] = availability[rank] - full_known[color, rank]

        # filter by hint
        space_knowledge = 35 * self.n_cards
        ix = len(vectorized) - (space_knowledge * (self.players - player_id))
        from_vec = vectorized[ix:ix + 35 * self.n_cards]
        mask = np.empty((self.n_cards,5,5), dtype=np.uint8)
        for i in range(self.n_cards):
            mask[i] = np.reshape(from_vec[i*35: i*35 + 25], (5,5))

        self.belief[index: index + self.n_cards] = np.multiply(self.belief[index:index + self.n_cards], mask)

        # divide by total
        for card in self.belief[:self.n_cards]:
            card /= card.sum()


    def encode(self, vectorized, player_id):
        """
        Encodes the observations and the knowledge into the belief
            player (int) 
            observation

        returns:
            belief matrix
        """

        # since an action may have been performed, we delete all the previous belief we had
        # of all non-common cards (fireworks and discard)
        self.reset()

        translated = state_translator(vectorized, self.players)

        # update the fireworks knowledge
        fireworks = np.asarray(translated.boardSpace)
        fireworks = np.reshape(fireworks, (5,5))
        index = self.n_cards * self.players
        self.belief[index] = fireworks

        # update the discard knowledge
        discard = np.asarray(translated.discardSpace)
        discard = np.reshape(discard, (5,10))
        self.discard_belief = discard

        # update the observed hand knowledge
        from_vec = vectorized[:25*self.n_cards*(self.players - 1)]
        mask = np.empty((self.n_cards,5,5), dtype=np.uint8)
        for i in range(self.n_cards * (self.players - 1)):
            self.belief[i + self.n_cards] = np.reshape(from_vec[i*25:(i+1)*25], (5,5))

        self.calculate_prob(player_id, vectorized)

        return self.belief[:self.n_cards]
