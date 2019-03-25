import numpy as np
import pyhanabi


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

    def insert_discard(self, card):
        """
            for the discard pile we don't need to know the specific order of the card
            and / or if there are duplicates
            this matrix is the exact knowledge of the discard pile
        """
        # extract color and rank of the card
        color = self.color_map[card['color']]
        rank = card['rank']

        # place the card in the correct spot
        if rank == 0 and not np.any(self.belief[:, color, 0]):
            self.discard_belief[color, 0] = 1
        else:
            # place the card in the correct spot
            # card 2 goes on 3 or 4, card 3 goes on 5 or 6, etc...
            placement = 1 + (rank * 2)
            if np.any(self.discard_belief[color, placement]):
                placement += 1
            self.discard_belief[color, placement] = 1

    def insert_many(self, index, color, rank):
        # all cards until the rank become 1
        color = self.color_map[color]
        self.belief[index, color, :rank] = 1

    def calculate_prob(self, player_id, obs):
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
        ix = len(obs) - (space_knowledge * (self.players - player_id))
        from_vec = obs[ix:ix + 35 * self.n_cards]
        mask = np.empty((self.n_cards,5,5), dtype=np.uint8)
        for i in range(self.n_cards):
            mask[i] = np.reshape(from_vec[i*35: i*35 + 25], (5,5))

        self.belief[index: index + self.n_cards] = np.multiply(self.belief[index:index + self.n_cards], mask)

        # divide by total
        for card in self.belief[:self.n_cards]:
            card /= card.sum()


    def encode(self, obs, player_id):
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

        # update the firework knowledge
        fireworks = obs['fireworks']
        for color in fireworks:
            if fireworks[color] != 0:
                # insert in the last spot
                self.insert_many(self.n_cards *
                                 self.players, color, fireworks[color])

        # update the discard knowledge
        discard = obs['discard_pile']
        for card in discard:
            self.insert_discard(card)

        # update the observed hand knowledge
        ob = obs['vectorized']
        from_vec = ob[:25*self.n_cards*(self.players - 1)]
        mask = np.empty((self.n_cards,5,5), dtype=np.uint8)
        for i in range(self.n_cards * (self.players - 1)):
            self.belief[i + self.n_cards] = np.reshape(from_vec[i*25:(i+1)*25], (5,5))

        self.calculate_prob(player_id, ob)

        return self.belief[:self.n_cards]
