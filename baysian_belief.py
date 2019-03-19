import numpy as np
import pyhanabi


class Belief():

    def __init__(self, n_players):
        """
        n_players: (int) the number of players in the game
        """
        if n_players <= 3:
            self.n_cards = 5
        else:
            self.n_cards = 4
        self.n_players = n_players

        self.belief = np.zeros((n_players * self.n_cards + 1, 5, 5))
        self.discard_belief = np.zeros((5, 10))

    def reset(self):
        """
        this function sets all the matrices back to 0
        """
        self.belief[:, :, :] = 0
        self.discard_belief[:, :] = 0

    def insert(self, index, card):
        """
            index (int): which we are inserting
            card (object): the characteristics of the card to insert
        """
        color = card.color()
        rank = card.rank()

        # set the probability for that specific number to 1
        # because we know it exists
        self.belief[index, color, rank] = 1

    def insert_discard(self, card):
        """
            for the discard pile we don't need to know the specific order of the card
            and / or if there are duplicates
            this matrix is the exact knowledge of the discard pile
        """
        # extract color and rank of the card
        color = card.color()
        rank = card.rank()

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
        self.belief[index, color, :rank] = 1

    def calculate_prob(self, player, card_knowledge):
        """
        this function calculates the probability distribution of each card
        over the possible colors and possible ranks

        player (int): 
        card_knowledge (object)

        """
        # calculate the probability for each card
        rank_p = np.zeros(10)

        full_known = np.sum(self.belief, axis=0)
        availability = [3, 2, 2, 2, 1]
        index = player * self.n_cards

        for color in range(5):
            for rank in range(5):
                self.belief[index:index + self.n_cards, color,
                            rank] = availability[rank] - full_known[color, rank]

        # filter by hint
        for cid in range(self.n_cards):
            for color in range(5):
                my_card = card_knowledge[player][cid]

                # FIXME: the following line does not work
                if not my_card.color_plausible(color):
                    self.belief[cid + index, color, :] = 0

        # divide by total
        for card in self.belief[index:index + self.n_cards, :, :]:
            card[:, :] /= card.sum()

        print(self.belief[index:index + self.n_cards])

    def encode(self, observation, player):
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
        fireworks = observation.fireworks()
        for color in range(len(fireworks)):
            if fireworks[color] != 0:
                # insert in the last spot
                self.insert_many(self.n_cards *
                                 self.n_players, color, fireworks[color])

        # update the discard knowledge
        discard = observation.discard_pile()
        for card in discard:
            self.insert_discard(card)

        # update the observed hand knowledge
        observed = observation.observed_hands()
        for p in range(len(observed)):
            if p == player:
                continue
            else:
                for c in range(len(observed[p])):
                    card = observed[p][c]
                    self.insert(p * self.n_cards + c, card)

        self.calculate_prob(player, observation.card_knowledge())

        return self.belief
