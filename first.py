import numpy as np
import pyhanabi

game = pyhanabi.HanabiGame({"players": 3, "random_start_player": True})
obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

class Belief():

    n_cards = 5

    def __init__(self, n_players):
        """
        n_players: (int) the number of players in the game
        """
        if n_players <= 3:
            self.n_cards = 5
        else:
            self.n_cards = 4
        self.n_players = n_players
    
        self.belief = np.zeros((n_players * self.n_cards + 1, 5,5))
        self.discard_belief = np.zeros((5,10))

    def reset(self):
        # reset everything to 0
        self.belief[:,:,:] = 0
        self.discard_belief[:,:] = 0

    def insert(self, index, card):
        # extract color and rank of the card
        color = card.color()
        rank = card.rank()

        self.belief[index, color, rank] = 1

    def insert_discard(self, card):
        
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
        # calculate the probability for each card
        rank_p = np.zeros(10)

        full_known = np.sum(self.belief, axis = 0)
        availability = [3,2,2,2,1]
        index = player * self.n_cards

        for color in range(5):
            for rank in range(5):
                self.belief [index:index + self.n_cards, color, rank] = availability[rank] - full_known[color, rank]

        # filter by hint
        for cid in range(self.n_cards):
            for color in range(5):
                my_card = card_knowledge[player][cid]
                print(my_card)
                if not my_card.color_plausible(color):
                    self.belief[index: index + self.n_cards, color, :] = 0

        # divide by total
        for card in self.belief[index:index + self.n_cards,:,:]:
            card[:,:] /= card.sum()

belief = Belief(3)

def encode(observation, player):
    """
    Encodes the observations and the knowledge into the belief
    player (int) 
    """

    global belief

    # since an action may have been performed, we delete all the previous belief we had
    # of all non-common cards (fireworks and discard)
    belief.reset()

    # update the firework knowledge
    fireworks = observation.fireworks()
    for color in range(len(fireworks)):
        if fireworks[color] != 0:
            # insert in the last spot
            belief.insert_many(belief.n_cards * belief.n_players, color, fireworks[color])

    # update the discard knowledge
    discard = observation.discard_pile()
    for card in discard:
        belief.insert_discard(card)

    # update the observed hand knowledge
    observed = observation.observed_hands()
    for p in range(len(observed)):
        if p == player:
            continue
        else:
            for c in range(len(observed[p])):
                card = observed[p][c]
                belief.insert(p * belief.n_cards + c, card)

    belief.calculate_prob(player, observation.card_knowledge())
    print(belief.belief)
        

state = game.new_initial_state()
while not state.is_terminal():
    if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      state.deal_random_card()
      continue

    observation = state.observation(state.cur_player())
  
    encode(observation, 0)

    legal_moves = state.legal_moves()
    print("Number of legal moves: {}".format(len(legal_moves)))

    move = np.random.choice(legal_moves)
    print("Chose random legal move: {}".format(move))

    state.apply_move(move)
