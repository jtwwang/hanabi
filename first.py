import numpy as np
import pyhanabi

game = pyhanabi.HanabiGame({"players": 3, "random_start_player": True})
obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

class Belief():

    def __init__(self, n_players):
        """
        n_players: (int) the number of players in the game
        """
        self.belief = np.zeros((n_players + 1, 5,10))
        self.my_belief = np.zeros((4,5,5))

    def reset(self):
        # reset everything except the fireworks and discard, which never change
        self.belief[2:,:,:] = 0
        self.my_belief = np.zeros((4,5,5))

    def is_valid_color(self,color):
        if color < 0 or color >= 5:
            return False
        else:
            return True

    def insert(self, index, card):
        
        # extract color and rank of the card
        color = card.color()
        rank = card.rank()

        # place the card in the correct spot
        if rank == 0 and not np.any(self.belief[0:index, color, 0]):
            self.belief[index, color, 0] = 1
        else:
            # place the card in the correct spot
            # card 2 goes on 3 or 4, card 3 goes on 5 or 6, etc...
            placement = 1 + (rank * 2)
            if np.any(self.belief[0:index, color, placement]):
                placement += 1
            self.belief[index, color, placement] = 1

    def insert_many(self, index, color, rank):

        if not self.is_valid_color(color):
            print("ERROR: INVALID COLOR")
            return

        if rank == 1 and not np.any(self.belief[:2, color,0]):
            self.belief[index, color,0] = 1
        elif rank >= 1 and rank <= 5:
            placement = 1 + ((rank - 1) * 2)
            if np.any(self.belief[:1, color, placement]):
                placement += 1
            self.belief[index, color, placement] = 1
        else:
            print("ERROR: INVALID CARD NUMBER {}".format(rank))

    def calculate_prob(self, index):
        # calculate the probability for each card
        rank_p = np.zeros(10)

        full_known = 1 - np.sum(self.belief, axis = 0)
        
        for color in range(5):
            self.my_belief [:, color, 0] = np.sum(full_known[color, 0:3])
            self.my_belief [:, color, 1] = np.sum(full_known[color, 3:5])
            self.my_belief [:, color, 2] = np.sum(full_known[color, 5:7])
            self.my_belief [:, color, 3] = np.sum(full_known[color, 7:9])
            self.my_belief [:, color, 4] = full_known[color, 9]

        # filter by hint

        # divide by total
        for card in self.my_belief:
            card[:,:] /= card.sum()

        print(self.my_belief)
            

belief = Belief(3)

def encoded(observation, player):
    global belief
    belief.reset()
    fireworks = observation.fireworks()
    for color in range(len(fireworks)):
        if fireworks[color] != 0:
            belief.insert_many(0,color, fireworks[color])

    discard = observation.discard_pile()
    for card in discard:
        belief.insert(1, card)

    observed = observation.observed_hands()
    for p in range(len(observed)):
        if p == player:
            continue
        else:
            for card in observed[p]:
                belief.insert(1 + p, card)

    belief.calculate_prob(2 + player)
    print(observation.card_knowledge())
        

state = game.new_initial_state()
while not state.is_terminal():
    if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      state.deal_random_card()
      continue

    observation = state.observation(state.cur_player())
  
    encoded(observation, 0)

    legal_moves = state.legal_moves()
    print("Number of legal moves: {}".format(len(legal_moves)))

    move = np.random.choice(legal_moves)
    print("Chose random legal move: {}".format(move))

    state.apply_move(move)
