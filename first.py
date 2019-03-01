import numpy as np
import pyhanabi

game = pyhanabi.HanabiGame({"players": 3, "random_start_player": True})
obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

class Belief():

    def __init__(self, n_players):
        self.belief = np.zeros((n_players + 2, 5,10))

    def is_valid_color(self,color):
        if color < 0 or color >= 5:
            return False
        else:
            return True

    def insert(self, index, color, number):

        if not self.is_valid_color(color):
            print("ERROR: INVALID COLOR")
            return

        if number == 1:
            if self.belief[index, color,0] != 1:   self.belief[index, color,0] = 1
            elif self.belief[index, color,1] != 1: self.belief[index, color,1] = 1
            else: self.belief[index, color,2] = 1
        elif number == 2:
            if self.belief[index, color,3] != 1: self.belief[index,color,3] = 1
            else: self.belief[index, color,4] = 1
        elif number == 3:
            if self.belief[index, color,5] != 1: self.belief[index, color,5] = 1
            else: self.belief[index, color,6] = 1
        elif number == 4:
            if self.belief[index,color,7] != 1: self.belief[index,color,7] = 1
            else: self.belief[index,color,8] = 1
        elif number == 5:
            if self.belief[index,color,9] != 1: self.belief[index,color,9] = 1
            else: print("Some error has occurred, the position has already been filled")
        else:
            print("ERROR: INVALID CARD NUMBER")
            

belief = Belief(3)

def encoded(observation, player):
    global belief
    fireworks = observation.fireworks()
    for color in range(len(fireworks)):
        if fireworks[color] != 0:
            belief.insert(0,color, fireworks[color])

    discard = observation.discard_pile()
    for card in range(len(discard)):
        belief.insert(1, discard[card].color(), discard[card].rank() + 1)

    observed = observation.observed_hands()
    for p in range(len(observed)):
        if p == player:
            continue
        else:
            for cards in observed[p]:
                print(cards)

    print(belief.belief)
        

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
