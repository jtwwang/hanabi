import numpy as np
import pyhanabi
import baysian_belief

n_players = 3

belief = baysian_belief.Belief(n_players)

game = pyhanabi.HanabiGame({
    "players": n_players,
    "random_start_player": True})
obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)       

state = game.new_initial_state()
while not state.is_terminal():
    if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      state.deal_random_card()
      continue

    observation = state.observation(state.cur_player())
  
    belief.encode(observation, 0)

    legal_moves = state.legal_moves()
    print("Number of legal moves: {}".format(len(legal_moves)))

    move = np.random.choice(legal_moves)
    print("Chose random legal move: {}".format(move))

    state.apply_move(move)
