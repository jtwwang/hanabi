## Observations
The observations are saved in two different ways, dictionary and vectorized form. While the dictionaries are easy to understand, the vectorized version is composed uniquely by 0s and 1s, and it is composed in a structured way. To better understand the vector we provide the following map of the vector.

Assuming `n` player game:
- hands 25 bits/card; 1 bit per player to show if player is missing a card
- boards 1 bit/ card in deck;
- 25 bits for board, 5 blocks of color
- 8 bits for info token
- 3 bits for life tokens
- 50 bits for discard, 5 blocks of color [1,1,1,2,2,3,3,4,4,5]
- last actions, n bits for active player, one-hot
- 4 bits for movetype, one-hot
- `n` bits for last move target, one-hot
- 5 bits for color revealed, one-hot
- 5 bits for rank revealed, one-hot
- `hand_size` bits for which card was hinted at, one-hot
- `hand_size` bits for which card was just played, one-hot
- 25 bits for card just played/discarded, one-hot
- `n` * `hand_size` * 35 bits for card knowledge
- 2 bits
