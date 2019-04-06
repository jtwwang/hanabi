This is a research project created by Lorenzo Mambretti, Justin Wang, Daniel Loran, Aron Sarmasi and Victoria Salova.

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

## Getting started
```
sudo apt-get install g++         # if you don't already have a CXX compiler
sudo apt-get install cmake       # if you don't already have CMake
sudo apt-get install python-pip  # if you don't already have pip
pip install cffi                 # if you don't already have cffi
pip install sklearn              # if you don't already have sklearn
pip install tensorflow           # if you don't already have tensorflow
pip install keras                # if you don't already have keras
cmake .
make
python custom_rl.py              # Runs RL episodes
python game_example.py           # Plays a game using the lower level interface
```

## Running our scripts

### Data collection
To collect data you can use the script
```
python custom_rl.py --agent_class <nameAgent>
```
currently supports 4 classes:
- `RandomAgent`
- `SimpleAgent`
- `RainbowAgent`
- `MCAgent`

The data is saved in a folder automatically created called `/experience_replay`. Other flags you can use:
```
--num_episodes <int>
--players <int 2 to 5>
--debug <true/false>
```

### Policy prediction
After you collected episodes you can train a neural network to predict the policy by using
```
python policy_predictor.py
```
There are two flags that you can currently use:
```
--epochs <int>          # to set the number of epochs
--batch_size <int>      # to set the batch size
--lr <float>            # to set the learning rate
--agent_class <string>  # to choose the data for a specific agent class
--cv <int>		# to use cross validation with a specific number of folds
```

If not doing cross validation, the trainging uses all data available. In both cases a model is saved with the name `model/predictor.h5`. *Note*: run this script when you already have data in the folder `/experience_replay/<agent_class>`.

### Monte Carlo Player
We implemented an agent that uses Monte Carlo Tree search with UCT in combination with the policy predictor to play with other agents. The algorithm samples from a probability distribution every time that encounters an undertermined state. It uses 1000 simulations with finite depth to search the best move to do.

To run the script
```
ptyhon MonteCarlo_game.py
```
Similarly to `custom_rl.py` you can use flags to set some of the parameters of the game
```
--agent_class <RainbowAgent, RandomAgent, SimpleAgent, SecondAgent>
--num_episodes <int>
--players <int 2 to 5>
--verbose True/False
```
