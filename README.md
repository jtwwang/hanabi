This is a research project created by Lorenzo Mambretti, Justin Wang, Daniel Loran, Aron Sarmasi and Victoria Salova.

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

## Getting started
```
sudo apt-get install g++         # if you don't already have a CXX compiler
sudo apt-get install cmake       # if you don't already have CMake
sudo apt-get install python-pip  # if you don't already have pip
pip install cffi                 # if you don't already have cffi
cmake .
make
python rl_env_example.py         # Runs RL episodes
python game_example.py           # Plays a game using the lower level interface
```

## Running our scripts

### Data collection
To collect data you can use the script
```
python custom_rl_example.py --agent_class <nameAgent>
```
currently supports 3 classes: `RandomAgent`, `SimpleAgent`, and `RainbowAgent`.The data is saved in a folder automatically created called `/experience_replay`. Other flags you can use:
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
--epochs <int>      # to specify the number of epochs
--batch_size <int>  # to specify the batch size
```
Note: run this script when you already have data in the folder `/experience_replay`.
