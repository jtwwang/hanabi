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
pip install matplotlib           # if you don't already have matplotlib
cmake .
make
python custom_rl.py              # Runs RL episodes
python game_example.py           # Plays a game using the lower level interface
```

## Running our scripts

### Data collection and evaluation agents
To collect data you can use the script
```
python custom_rl.py --agent_class <nameAgent>
```
currently supports 6 classes:
- `MCAgent`
- `NNAgent`
- `RainbowAgent`
- `RandomAgent`
- `SecondAgent`
- `SimpleAgent`

The data is saved in a folder automatically created called `/experience_replay`. Other flags you can use:
```
--num_episodes <int>
--players <int 2 to 5>
--debug <true/false>
--agent_predicted <str>     # necessary if using MCAgent or NNAgent. Use one of the other classes as string
--model_class <str>         # Network type ["dense", "conv", "lstm"]
--model_name <str>          # Model name of a pre-trained model
--agent2 <str>              # to play 'ad hoc' against another agent. 
                            # The second agent cannot be one of thetwo customs agents ["MCAgent", "NNAgent"]
--checkpoint_dir <str>      # path to the checkpoints for RainbowAgent
--checkpoint_dir2 <str>     # path to the checkpoints for RainbowAgent as agent2
```
The script will print out to screen an average score and average number of steps taken during the episodes.

### Policy prediction
You can train a neural network to predict the policy by using
```
python run_pred.py
```
There are several flags that you can currently use:
```
--agent_class <str      # Agent type ["SimpleAgent", "RainbowAgent"]
--batch_size <int>      # Batch size
--cv <int>				# Optional. Run cross-validation @cv number of times.
--epochs <int>          # Number of training epochs
--games <int>           # The number of games to load
--load <bool>           # Whether to laod an existing model (if exists)
--lr <float>            # Learning rate
--model_class <str>		# Network type ["dense", "conv", "lstm"]
--model_name <str>      # The name to give to the model
--summary <bool>        # Whether to print the summary of the model
--val_split <float>		# Proportion of data used to validate
```

Make sure there is training data in the folder `experience_replay` before starting training, or you might incurr into errors. For the agents `RainbowAgent` are already available 5K episodes of memory, and there are 20K available for `SimpleAgent`.
