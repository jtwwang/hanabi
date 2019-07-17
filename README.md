# A Cooperative Agent for Hanabi

This is a research project created by Lorenzo Mambretti, Justin Wang, Daniel Loran, Aron Sarmasi and Victoria Salova.

We provide a set of scripts, agents, and models that are used in the attempt to solve the ad-hoc challenge purposed by Nolan Bard et al. in the paper [The Hanabi Challenge: A New Frontier for AI Research](
https://arxiv.org/abs/1902.00506). These scripts are built on top of the hanabi\_learning\_environment provided by Google Deepmind.

If you use this code for your research, please cite:

A cooperative agent for Hanabi <br>
[Mambretti Lorenzo](https://github.com/LorenzoM1997), [Wang Justin](https://github.com/jtwwang). July 2019.

## Getting started
```
sudo apt-get install g++         # if you don't already have a CXX compiler
sudo apt-get install cmake       # if you don't already have CMake
sudo apt-get install python-pip  # if you don't already have pip
pip install cffi                 # if you don't already have cffi
pip install sklearn              # if you don't already have sklearn
pip install tensorflow           # if you don't already have tensorflow
pip install matplotlib           # if you don't already have matplotlib
cmake .
make
python game_example.py           # Plays a game using the lower level interface
```

## Running our scripts

### Data collection and evaluation agents
To run an arbitrary number of games between some of the existent agents and collect data you can use the script
```
python custom_rl.py --agent_class <nameAgent>
```
currently supports 8 classes:
- `MCAgent`
- `NewestCardAgent`
- `NNAgent`
- `ProbabilisticAgent`
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
--cross_play <True/False>   # cross_play between all agents
```
The script will print out to screen an average score and average number of steps taken during the episodes.

### Policy prediction
You can train a neural network to predict the policy by using
```
python run_pred.py
```
There are several flags that you can currently use:
```
--agent_class <str>	# Agent type ["SimpleAgent", "RainbowAgent"]
--balance <bool>	# Optional. Whether to make classes balanced
--batch_size <int>	# Batch size
--cv <int>		# Optional. Run cross-validation @cv number of times.
--epochs <int>		# Number of training epochs
--games <int>		# The number of games to load
--load <bool>		# Whether to laod an existing model (if exists)
--lr <float>		# Learning rate
--model_class <str>	# Network type ["dense", "conv", "lstm"]
--model_name <str>	# The name to give to the model
--summary <bool>	# Whether to print the summary of the model
--val_split <float>	# Proportion of data used to validate
```

Make sure there is training data in the folder `experience_replay` before starting training, or you might incurr into errors.

For the agents `RainbowAgent` and `SecondAgent` 5k episodes are already provided, and there are 20K available for `SimpleAgent`.
