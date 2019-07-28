# A Cooperative Agent for Hanabi

This is a research project directed by Lorenzo Mambretti and Justin Wang, with the contributions of Daniel Loran, Aron Sarmasi and Victoria Salova.

We provide a set of scripts, agents, and models that are used in the attempt to solve the ad-hoc challenge proposed by Nolan Bard et al. in the paper [The Hanabi Challenge: A New Frontier for AI Research](
https://arxiv.org/abs/1902.00506). These scripts are built on top of the hanabi\_learning\_environment provided by Google Deepmind.

If you use this code for your research, please cite:

A cooperative agent for Hanabi <br>
[Mambretti Lorenzo](https://github.com/LorenzoM1997), [Wang Justin](https://github.com/jtwwang). July 2019.

## Getting started
### Install git lfs
Before cloning the repository, you need to install **git lfs**. This will allows you to download some large files that we provide correctly. Specifically, you will have access to the replay memory for the provided agents. If you do not need this, skip this step and proceed with next section

```
sudo apt-get install git-lfs
git lfs install
```
For more instructions, please refer to [https://github.com/git-lfs/git-lfs/wiki/Installation](https://github.com/git-lfs/git-lfs/wiki/Installation)
### Clone the repository
```
git clone https://github.com/jtwwang/hanabi.git
```
### Install dependencies
```
sudo apt-get install g++         # if you don't already have a CXX compiler
sudo apt-get install cmake       # if you don't already have CMake
sudo apt-get install python-pip  # if you don't already have pip
pip install cffi                 # if you don't already have cffi
pip install tensorflow           # if you don't already have tensorflow
pip install matplotlib           # if you don't already have matplotlib
cmake .
make
python2 game_example.py           # Plays a game using the lower level interface
```

## Usage

### Data collection and evaluation agents
To run an arbitrary number of games between some of the existent agents and collect data you can use the script
```
python2 run_simulations.py --agent_class <nameAgent>
```
currently supports 9 classes:
- `MCAgent`
- `NeuroEvoAgent`
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
--agent2_class		    # to play 'ad hoc' against another agent. 
--agent_predicted <str>     # necessary if using MCAgent or NNAgent. Use one of the other classes as string
--agent_predicted2 <str>    # necessary if using MCAgent or NNAgent for agent2
--model_class <str>         # Network type ["dense", "conv", "lstm"]
--model_class2 <str>	    # Network type for agent2
--model_name <str>          # Model name of a pre-trained model
--model_name2 <str>	    # Model name of a pre-trained model for agent2
--checkpoint_dir <str>      # path to the checkpoints for RainbowAgent
--checkpoint_dir2 <str>     # path to the checkpoints for RainbowAgent as agent2
--cross_play <True/False>   # cross_play between all agents
```
The script will print out to screen an average score and average number of steps taken during the episodes.

### Policy prediction
You can train a neural network to predict the policy by using
```
python2 run_pred.py
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

## Experience
We provide 20k of experience for the agents `RainbowAgent`, `SimpleAgent`, `SecondAgent` and `ProbabilisticAgent`. They can be immediately used to train a predictor. All files are saved in .npy format and thus can be opened with numpy if necessary.

The experience is managed by the class Experience, that is called during the simulations in `run_simulations.py` to save the observations, moves, and rewards. We call again the class in `run_pred.py` to access the experience memory.

## Neuro-Evolution Agents
We provide a script to train agents based on genetic algorithm to evolve CNN to play the game of hanabi.
The development of this process is still undergoing development.

```
python2 neuroEvo_train.py
```
You can use the following flags:
```
--players <int>		   # the number of players
--num_episodes <int>	   # number of episodes
--initialize <True/False>  # whether to re-initialize the weights of all agents
--models <int>		   # how many specimens in the simulations
--generations <int>	   # how many generations to run
```