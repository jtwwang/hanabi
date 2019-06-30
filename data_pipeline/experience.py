from data_pipeline.util import one_hot
import rl_env
import os
import sys
import numpy as np
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Experience():

    path = "experience_replay"

    def __init__(self, agent_class, numAgents=-1, load=False, size=1000000):
        """
        Args:
            agent_class (string): the class of the agent, which can be one of:
                - 'SimpleAgent'
                - 'RainbowAgent'
                - 'RandomAgent'
            numAgents (int, optional): the number of agents
            load (boolean, optional): whether we have to load possible
                existent data of the given class of agents.
            size (int, optional): how many steps are going to be saved,
                default is 100K. This size is used to allocate memory at the beginning
        """

        self.size = size
        self.ptr = 0
        self.full = False
        self.path = os.path.join(self.path, agent_class)

        if not load and numAgents == -1:
            print(
                "Bad parameter initialization. Use either 'numAgents' or 'load' to initialize the object.")
            exit()
        else:
            if load:
                # load the configurations from file
                self.config = pickle.load(
                    open(os.path.join(self.path, "config.pickle"), "rb"))
                numAgents = self.config["numAgents"]

            else:
                self.config = {}                        # create empty dict
                self.config["numAgents"] = numAgents    # insert config data

        try:
            # detect the size of the observations
            env = rl_env.make(num_players=numAgents)
            obs = env.reset()
            self.config["size_obs"] = len(
                obs['player_observations'][0]['vectorized'])

            # detect the size of move
            self.n_moves = env.num_moves()

            # initialize matrices for all values
            self.moves = np.empty(size, dtype=np.uint8)
            self.rs = np.empty(size)
            self.obs = np.empty((size, self.config["size_obs"]), dtype=bool)
            self.eps = []

            # initialize last episode
            self.last_ep = -1
        except BaseException:
            # if the environment can't be create, we still can load
            if numAgents == 2 or numAgents == 3:
                self.n_cards = 5
            elif numAgents == 4 or numAgents == 4:
                self.n_cards = 4
            else:
                print("ERROR: invalid number of players")
                return

            self.n_moves = numAgents * 10 + self.n_cards * 2

            print("WARNING: the environment could not be created.")
            print("Some functionality may be compromised. You CAN still load data.")

    def update_ep(self, eps):
        if eps != self.last_ep:
            if self.last_ep != -1:
                # append to the list the extreme points of the last episode
                self.eps.append((self.ep_start_id, self.ptr))

            # save the id of the new episode
            self.ep_start_id = self.ptr

        # update which episode we are in
        self.last_ep = eps

    def add(self, ob, reward, move, eps):
        """
        args:
            ob: the observation from the current player
            reward (int): the reward obtain at this step
            move (int): the action performed by the agent
            eps (int): the number of the episode
        """

        self.moves[self.ptr] = move
        self.obs[self.ptr, :] = ob['vectorized']
        self.rs[self.ptr] = reward

        self.update_ep(eps)

        if self.ptr == self.size - 1:
            # set the flag to true if we reached the end of the matrix
            self.full = True
        self.ptr = (self.ptr + 1) % self.size  # increment pointer

    def save(self):

        # if it doesn't exists, create directory
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s" % self.path)

        if self.full:
            index = self.size
        else:
            index = self.ptr

        # update episodes
        self.update_ep(self.last_ep + 1)

        # pack bits of observations for compression
        packed_obs = np.packbits(self.obs[:index, :], axis=1)

        # save to file
        np.save(os.path.join(self.path, "obs"), packed_obs)
        np.save(os.path.join(self.path, "rewards"), self.rs[:index])
        np.save(os.path.join(self.path, "moves"), self.moves[:index])
        np.save(os.path.join(self.path, "eps"), self.eps[:index])

        # pickle the configurations
        pickle.dump(self.config, open(
            os.path.join(self.path, "config.pickle"), "wb"))

    def load(self, games=-1):
        """
        load all the data from numpy files previously saved
        returns [moves, rs, obs, eps]
            numpy matrices with data
        """

        packed_obs = np.load(os.path.join(self.path, "obs.npy"))

        self.moves = np.load(os.path.join(self.path, "moves.npy"))
        self.obs = np.unpackbits(packed_obs, axis=1)
        self.rs = np.load(os.path.join(self.path, "rewards.npy"))
        self.eps = np.load(os.path.join(self.path, "eps.npy"))

        # restore pointer to end
        self.ptr = len(self.moves)
        if self.ptr == self.size - 1:
            self.full = True

        # restore size observation
        size_obs = self.config["size_obs"]
        self.obs = self.obs[:, :size_obs]

        if games > 0:
            last = self.eps[games - 1][1]
            self.moves = self.moves[:last]
            self.rs = self.rs[:last]
            self.obs = self.obs[:last]
            self.eps = self.eps[:games]
            if last + 1 < self.size:
                self.ptr = last
                self.full = False

        return [self.moves, self.rs, self.obs, self.eps]

    def _obs(self):
        """
        Returns the observations in a numpy matrix
        """

        if self.full:
            index = self.size
        else:
            index = self.ptr

        return self.obs[:index, :]

    def _one_hot_moves(self):
        """
        Returns a one-hot encoded vector with the moves in the experience
        replay. The output is in a 1-D numpy vector of type uint8
        """

        if self.full:
            index = self.size
        else:
            index = self.ptr

        a = self.moves[:index]

        # create one-hot encoding
        return one_hot(a, self.n_moves)
