import numpy as np
import rl_env
import os

class Experience():

    path = "experience_replay"

    def __init__(self, numAgents, agent_class, size = 1000000):
        """
        args:
            numAgents (int)
            size (int) (optional)
        """

        self.size = size
        self.ptr = 0
        self.full = False
        self.path = os.path.join(self.path, agent_class)
        
        try:
            # detect the size of the observations
            env = rl_env.make(num_players = numAgents)
            obs = env.reset()
            size_obs = len(obs['player_observations'][0]['vectorized'])

            # detect the size of move
            self.n_moves = env.num_moves()

            # initialize matrices for all values
            self.moves = np.empty(size, dtype = np.uint8)
            self.rs = np.empty(size)
            self.obs = np.empty((size, size_obs), dtype = bool)
        except:
            # if the environment can't be create, we still can load
            if numAgents == 2 or numAgents == 3:
                self.n_cards = 5
            elif numAgents == 4 or numAgents == 4:
                self.n_cards = 4
            else:
                print("ERROR: invalid number of players")
                return

            self.n_moves = numAgents * 10 + self.n_cards * 2

            print("WARNING: the environment could not be created, some \
                    functionality may be compromised. You CAN still load \
                    data.")


    def add(self, ob, reward, move):
        """
        args:
            ob: the observation from the current player
            reward (int): the reward obtain at this step
            move (int): the action performed by the agent
        """

        self.moves[self.ptr] = move
        self.obs[self.ptr,:] = ob['vectorized']
        self.rs[self.ptr] = reward

        if self.ptr == self.size - 1:
            # set the flag to true if we reached the end of the matrix
            self.full = True
        self.ptr = (self.ptr + 1) % self.size # increment pointer
 

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
        
        # pack bits of observations for compression
        packed_obs = np.packbits(self.obs[:index,:], axis = 1)
                        
        # save to file
        np.save(os.path.join(self.path, "obs"), packed_obs)
        np.save(os.path.join(self.path, "rewards"), self.rs[:index])
        np.save(os.path.join(self.path, "moves"), self.moves[:index])

    def load(self):
        """
        load all the data from numpy files previously saved

        returns [moves, rs, obs]
            numpy matrices with data
        """

        packed_obs = np.load(os.path.join(self.path, "obs.npy"))

        self.moves = np.load(os.path.join(self.path, "moves.npy"))
        self.obs = np.unpackbits(packed_obs, axis = 1)
        self.rs = np.load(os.path.join(self.path, "rewards.npy"))

        self.ptr = len(self.moves)
        if self.ptr == self.size - 1:
            self.full = True
       
        return [self.moves, self.rs, self.obs]

    def _obs(self):
        """
        returns the observations
        """

        if self.full:
            index = self.size
        else:
            index = self.ptr

        return self.obs[:index,:]

    def _one_hot_moves(self):
        """
        returns a one-hot encoded vector with the moves in the experience replay

        a: numpy matrix (size, n_moves)
        """

        if self.full:
            index = self.size
        else:
            index = self.ptr

        # create one-hot encoding
        a = np.zeros((index, self.n_moves))
        a[np.arange(index), self.moves[:index]] = 1

        return a

