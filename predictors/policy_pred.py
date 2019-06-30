from __future__ import print_function
from data_pipeline.experience import Experience
from data_pipeline.balance_data import balance_data
from data_pipeline.util import one_hot
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import os

# shut up info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class EvalAcc(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("acc: {} - val_acc: {} - loss: {} - eval_loss: {}".format(
            logs.get('acc'), logs.get('val_acc'), logs.get('loss'), logs.get('val_loss')))


class policy_pred(object):
    def __init__(self, agent_class, model_type=None):
        self.X = None  # nn input
        self.y = None  # nn output
        self.input_dim = None
        self.action_space = None

        self.model = None

        # Create the directory for this particular model
        self.path = os.path.join("model", agent_class)
        if model_type is not None:
            self.path = os.path.join(self.path, model_type)
            self.make_dir(self.path)
            print("Writing to", self.path)
        else:
            print(self.path, "already exists. Writing to it.")
        self.checkpoint_path = os.path.join(self.path, "checkpoints")

    def make_dir(self, path):
        # Create the directory @path if it doesn't exist already
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % self.path)
            else:
                print("Successfully created the directory %s" % self.path)

    def reshape_data(self, X):
        pass

    def create_model(self):
        # Override
        pass

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):
        """
        args:
                val_split(float, between 0 and 1): Fraction of the data to be used as validation data
        """
        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)

        self.create_model()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam, metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir=self.path)

        self.model.fit(
            self.X,
            self.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=[tensorboard],
            shuffle=True
        )

    def predict(self, X):
        """
        args:
                X (input)
        returns:
                prediction given the model and the input X
        """
        X = self.reshape_data(X)
        self.create_model()
        pred = self.model.predict(X)
        return pred

    def save(self, model_name="predictor.h5"):

        self.make_dir(self.path)
        model_path = os.path.join(self.path, model_name)
        self.model.save(model_path)

    def load(self, model_name="predictor.h5"):
        """
        function to load the saved model
        """
        try:
            self.model = load_model(os.path.join(self.path, model_name))
        except:
            print("Create new model")

    def extract_data(self, agent_class, val_split=0.3, games=-1, balance=False):
        """
        args:
                agent_class (string): "SimpleAgent", "RainbowAgent"
                num_games (int): how many games we want to load
        """
        print("Loading Data...", end='')
        replay = Experience(agent_class, load=True)
        moves, _, obs, eps = replay.load(games=games)

        assert obs.shape[0] == moves.shape[0]

        # split dataset here
        size_test = int(val_split * obs.shape[0])
        X_test, X_train = obs[:size_test], obs[size_test:]
        y_test, y_train = moves[:size_test], moves[size_test:]

        if balance:
            # make class balanced
            X_train, y_train = balance_data(X_train, y_train)

        print(y_train.shape)

        # conver to one-hot encoded tensor
        y_train = one_hot(y_train, replay.n_moves)
        y_test = one_hot(y_test, replay.n_moves)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        print("Experience Loaded!")
        return X_train, y_train, eps

    def define_model_dim(self, input_dim, action_space):
        self.input_dim = input_dim
        self.action_space = action_space
