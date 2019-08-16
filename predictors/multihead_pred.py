# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied

from __future__ import print_function
from data_pipeline.experience import Experience
from data_pipeline.util import one_hot, split_dataset
from .policy_pred import policy_pred

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras import optimizers

from .blocks import conv_block

import numpy as np

import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v1 import variable_scope, get_variable, reset_default_graph
from tensorflow.compat.v1.losses import softmax_cross_entropy
from tensorflow.compat.v1.layers import Flatten
import math
import time


class MultiHeadModel(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def conv_relu(self, inputs, kernel_shape, stride):
        # Create variable named "weights"
        weights = get_variable("weights", kernel_shape,
                               initializer=tf.random_normal_initializer())
        # Create variable named "biases"
        bias_shape = kernel_shape[-1]
        biases = get_variable("biases", [bias_shape],
                              initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv1d(inputs, weights,
                            stride=stride, padding='SAME')

        # batch normalization TODO: seperate into func
        mean, variance = tf.nn.moments(conv, axes=[0])
        scale = tf.Variable(tf.ones([bias_shape]), name="scale")
        beta = tf.Variable(tf.zeros([bias_shape]), name="beta")
        epsilon = 1e-3
        bn = tf.nn.batch_normalization(
            conv, mean, variance, scale, beta, epsilon)
        return tf.nn.relu(bn + biases)

    def dense(self, inputs, units):
        # Create vriable named "weights"
        dense_shape = [inputs.shape[-1], units]
        weights = get_variable("weights", dense_shape,
                               initializer=tf.random_normal_initializer())
        # Create variable named "baises"
        biases = get_variable("biases", [units],
                              initializer=tf.constant_initializer(0.0))
        y = tf.linalg.matmul(inputs, weights) + biases
        return y

    def __call__(self, x):

        with variable_scope("conv1", reuse=tf.AUTO_REUSE):
            relu1 = self.conv_relu(x, [5, 1, 16], stride=2)
            maxp1 = tf.nn.max_pool1d(relu1, 3, 2, 'SAME')
        with variable_scope("conv2", reuse=tf.AUTO_REUSE):
            relu2 = self.conv_relu(maxp1, [3, 16, 32], stride=2)
            maxp2 = tf.nn.max_pool1d(relu2, 2, 2, 'SAME')
        with variable_scope("conv3", reuse=tf.AUTO_REUSE):
            relu3 = self.conv_relu(maxp2, [3, 32, 64], stride=2)
            maxp3 = tf.nn.max_pool1d(relu3, 2, 2, 'SAME')
        with variable_scope("conv4", reuse=tf.AUTO_REUSE):
            relu4 = self.conv_relu(maxp3, [3, 64, 64], stride=2)
            maxp4 = tf.nn.max_pool1d(relu4, 2, 2, 'SAME')
        with variable_scope("flatten"):
            flattened = Flatten()(maxp4)
        with variable_scope("dense1", reuse=tf.AUTO_REUSE):
            dense1 = self.dense(flattened, 64)
            relu5 = tf.nn.relu(dense1)
            drop1 = tf.nn.dropout(relu5, rate=0.2)
        with variable_scope("dense2", reuse=tf.AUTO_REUSE):
            y_pred = self.dense(drop1, self.action_space)

        return y_pred

    def save(self, model_path):
        """ 
        save is not implemented cause it's saving automatically during training
        """
        pass


class multihead(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "multihead"
        super(multihead, self).__init__(agent_class, self.model_type)

    @staticmethod
    def policy_head(inputs, action_space):
        # fully connected layer
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)

        # fully connected layer with softmax
        x = Dense(action_space,
                  activation='softmax',
                  name='policy_output')(x)
        return x

    @staticmethod
    def value_head(inputs):
        # fully connected layer
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)

        # fully connected layer with softmax
        x = Dense(1, name='value_output')(x)
        return x

    def create_model(self):
        """
        Function to create the model
        """
        inputs = Input(shape=(self.input_dim,1))
        x = conv_block(inputs, 16, 5, 2, 3, 2)
        x = conv_block(x, 32, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = conv_block(x, 64, 3, 2, 2, 2)
        x = Flatten()(x)

        policy_head = multihead.policy_head(x, self.action_space)
        value_head = multihead.value_head(x)

        self.model = Model(inputs = inputs,
                           outputs=[policy_head, value_head],
                           name="multihead")
        return self.model

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X

    def fit(self, epochs= 100, batch_size=64, learning_rate=0.01):
        """
        args:
            epochs (int): the number of epochs
            batch_size (int): the size of the batch in training and testing
            learning_rate (float): the relative size of the step taken in
                                   the steepest descent direction
        """
    
        # setup the training dataset
        dataset = self.train_dataset.repeat(epochs)
        dataset = dataset.batch(batch_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_example, next_label = iterator.get_next()

        # setup the test dataset
        test_dataset = self.test_dataset.repeat(epochs)
        test_dataset = test_dataset.batch(batch_size=batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        next_test_example, next_test_label = test_iterator.get_next()

        # create global step (required by MonitoredTrainingSession)
        global_step = tf.train.get_or_create_global_step()

        steps_per_epoch = int(math.ceil(self.X_train.shape[0]/batch_size))

        loss = self.loss(self.model(next_example),
                         next_label)  # define the loss
        test_loss = self.loss(self.model(next_test_example),
                next_test_label) # define test loss

        # # define accuracy
        # accuracy, _ = tf.metrics.accuracy(
        #     labels=tf.argmax(next_label, 1),
        #     predictions=tf.argmax(self.model(next_example), 1))
        # # define test accuracy
        # test_accuracy, _ = tf.metrics.accuracy(
        #     labels=tf.argmax(next_test_label, 1),
        #     predictions=tf.argmax(self.model(next_test_example), 1))
        accuracy = tf.reduce_sum(tf.cast(tf.equal(
            next_label, self.model(next_example)), tf.float32))
        test_accuracy = tf.reduce_sum(tf.cast(tf.equal(
            next_test_label, self.model(next_test_example)), tf.float32))

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

        # Create an optimizer with the desired parameters
        opt = AdamOptimizer(learning_rate=learning_rate)
        opt_op = opt.minimize(loss, global_step=global_step)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.path,
                summary_dir=self.path,
                save_checkpoint_steps=10 * steps_per_epoch,
                save_summaries_steps=steps_per_epoch) as sess:

            for e in range(epochs):
                print("Epoch %i/%i" % (e + 1, epochs))
                start_time = time.time()
                for i in range(steps_per_epoch):
                    # run one training step
                    sess.run(opt_op)

                # compute loss and accuracy for both training and test set
                loss_train, acc_train = sess.run([loss, accuracy])
                loss_test, acc_test = sess.run([loss, accuracy])

                elapsed_time = time.time() - start_time  # compute elapsed time
                print("steps: %i - %i s -loss: %f - accuracy: %f - val_loss: %f - val_accuracy: %f"
                      % (steps_per_epoch, elapsed_time,
                         loss_train, acc_train, loss_test, acc_test))

        adam = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)

        if self.model == None:
            self.create_model()

        losses={'policy_output': "categorical_crossentropy",
                'value_output': "mean_squared_error"}
        loss_weights={'policy_output': 1.0,
                      'value_output': 1.0}

        self.model.compile(loss=losses,
                           loss_weights=loss_weights,
                           optimizer=adam,
                           metrics=['accuracy'])

        self.model.fit(
            self.X_train,
            {'policy_output': self.moves_train,
            'value_output': self.rewards_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=[self.X_test,
            {'policy_output': self.moves_test,
            'value_output': self.rewards_test}],
            callbacks=[self.tensorboard],
            shuffle=True,
            verbose = 2
        )

    def extract_data(self, agent_class, val_split=0.3,
                     games=-1, balance=False):
        """
        args:
                agent_class (string)
                val_split (float): default 0.3
                games (int): default it loads all
                balance: this class does not support the balance function
        """
        print("Loading Data...", end='')
        replay = Experience(agent_class, load=True)
        moves, rewards, obs, eps = replay.load(games=games)

        # split dataset here
        data_list = [moves, rewards, obs]
        training_list, test_list = split_dataset(data_list, val_split)
        self.moves_train, self.moves_test = training_list[0], test_list[0]
        self.rewards_train, self.rewards_test = training_list[1], test_list[1]
        self.X_train, self.X_test = training_list[2], test_list[2]

        # convert to one-hot encoded tensor
        self.moves_train = one_hot(self.moves_train, replay.n_moves)
        self.moves_test = one_hot(self.moves_test, replay.n_moves)

        # reshape data for the network input layer
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        self.input_dim = self.X_train.shape[1]
        self.action_space = self.moves_train.shape[1]
        print("DONE")
