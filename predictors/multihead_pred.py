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
from .policy_pred import policy_pred

import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer, Saver
from tensorflow.compat.v1 import variable_scope, get_variable, placeholder, reset_default_graph
from tensorflow.compat.v1 import global_variables_initializer, variables_initializer
from tensorflow.compat.v1.losses import softmax_cross_entropy
from tensorflow.compat.v1.layers import Flatten
from tensorflow.compat.v1.summary import FileWriter
import math
import time

class MultiHeadModel(object):
    def __init__(self, action_space):
        self.var_list = []
        self.action_space = action_space

        self.x = placeholder(tf.float32, shape=(None, 658, 1))
        self.y = placeholder(tf.float32, shape=(None, 20))

        with variable_scope("conv1"):
            relu1 = self.conv_relu(self.x, [5,1,16], stride = 2)
        with variable_scope("conv2"):
            relu2 = self.conv_relu(relu1, [3,16,32], stride = 2)
        with variable_scope("conv3"):
            relu3 = self.conv_relu(relu2, [3,32,64], stride = 2)
        with variable_scope("conv4"):
            relu4 = self.conv_relu(relu3, [3,64,64], stride = 2)
        with variable_scope("flatten"):
            flattened = Flatten()(relu4)
        with variable_scope("dense"):
            self.y_pred = self.dense(flattened, self.action_space)

    def conv_relu(self, inputs, kernel_shape, stride):
        # Create variable named "weights"
        weights = get_variable("weights", kernel_shape,
                initializer = tf.random_normal_initializer())
        # Create variable named "biases"
        bias_shape = kernel_shape[-1]
        biases = get_variable("biases", [bias_shape],
                initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv1d(inputs, weights,
                stride=stride, padding='SAME')
        return tf.nn.relu(conv + biases)

    def dense(self, inputs, units):
        # Create vriable named "weights"
        dense_shape = [inputs.shape[-1], units]
        weights = get_variable("weights", dense_shape,
                initializer = tf.random_normal_initializer())
        # Create variable named "baises"
        biases = get_variable("biases", [units],
                initializer=tf.constant_initializer(0.0))

        self.var_list.append(weights)
        self.var_list.append(biases)

        y = tf.linalg.matmul(inputs,weights) + biases
        return y

    def __call__(self, x):
        
        return self.y_pred

    def save(self, model_path):
        """ 
        save is not implemented cause it's saving automatically during training
        """
        pass


class multihead_pred(policy_pred):
    def __init__(self, agent_class):
        self.model_type = "multihead"
        super(multihead_pred, self).__init__(agent_class, self.model_type)

        reset_default_graph()

    def create_model(self):
        """ Create the model of the network in separate class """
        self.model = MultiHeadModel(self.action_space)

        # Add ops to save and restore all the variables
        self.saver = Saver(self.model.var_list)

        return self.model

    def load(self, model_path):
        self.load = True

    def loss(self, predicted_y, desired_y):
        """ Define the loss of the model """
        CE = softmax_cross_entropy(desired_y, predicted_y)
        return CE

    def reshape_data(self, X_raw):
        if X_raw.shape == (self.action_space,):  # If only one sample is put in
            X_raw = np.reshape(X_raw, (1, X_raw.shape[0]))

        # Add an additional dimension for filters
        X = np.reshape(X_raw, (X_raw.shape[0], X_raw.shape[1], 1))
        return X
    
    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):
        """
        Train the model using the data that has been loaded

        Args:
            epochs: (int) number of iterations through the entire data set
            batch_size: (int) size of the mini-batch that we're using
            learning rate: (float) the relative size of the training step
        """
        
        steps_per_epoch = int(math.ceil(self.X_train.shape[0]/batch_size))

        loss = self.loss(self.model.y_pred, self.model.y) # define the loss
        
        # define accuracy
        accuracy, _ = tf.metrics.accuracy(self.model.y, self.model.y_pred)

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        
        # Create an optimizer with the desired parameters
        variables = self.model.var_list
        opt = AdamOptimizer(learning_rate = learning_rate)
        opt_op = opt.minimize(loss)

        # Create hooks for the session
        saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir = self.path,
                save_steps = 10 * steps_per_epoch)
        summary_hook = tf.train.SummarySaverHook(
                save_steps = steps_per_epoch,
                output_dir = self.path,
                summary_op = merged)
        hooks = [saver_hook, summary_hook]

        tf.train.create_global_step()

        with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
            for e in range(epochs):
                print("Epoch %i/%i" %(e, epochs))
                start_time = time.time()
                for i in range(steps_per_epoch):
                    # divide the data in batches
                    start = i * batch_size
                    end = min(self.X_train.shape[0], (i + 1) * batch_size)
                    batch_x = self.X_train[start:end]
                    batch_y = self.y_train[start:end]

                    # run one training step
                    sess.run(opt_op, feed_dict={
                        self.model.x: batch_x,
                        self.model.y: batch_y})

                # compute loss and accuracy for both training and test set
                loss_train, acc_train = sess.run([loss, accuracy], feed_dict={
                    self.model.x: self.X_train,
                    self.model.y: self.y_train})
                loss_test, acc_test = sess.run([loss, accuracy], feed_dict={
                    self.model.x: self.X_test,
                    self.model.y: self.y_test})

                elapsed_time = time.time() - start_time # compute elapsed time
                print("steps: %i - %i s -loss: %f - accuracy: %f - val_loss: %f - val_accuracy: %f" 
                        %(steps_per_epoch, elapsed_time,
                            loss_train, acc_train, loss_test, acc_test))

    def extract_data(self, agent_class, val_split=0.3,
                     games=-1, balance=False):
        """
        args:
                agent_class (string)
                val_split (float)
                games (int)
        """
        obs, actions, _ = super(multihead_pred, self).extract_data(agent_class,
                                                              val_split, games=games, balance=balance)

        # change the type to float32
        self.X_train = self.X_train.astype('float32')
        self.y_train = self.y_train.astype('float32')

        # reshape the data
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)

        self.input_dim = self.X_train.shape[1]
        self.action_space = self.y_train.shape[1]

        self.dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))

        # Create the model
        self.model = self.create_model()

