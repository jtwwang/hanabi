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
from .treenet import treenet
from blocks import conv_block

from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense, Flatten, Input
from tensorflow.keras import regularizers

from data_pipeline.util import accuracy

import numpy as np
import pandas as pd
import tensorflow as tf
import time

tf.disable_eager_execution()


class CMCL(treenet):
    def __init__(self, agent_class, predictor_name):
        super(CMCL, self).__init__(agent_class, predictor_name, "CMCL")
        self.beta = 0.5  # penalty parameter
        self.n_heads = 4 # number of heads
        self.k = 2  # overlap parameter - must be less than n_heads
        self.WD_FACTOR = 10e-4 # weight decay

    def policy_head(self, inputs, output_dim, name_head):
        """
        Architecture for each of the heads

        args:
            inputs: a tensor
            output_dim: the number of classes we are predicting
            name_head: the name to give to the output layer
        """

        x = conv_block(inputs, 50, 3, 2, 2, 2)
        x = conv_block(x, 50, 3, 2, 2, 2)
        x = Dropout(0.1)(x)
        x = conv_block(x, 50, 3, 2, 2, 2)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(output_dim,
                  activation='softmax',
                  name=name_head)(x)
        return x

    def create_model(self):
        """
        Function to create the model
        """
        inputs = Input(shape=(self.input_dim, 1))
        x = conv_block(inputs, 50, 5, 2, 3, 2)

        heads = []
        for i in range(self.n_heads):
            name_head = 'ph' + str(i)
            heads.append(self.policy_head(x, self.action_space, name_head))

        self.model = Model(inputs=inputs,
                           outputs=heads,
                           name="treenet")
        return self.model

    def loss(self, logits_list, labels):
        """
        CMCL version 0: confident oracle loss with exact gradient

        args:
            logits_list: List of logits calculated from models to ensemble.
            labels: Label input corresponding to the calculated batch.
        """
        # regularization
        weights = self.model.get_weights()
        rloss = self.WD_FACTOR * tf.add_n([tf.nn.l2_loss(w) for w in weights])
        total_loss = rloss

        # classification loss
        closs_list = [K.categorical_crossentropy(labels, logits)
                      for logits in logits_list]

        softmax_list = [K.clip(logits, 1e-10, 1.0) for logits in logits_list]
        entropy_list = [K.log(
            self.action_space+0.)-K.mean(K.log(softmax), 1) for softmax in softmax_list]
        loss_list = []
        for m in range(self.n_heads):
            loss_list.append(
                closs_list[m] + self.beta*tf.add_n(entropy_list[:m]+entropy_list[m+1:]))
        # top k lowest losses
        temp, min_index = tf.nn.top_k(-K.transpose(loss_list), self.k)
        min_index = K.transpose(min_index)

        for m in range(self.n_heads):
            for topk in range(self.k):
                condition = K.equal(min_index[topk], m)
                total_loss += K.sum(tf.where(condition,
                                             closs_list[m] -
                                             self.beta*entropy_list[m],
                                             tf.zeros_like(closs_list[0])))
        total_loss += tf.reduce_sum(self.beta*tf.add_n(entropy_list))
        return total_loss

    def custom_optimizer(self):
        labels = K.placeholder(shape=(None, self.action_space))
        logits_list = self.model.outputs

        # compute loss
        loss = self.loss(logits_list, labels)
        adam = optimizers.Adam(lr=self.lr)
        updates = adam.get_updates(loss, self.model.trainable_weights)
        train = K.function([self.model.inputs, labels], [
                           self.model.outputs], updates=updates)
        return train

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):
        """
        args:
            epochs (int): the number of epochs
            batch_size (int): the size of the batch in training and testing
            learning_rate (float): the relative size of the step taken in
                                   the steepest descent direction
        """

        # reshape the data
        self.X_train = self.reshape_data(self.X_train)
        self.X_test = self.reshape_data(self.X_test)
        train_entries = self.X_train.shape[0]
        test_entries = self.X_test.shape[0]
        train_steps = int(train_entries / batch_size)

        print("Training on %i samples, validating on %i." %
              (train_entries, test_entries))

        self.lr = learning_rate

        if self.model is None:
            self.create_model()

        # we must not compile, just create the predict function
        self.model._make_predict_function()

        opt = self.custom_optimizer()

        # initialize log
        log = {'acc':[], 'val_acc':[]}

        for e in range(epochs):
            print("Epoch %i/%i" % (e+1, epochs))
            start = time.time()
            # shuffle the data
            p = np.random.permutation(train_entries)
            self.X_train = self.X_train[p]
            self.y_train = self.y_train[p]

            p = np.random.permutation(test_entries)
            self.X_test = self.X_test[p]
            self.y_test = self.y_test[p]

            for s in range(train_steps):
                batch_x = self.X_train[s*batch_size: (s+1)*batch_size]
                batch_y = self.y_train[s*batch_size: (s+1)*batch_size]
                opt([batch_x, batch_y])

            acc = 0
            test_pred = self.predict(self.X_test, batch_size)
            train_pred = self.predict(self.X_train, batch_size)
            acc = accuracy(self.y_train, train_pred)
            val_acc = accuracy(self.y_test, test_pred)

            log['acc'].append(acc)
            log['val_acc'].append(val_acc)

            end = time.time()
            print("%i/%i - %is - acc: %f - val_acc: %f"
                  % (train_entries, train_entries, (end-start), acc, val_acc))

        # create log
        log['pred_name'] = self.predictor_name
        log['epochs'] = epochs
        log['batch_size'] = batch_size
        log['episodes'] = 'TODO'
        self.log = pd.DataFrame(log)
