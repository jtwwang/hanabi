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

from .conv_pred import conv_pred
from os import listdir, path
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Softmax
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

class ensemble(conv_pred):

    def __init__(self, agent_class, predictor_name):

        # initialize
        model_dir = path.join("model", agent_class)
        model_dir = path.join(model_dir, "conv")

        # open the path
        files = listdir(model_dir)
        print("Building an ensemble of %i networks" %(len(files)))

        self.models = [] # initialize empty list of models
        self.files = []
        for file in files:
            pred_dir = path.join(model_dir, file)
            pred_dir = path.join(pred_dir, "predictor.h5")
            self.files.append(file)
            self.models.append(load_model(pred_dir))

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):
        print("The ensemble model does not support fit().\n"
              "Train the networks separately before using this class.")
        print("Evaluating the ensemble")

        self.X_test = self.reshape_data(self.X_test)

        steps = int(self.X_test.shape[0]/batch_size)

        acc = CategoricalAccuracy()
        single_acc = CategoricalAccuracy()
        avg_pred = np.zeros(self.y_test.shape)
        for i in range(len(self.files)):
            #model = load_model(self.files[i])
            pred = self.models[i].predict(
                x=self.X_test,
                batch_size=batch_size,
                verbose=0)
            avg_pred += pred
            single_acc.update_state(self.y_test, pred)
            print(" * "+self.files[i]+" : "+str(single_acc.result().numpy()))
            single_acc.reset_states()

        # compute accuracy
        avg_pred /= len(self.files)
        acc.update_state(self.y_test, avg_pred)
        print('Final result: %f' %acc.result().numpy())

    def save(self):
        print("Model is not saved because it is an ensemble.")

    def load(self):
        print("Ensemble loads all models by default")
