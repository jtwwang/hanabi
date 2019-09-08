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

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal


class transfer_tn(treenet):
    def __init__(self, agent_class, predictor_name, model_type=None):
        """
        the network is defined in a identical manner to the the treenet
        """
        if model_type is None:
            model_type = "transfer_tn"
        self.model_type = model_type
        super(transfer_tn, self).__init__(agent_class=agent_class,
                                          model_type=self.model_type,
                                          predictor_name=predictor_name)

        self.n_heads = 4  # standard size for ensemble
        self.freeze_layers = True  # whether we are freezing the layer or not
        self.reset = False  # whether the unfrozen layers should be initialized
        self.n_frozen = 2

    def fit(self, epochs=100, batch_size=64, learning_rate=0.01):

        # if the model was not loaded, create model
        if self.model is None:
            self.create_model()

        # freeze layers if we require so
        if self.freeze_layers:
            for layer in self.model.layers[:self.n_frozen]:
                layer.trainable = False

        # reset all the not frozen layers
        if self.reset:
            session = K.get_session()
            for layer in self.model.layers[self.n_frozen:]:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel_intializer = he_normal()
                    layer.kernel.initializer.run(session=session)

        # fit the model
        super(transfer_tn, self).fit(epochs, batch_size, learning_rate)
