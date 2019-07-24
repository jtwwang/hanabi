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

def load_predictor(model_class):
    """
    mediator function that allows to load only the required network
    to save memory and loading time
    """

    if model_class == 'conv':
        from conv_pred import conv_pred
        return conv_pred
    elif model_class == 'dense':
        from dense_pred import dense_pred
        return dense_pred
    elif model_class == 'lstm':
        from lstm_pred import lstm_pred
        return lstm_pred
    elif model_class == 'multihead':
        from multihead_pred import multihead_pred
        return multihead_pred
    else:
        raise ValueError("model class not recognized")
