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

import getopt
import sys

from predictors.load_predictors import load_predictor

if __name__ == "__main__":
    flags = {'model_class': "conv",
             'epochs': 40,
             'batch_size': 16,
             'lr': 0.001,
             'agent_class': 'RainbowAgent',
             'val_split': 0.3,
             'cv': -1,
             'load': False,
             'summary': False,
             'games': -1,
             'balance': False,
             'model_name': "predictor.h5"}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['model_class=',
                                        'epochs=',
                                        'batch_size=',
                                        'lr=',
                                        'agent_class=',
                                        'val_split=',
                                        'cv=',
                                        'load=',
                                        'summary=',
                                        'games=',
                                        'balance=',
                                        'model_name='])
    if arguments:
        sys.exit()
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    agent_class = flags['agent_class']

    pp = load_predictor(flags['model_class'])(agent_class)
    pp.extract_data(agent_class,
                    val_split=flags['val_split'],
                    games=flags['games'],
                    balance=flags['balance'])

    if flags['load']:
        pp.load(flags['model_name'])

    if flags['summary']:
        print(pp.model.summary())

    pp.fit(epochs=flags['epochs'],
           batch_size=flags['batch_size'],
           learning_rate=flags['lr'])
    pp.save(flags['model_name'])
