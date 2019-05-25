import getopt
import sys
import numpy as np

from experience import Experience

from predictors.conv_pred import conv_pred
from predictors.dense_pred import dense_pred
from predictors.lstm_pred import lstm_pred

model_dict = {
    "lstm": lstm_pred,
    "dense": dense_pred,
    "conv": conv_pred
}

if __name__ == "__main__":
    flags = {'model_class': "dense",
             'epochs': 40,
             'batch_size': 16,
             'lr': 0.001,
             'agent_class': 'SimpleAgent',
             'val_split': 0.3,
             'cv': -1,
             'load': False}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['model_class=',
                                        'epochs=',
                                        'batch_size=',
                                        'lr=',
                                        'agent_class=',
                                        'val_split=',
                                        'cv=',
                                        'load='])
    if arguments:
        sys.exit()
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)

    agent_class = flags['agent_class']
    model_class = model_dict[flags['model_class']]

    pp = model_class(agent_class)
    pp.extract_data(agent_class)
    pp.create_model()  # Add Model_name here to create different models

    # ip.embed()

    if flags['load']:
        pp.load()

    pp.fit(epochs=flags['epochs'],
           batch_size=flags['batch_size'],
           learning_rate=flags['lr'],
           val_split=flags['val_split'])
    pp.save()
