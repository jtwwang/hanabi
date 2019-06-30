import numpy as np

def one_hot(data, classes):

    entries = data.shape[0]
    one_hot_data = np.zeros((entries, classes))
    one_hot_data[np.arange(entries), data[:entries]] = 1
    return one_hot_data
