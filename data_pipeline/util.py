import numpy as np


def one_hot(data, classes):
    """
    Args:
        data (np.ndarray)
        classes (int)
    """
    entries = data.shape[0]
    one_hot_data = np.zeros((entries, classes))
    one_hot_data[np.arange(entries), data[:entries]] = 1
    return one_hot_data

def split_dataset(examples, labels, val_split):
    """
    Args:
        examples (np.ndarray)
        labels (np.ndarray)
    """
    size_test = int(val_split * examples.shape[0])
    X_test, X_train = examples[:size_test], examples[size_test:]
    y_test, y_train = labels[:size_test], labels[size_test:]

    return X_train, y_train, X_test, y_test

