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


def one_hot_list(data_list, classes):

    new_list = []
    for item in data_list:
        new_list.append(one_hot(item, classes))

    return np.asarray(new_list)


def split_dataset(data_list, val_split):
    """
    Args:
        examples (np.ndarray)
        labels (np.ndarray)
    """
    size_test = int(val_split * data_list[0].shape[0])

    # initialize empty list for output
    training_list = []
    test_list = []

    for set in data_list:
        set_test, set_training = set[:size_test], set[size_test:]
        training_list.append(set_training)
        test_list.append(set_test)

    return training_list, test_list
