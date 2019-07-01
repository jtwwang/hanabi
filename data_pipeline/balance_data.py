# Developed by Lorenzo Mambretti
# June 2019
# lorenz.m97@gmail.com

from experience import Experience
import numpy as np
import matplotlib.pyplot as plt


def remove_duplicates(class_data):
    for k in class_data.keys():
        class_data[k] = np.unique(class_data[k], axis=0)
    return class_data


def plot_classes(class_count):
    plt.bar(list(class_count.keys()), class_count.values(), color='blue')
    plt.show()


def undersample(data, target):
    sample_index = np.random.choice(data.shape[0], size=target)
    return data[sample_index]


def oversample(data, target):
    size_to_add = target - data.shape[0]
    sample_index = np.random.choice(data.shape[0], size=size_to_add)
    sample = data[sample_index]
    return np.append(data, sample, axis=0)


def compute_mean(class_count):
    count = 0
    mean = 0
    for k in class_count.keys():
        mean += class_count[k]
        count += 1
    mean /= count
    return int(mean)


def divide(examples, labels):
    """
    Function to divide the examples in classes based on the labels
    """
    class_data = {}
    # quantify class imbalance
    for i in range(labels.shape[0]):
        k = labels[i]
        if k not in class_data.keys():
            class_data[k] = []
        else:
            class_data[k].append(examples[i])

    # convert into numpy arrays
    for k in class_data.keys():
        class_data[k] = np.asarray(class_data[k])

    return class_data


def count(class_data):
    """
    Function to count each class, given than the classes are divided
    """
    class_count = {}
    for k in class_data.keys():
        class_count[k] = class_data[k].shape[0]
    return class_count


def divide_and_count(examples, labels):
    """
    Function that counts and divide classes based on labels
    """
    class_data = divide(examples, labels)
    class_count = count(class_data)
    return class_count, class_data


def balance_data(examples, labels, randomize=True):
    """
    Args:
        examples: 2d ndarray
        labels: 1d ndarray
        randomize: boolean, default is True
    """

    class_count, class_data = divide_and_count(examples, labels)
    mean = compute_mean(class_count)

    # make all class having the same amount of data
    for label in class_count.keys():
        if class_count[label] < mean:
            class_data[label] = oversample(class_data[label], mean)
        elif class_count[label] > mean:
            class_data[label] = undersample(class_data[label], mean)
        else:
            pass

    # recompose the data
    new_examples = []
    new_labels = []
    for k in class_data.keys():
        new_examples.append(class_data[k])
        new_labels.append(np.repeat(k, mean))

    # convert into numpy arrays
    new_examples = np.asarray(new_examples).reshape((-1, examples.shape[1]))
    new_labels = np.asarray(new_labels).flatten()

    # randomize
    if randomize:
        p = np.random.permutation(new_labels.shape[0])
        new_examples, new_labels = new_examples[p], new_labels[p]
    return new_examples, new_labels


def main():
    exp = Experience('RainbowAgent',
                     load=True)
    labels, _, examples, _ = exp.load()
    new_examples, new_labels = balance_data(examples, labels)
    class_count, class_data = divide_and_count(examples, labels)
    plot_classes(class_count)


if __name__ == "__main__":
    main()
