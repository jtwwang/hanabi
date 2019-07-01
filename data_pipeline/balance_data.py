from experience import Experience
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

global accepted_types
accepted_types = [int,
                  np.uint8]


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


def balance_data(examples, labels, randomize=True):
    """
    Args:
        examples: 2d ndarray
        labels: 1d ndarray
        randomize: boolean, default is True
    """

    # create dictionaries
    class_count = {}
    class_data = {}

    # quantify class imbalance
    for i in range(labels.shape[0]):
        l = labels[i]
        if l not in class_count.keys():
            class_count[l] = 1
            class_data[l] = []
        else:
            class_count[l] += 1
            class_data[l].append(examples[i])

    mean = compute_mean(class_count)

    # convert into numpy arrays
    for k in class_data.keys():
        class_data[k] = np.asarray(class_data[k])

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
    exp = Experience('SimpleAgent',
                     load=True)
    labels, _, examples, _ = exp.load()
    new_examples, new_labels = balance_data(examples, labels)


if __name__ == "__main__":
    main()
