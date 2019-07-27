import matplotlib.pyplot as plt
import math

def hist_classes(class_count):
    """
    Function to plot class distribution as histogram
    
    args:
        class_count: a dictionary that maps each class to its samples
    """
    plt.bar(list(class_count.keys()), class_count.values(), color='blue')
    plt.show()

def obtain_coords(class_count):
    """
    Function to obtain coordinates in two dimensions from a class distribution
    """

    # compute the total
    total = 0.0
    n_classes = 0
    for k in class_count.keys():
        total += class_count[k]
        n_classes += 1

    # initialize coordinates
    x = 0
    y = 0
    idx = 0.0 # index of class
    angle_fraction = (2.0 * math.pi / n_classes) # fraction of the angle
    for k in class_count.keys():
        # standardize the number of samples to have that each distribution has
        # sum 1 and each class is between 0 and 1.
        class_count[k] = float(class_count[k])/ total
        # edit coordinates
        x += math.cos(idx * angle_fraction) * class_count[k]
        y += math.sin(idx * angle_fraction) * class_count[k]
        
        idx += 1

    return x, y

def scatter_classes(class_count_list, labels_list = None):
    """
    Function to create a scatter plot of different class distributions in 2d

    args:
        class_count_list: a list of dictionaries, each representing a mapping
        of classes to number of samples in that class.
    """

    x_list = []
    y_list = []

    for class_count in class_count_list:
        x, y = obtain_coords(class_count)
        x_list.append(x)
        y_list.append(y)

    fig, ax = plt.subplots()
    ax.scatter(x_list, y_list)
    
    if labels_list is not None:
        for i, txt in enumerate(labels_list):
            ax.annotate(txt, (x_list[i], y_list[i]))
    
    plt.show()
