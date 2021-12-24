from sklearn.model_selection import StratifiedShuffleSplit
from numpy.random import standard_normal
import numpy as np
from torch.utils.data import Dataset
import torch
import random


def sampling(X, Y, size, num_classes, args):
    """
    Function used to reorder data due to classes appear disordered.
    :param X: Samples
    :param Y: Labels
    :param size: minimum size of samples per class.
    :param num_classes:
    :return:
    """
    # reorder axes
    X = np.squeeze(X)

    if args.preprocess_format == 'cnn':
        new_x = np.zeros((num_classes, size, X.shape[1], X.shape[2], X.shape[3]))
    elif args.preprocess_format == 'rnn':
        new_x = np.zeros((num_classes, size, X.shape[1], X.shape[2]))
    else:
        print("File modules - ERROR [1]: option selected for args.preprocess_format")
        exit(-1)

    # labels
    new_y = np.zeros((num_classes, size))

    # iterate over number of classes
    for cls in range(num_classes):
        # obtain samples in class 'cls'
        indices = np.where(Y == cls)
        samples = X[indices]
        labels = Y[indices]

        # Resampling
        if samples.shape[0] >= size:
            # generate random indices
            rand_indices = random.sample(range(len(samples)), size)
            # select samples and labels using random indices
            new_x[cls, :, :] = samples[rand_indices]
            new_y[cls, :] = labels[rand_indices]

        else: # Oversampling
            # number of samples required
            num_samples_req = size - samples.shape[0]
            # generate random indices
            list_indices = np.array(range(0, samples.shape[0])).astype(int)
            rand_indices = np.random.choice(list_indices, size=num_samples_req, replace=True)

            # select data to add noise
            x = samples[rand_indices]
            y = labels[rand_indices]

            new_x[cls, :, :] = np.concatenate((samples, x), axis=0)
            new_y[cls, :] = np.concatenate((labels, y), axis=0)

    # copy to a final array
    Sx = None
    Sy = None
    # iterate over the number of classes
    for i in range(len(new_x)):
        if i == 0:
            Sx = new_x[i]
            Sy = new_y[i]
        else:
            Sx = np.concatenate((Sx, new_x[i]), axis=0)
            Sy = np.concatenate((Sy, new_y[i]), axis=0)

    if args.preprocess_format == 'cnn':
        Sx = Sx.reshape(-1, Sx.shape[1], Sx.shape[2], Sx.shape[3], 1).astype('float32')
    elif args.preprocess_format == 'rnn':
        Sx = Sx.reshape(-1, Sx.shape[1], Sx.shape[2], 1).astype('float32')
    else:
        print("File modules - ERROR [2]: option selected for args.preprocess_format")
        exit(-1)

    return Sx, Sy


# dataset definition
class Data(Dataset):
    # load the dataset
    def __init__(self, X, Y):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

