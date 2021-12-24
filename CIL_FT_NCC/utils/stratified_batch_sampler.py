import torch
import torch as th
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, num_classes, shuffle=False):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.num_classes = num_classes
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        # variable temporal
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        #print(self.y)
        #input("so")

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):

            # OBTAIN num samples for each class
            samples_per_class = []
            lbls = self.y[test_idx]
            for c in range(self.num_classes):
                t = np.where(lbls == c)
                # agregamos al menos un elemento de clase
                if t[0].shape[0] < 1:
                    # buscamos dentro del conjunto de datos original
                    ind = np.where(self.y == c)
                    # selecionamos aleatoriamente alguno
                    a = random.randint(0, len(ind[0])-1)
                    # agregamos el indice
                    test_idx = np.concatenate((test_idx, np.array([ind[0][a]])), axis=0)

            yield test_idx

    def __len__(self):
        return len(self.y)