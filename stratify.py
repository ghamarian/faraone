from sklearn.model_selection._split import BaseShuffleSplit, _approximate_mode
from sklearn.utils import check_random_state, check_array
import numpy as np
from sklearn.utils.validation import _num_samples, indexable


class StratifiedShuffle(BaseShuffleSplit):

    def __init__(self, X, y, batch_size):
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        self.n_classes = self.classes.shape[0]
        self.class_counts = np.bincount(self.y_indices)
        self.random_state = None
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.n_splits = _num_samples(self.X) // self.batch_size

    def _iter_indices(self, groups=None):
        n_samples = self.batch_size
        y = check_array(self.y, ensure_2d=False, dtype=None)
        # n_train = _validate_shuffle_split(n_samples, 0, self.train_size)
        n_train = n_samples

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        if np.min(self.class_counts) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of groups for any class cannot"
                             " be less than 2.")

        if n_train < self.n_classes:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, self.n_classes))

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(self.y_indices, kind='mergesort'),
                                 np.cumsum(self.class_counts)[:-1])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(self.class_counts, n_train, rng)

            train = []

            for i in range(self.n_classes):
                permutation = rng.permutation(self.class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])

            train = rng.permutation(train)

            yield train

    def split(self, groups=None):
        X, y, groups = indexable(self.X, self.y, groups)
        for train in self._iter_indices(groups):
            yield train


if __name__ == '__main__':

    y = np.array([1] * 2 + [2] * 25 + [3] * 35 + [4] * 38)
    X = np.arange(100)

    s = StratifiedShuffle(X, y, 4)
    for x in s.split():
        print(X[x], y[x])
