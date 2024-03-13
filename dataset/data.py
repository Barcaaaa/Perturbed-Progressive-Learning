import itertools
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """ This two stream batch sampler is used to read data from '_SSLDatasetWrapper'.

    It iterates two sets of indices simultaneously to read mini-batch for SSL.
    There are two sets of indices:
        labeled_idxs, unlabeled_idxs
    An 'epoch' is defined by going through the longer indices once.
    In each 'epoch', the shorter indices are iterated through as many times as needed.
    """

    def __init__(self, labeled_idxs, unlabeled_idxs, labeled_batch_size, unlabeled_batch_size):
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size

        assert len(self.labeled_idxs) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idxs) >= self.unlabeled_batch_size > 0

        self.unlabeled_batchs = len(self.unlabeled_idxs) // self.unlabeled_batch_size
        self.labeled_batchs = len(self.labeled_idxs) // self.labeled_batch_size

    def __iter__(self):
        if self.unlabeled_batchs >= self.labeled_batchs:
            unlabeled_iter = self.iterate_once(self.unlabeled_idxs)
            labeled_iter = self.iterate_eternally(self.labeled_idxs)
        else:
            unlabeled_iter = self.iterate_eternally(self.unlabeled_idxs)
            labeled_iter = self.iterate_once(self.labeled_idxs)

        return (labeled_batch + unlabeled_batch
                for (labeled_batch, unlabeled_batch) in zip(
                    self.grouper(labeled_iter, self.labeled_batch_size),
                    self.grouper(unlabeled_iter, self.unlabeled_batch_size)))

    def __len__(self):
        return max(self.unlabeled_batchs, self.labeled_batchs)

    def iterate_once(self, iterable):
        return np.random.permutation(iterable)

    def iterate_eternally(self, indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)

        return itertools.chain.from_iterable(infinite_shuffles())

    def grouper(self, iterable, n):
        # e.g., grouper('ABCDEFG', 3) --> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)
