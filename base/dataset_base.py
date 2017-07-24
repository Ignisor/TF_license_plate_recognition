import random
from abc import ABCMeta, abstractmethod

import numpy


class DataSetBase(metaclass=ABCMeta):
    @abstractmethod
    def _get_set(self, amount=None, test=False):
        pass

    def get_batch(self, amount=None, test=False):
        """Returns 'amount' random images as vectors with answers"""
        set = self._get_set(amount, test)

        return self._process_set(set)

    def _process_set(self, set):
        return set
