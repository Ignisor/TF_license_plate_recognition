import random
from abc import ABCMeta, abstractmethod

import numpy


class DataSetBase(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _get_set(amount=None, test=False):
        pass

    @staticmethod
    @abstractmethod
    def get_batch(amount=None, test=False):
        """Returns 'amount' random images as vectors with answers"""
        set = DataSetBase._get_set(amount, test)

        return DataSetBase.process_set(set)

    @staticmethod
    @abstractmethod
    def process_set(set):
        return set
