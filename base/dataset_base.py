import random
from abc import ABCMeta, abstractmethod

import numpy


class DataSetBase(metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def _get_set(amount=None, test=False):
        pass

    @abstractmethod
    @staticmethod
    def get_batch(amount=None, test=False):
        """Returns 'amount' random images as vectors with answers"""
        set = ImageSet._get_set(amount, test)

        return ImageSet.process_set(set)

    @abstractmethod
    @staticmethod
    def process_set(set):
        return set
