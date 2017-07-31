from base import NeuralModelBase
from .dataset import LPDataset


class LPRecogniser(NeuralModelBase):
    INPUT_SIZE = [-1, 256, 64, 3]
    dataset_class = LPDataset

    def process_result(self, result):
        return result[0][0] > result[0][1]
