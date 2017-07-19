from base import NeuralModelBase
from .dataset import LPDataset

class LPRecogniser(NeuralModelBase):
    INPUT_SIZE = [-1, 256, 64, 3]
    dataset_class = LPDataset
