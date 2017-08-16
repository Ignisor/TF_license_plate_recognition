from base import NeuralModelBase
from .dataset import LPDataset, MongoLPDataset


class LPRecogniser(NeuralModelBase):
    INPUT_SIZE = [-1, 256, 64, 3]
    dataset_class = MongoLPDataset

    def process_results(self, results):
        for res in results:
            yield res[0] > res[1]
