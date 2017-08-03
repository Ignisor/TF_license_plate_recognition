from urllib.request import urlopen
from io import BytesIO
import os

from numpy import array
from PIL import Image

from base import NeuralModelBase


class CarRecogniser(NeuralModelBase):
    """Neuro model used to recognise is it car on image. It works with 64x64 RGB images."""

    def process_data(self, data):
        """
        prepares data for neural network
        :param data: openable by PIL.Image (for ex. path to file)
        :return: image as numpy array 
        """
        image = Image.open(data)
        image = image.resize((64, 64))
        image = image.convert('RGB')

        vector = []
        for pixel in image.getdata():
            vector.append(tuple((color / 255 for color in pixel)))

        vector = array(vector).reshape(1, 64 * 64, 3)

        return vector

    def process_result(self, result):
        result = result[0]
        return result[0] > result[1]

    def is_car_from_url(self, image_url):
        """Check is car on image loaded from url. Used for testing purposes."""
        img_file = BytesIO(urlopen(image_url).read())

        return self.run_single(img_file)
