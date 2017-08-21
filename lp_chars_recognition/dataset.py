import random
import logging
from io import BytesIO
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from base import DataSetBase


class CharsDataset(DataSetBase):
    CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

    def _get_set(self, amount=None, test=False):
        imgs = []
        labels = []
        amount = amount or 100

        for i in range(amount):
            img, char = self.generate_char_img()

            img = np.reshape(img, (24 * 32, 1))

            imgs.append(img)
            labels.append([int(self.CHARS.index(char) == i) for i in range(36)])

        return imgs, labels

    def generate_char_img(self):
        """generate random char with noise"""
        char = random.choice(self.CHARS)

        img = Image.new('L', (24, 32))

        draw = ImageDraw.Draw(img)

        # draw text
        text_font = ImageFont.truetype("lp_finder/data/din1451alt.ttf", 30)
        w, h = draw.textsize(char, font=text_font)
        text_xy = ((img.width - w) / 2, (img.height - h - 5) / 2)
        draw.text(text_xy, char, fill=255, font=text_font)

        img = np.array(img)

        # Add noise
        for i in range(random.randint(1, 4)):
            noise = np.zeros(img.shape, np.uint8)
            cv2.randn(noise, -255, 255)

            img = np.clip(img + noise, 0, 255)

        return img, char
