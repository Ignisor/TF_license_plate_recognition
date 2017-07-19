import random
import time

from PIL import Image, ImageDraw, ImageFont

from base import DataSetBase


class LPDataset(DataSetBase):
    def _get_set(self, amount=None, test=False):
        imgs = []
        labels = []
        amount = amount or 1000

        for i in range(amount):
            img = LPDataset.generate_lp()
            img = LPDataset._process_image(img)
            imgs.append(img)
            labels.append([1, 0])

        return imgs, labels

    @staticmethod
    def generate_lp():
        """generate random license plate"""
        rand_letter = lambda count: ''.join(random.choice('QWERTYUIOPASDFGHJKLZXCVBNM') for _ in range(count))
        rand_num = lambda count: ''.join(str(random.randint(0, 9)) for _ in range(count))
        text = f'{rand_letter(2)} {rand_num(4)} {rand_letter(2)}'

        img = Image.new('RGBA', (256, 64))

        # generate background
        colors_amount = 20
        back_colors = [LPDataset.get_random_color() for _ in range(colors_amount)]
        n = 0
        pixels = img.load()
        for i in range(img.size[0]):
            step = random.randint(1, 7)
            for j in range(img.size[1]):
                pixels[i, j] = back_colors[n]
                n += step
                if n > colors_amount - 1:
                    n = 0

        # generate plate
        plate = Image.new('RGB', (140, 30), LPDataset.get_random_color(235, 255))
        draw = ImageDraw.Draw(plate)

        # draw random flag
        draw.rectangle([(0, 0), (plate.width * 0.1, plate.height / 2)], fill=LPDataset.get_random_color())
        draw.rectangle([(0, plate.height / 2), (plate.width * 0.1, plate.height)], fill=LPDataset.get_random_color())

        # draw border
        for i in range(2):
            i = i + 1
            xy = [(-1 + i, -1 + i), (plate.size[0] - i, plate.size[1] - i)]
            draw.rectangle(xy, outline=(0, 0, 0))

        # draw text
        text_xy = (plate.width * 0.15, plate.height * 0.15)
        text_font = ImageFont.truetype("arial.ttf", 20)
        draw.text(text_xy, text, fill=LPDataset.get_random_color(0, 25), font=text_font)

        # rotate and stretch plate
        n_plate = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
        size = (int(plate.width * random.uniform(0.9, 1)), int(plate.height * random.uniform(0.9, 1)))
        plate = plate.resize(size, Image.LANCZOS)
        n_plate.paste(plate, ((n_plate.width - plate.width) // 2, (n_plate.height - plate.height) // 2))
        n_plate = n_plate.rotate(random.randint(-7, 7), resample=Image.BILINEAR)

        img = Image.alpha_composite(img, n_plate)

        return img

    @staticmethod
    def _process_image(img):
        # convert to vector
        vector = []
        for pixel in img.getdata():
            vector.append((pixel[0] / 255, pixel[1] / 255, pixel[2] / 255))

        return vector

    @staticmethod
    def get_random_color(min=0, max=255, alpha=1):
        return tuple(random.randint(min, max) for _ in range(3)) + (alpha, )