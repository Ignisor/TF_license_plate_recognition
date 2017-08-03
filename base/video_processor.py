from abc import ABCMeta
import time
import logging

from PIL import Image, ImageDraw
import numpy as np
import cv2

from ftplib import FTP


class VideoProcessor(metaclass=ABCMeta):
    def __init__(self, video_path, nn_class, part_ratio):
        self.nn = nn_class()
        self.part_ratio = part_ratio

    def process_frame(self, frame, part_ratio=None, step=0.25):
        """
        Split frame on many small images and send them to process_part
        :param PIL.Image.Image frame: image with frame to process
        :param part_ratio: ratio of the frame parts
        :param float step: how much pixels skip per iteration
        :return: data from process_part method
        """
        def mul_range(start, end, step):
            while start > end:
                yield start
                start = int(start * step)

        if not part_ratio:
            part_ratio = self.part_ratio

        for height in mul_range(frame.height, 64, 1 - step):
            width = int(height * part_ratio)

            t = time.time()
            parts_batch = []
            for x in range(0, frame.width, int(width * 0.33)):
                for y in range(0, frame.height, int(height * 0.33)):
                    part = frame.crop((x, y, x + width, y + height))
                    part = part.resize(self.nn.INPUT_SIZE[1:3])
                    parts_batch.append(np.array(part))

            logging.debug(f"parts batch got in: {time.time() - t:.4f}s")

            results = self.nn.run_batch(parts_batch)

            all_imags = np.array([])

            for part, result in zip(parts_batch, results):
                if result:
                    border = cv2.copyMakeBorder(part, top=1, bottom=1, left=1, right=1,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=[255, 0, 0])

                    all_imags = np.append(all_imags, border, axis=0) if all_imags.size > 0 else border

            if all_imags.size > 0:
                cv2.imshow('plate', cv2.cvtColor(all_imags, cv2.COLOR_BGR2RGB))

                while True:
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break