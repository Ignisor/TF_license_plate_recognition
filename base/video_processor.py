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

        self.find_lp(frame)

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

    def find_lp(self, frame):
        frame = np.array(frame)
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 9), np.uint8)

        img = grey_frame
        # img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        img = cv2.dilate(img, kernel, iterations=3)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.subtract(img, grey_frame)

        for _ in range(3):
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=2)

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rects = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            area = abs(cv2.contourArea(contour))
            bb_area = rect[1][0] * rect[1][1]
            ratio = area/bb_area if area and bb_area else 0.0

            if ratio >= 0.45 and bb_area >= 400:
                rects.append(rect)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for rect in rects:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img = cv2.drawContours(img, [box], 0, (0, 0, 255), 5)

        cv2.imshow('img', cv2.resize(img, tuple(sh//4 for sh in img.shape[:2])))

        while True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
