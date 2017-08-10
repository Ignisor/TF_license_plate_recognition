from abc import ABCMeta
import time
import logging

from PIL import Image, ImageDraw
import numpy as np
import cv2


class VideoProcessor(metaclass=ABCMeta):
    SEARCH_RECTANGLE = ((400, 500), (1000, 700))

    def __init__(self, video_path, nn_class, part_ratio):
        self.nn = nn_class()
        self.part_ratio = part_ratio

    def process_video(self, video_file, frameskip=5):
        """
        Split video in to frames
        :param str video_file: path to video 
        :param int frameskip: how many frames to skip
        """
        video = cv2.VideoCapture(video_file)
        c = 0
        while video.isOpened():
            ret, frame = video.read()

            d_frame = np.empty_like(frame)
            d_frame[:] = frame
            cv2.rectangle(d_frame, self.SEARCH_RECTANGLE[0], self.SEARCH_RECTANGLE[1], color=(0, 0, 255))

            c += 1
            if c >= frameskip:
                plates = self.process_frame(frame)
                c = 0

                for plate in plates:
                    cv2.imshow('plate', plate)
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('n'):
                            break

                cv2.imshow('frame', d_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    def process_frame(self, frame, part_ratio=None, step=0.25):
        """
        Use provided NN for many parts in SEARCH_RECTANGLE
        :param np.array frame: image with frame to process
        :param part_ratio: ratio of the frame parts
        :param float step: how much pixels skip per iteration
        :return: positions of found LPs
        """
        def mul_range(start, end, step):
            while start < end:
                yield start
                start = int(start * step)

        if not part_ratio:
            part_ratio = self.part_ratio

        # crop image
        s_rect = self.SEARCH_RECTANGLE
        img = frame[s_rect[0][1]:s_rect[1][1], s_rect[0][0]:s_rect[1][0]]

        img_height, img_width = img.shape[:2]
        img_center = (img_width // 2, img_height // 2)

        parts_batch = []
        t = time.time()
        for height in mul_range(32, img_height, 1 + step):
            width = int(height * part_ratio)

            part_rect = (
                (img_center[0] - width, img_center[1] - height),
                (img_center[0] + width, img_center[1] + height),
            )

            part = img[part_rect[0][1]:part_rect[1][1], part_rect[0][0]:part_rect[1][0]]
            part = cv2.resize(part, tuple(self.nn.INPUT_SIZE[1:3]))

            parts_batch.append(part)

        logging.debug(f"parts batch got in: {time.time() - t:.4f}s")

        results = self.nn.run_batch(parts_batch)

        for result, rect in zip(results, parts_batch):
            if result:
                yield rect
