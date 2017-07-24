import time
import random

import cv2
from PIL import Image

from neuro_model import CarRecogniser


cr = CarRecogniser()

vid = cv2.VideoCapture('/home/ignisor/dev/tf_lpr/tf_lpr/car_recognition/data/test/test_video.mp4')

i = 0
while vid.isOpened():
    i += 1
    ret, frame = vid.read()

    frame = frame[300:800, 300:1000]

    cv2.imshow('video', frame)

    cv2.waitKey(1)

    img = Image.fromarray(frame)
    img = img.resize((64, 64))
    img = img.convert('RGB')

    is_car = cr.is_car(img)

    print(i, is_car)
    if is_car:
        while True:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('s'):
                save_img = Image.fromarray(frame)
                save_img.save(f'/home/ignisor/dev/tf_lpr/tf_lpr/car_recognition/data/test/test_frame_{random.randint(0, 999999)}.jpg')
                break

vid.release()
cv2.destroyAllWindows()
