import cv2
import numpy as np


class LPProcessor(object):
    def crop(self, lp):
        big_img = cv2.resize(lp, dsize=(256 * 4, 64 * 4), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(big_img, cv2.COLOR_RGB2GRAY)

        img = cv2.GaussianBlur(gray, (9, 9), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        morph_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=15)

        img = cv2.subtract(gray, morph_image)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        img = cv2.Canny(img, 250, 255)

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        new, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        lp_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            # Select the contour with 4 corners
            if len(approx) == 4:
                lp_contour = approx
                break

        cropped = []
        x, y, w, h = cv2.boundingRect(lp_contour)
        if w > 0 and h > 0:
            cropped = big_img[y:y + h, x:x + w]

        return cropped if len(cropped) > 0 else None

    def split(self, lp):
        img = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        img = 255 - img

        proj_x = np.sum(img, axis=0)
        proj_x = proj_x / max(proj_x)

        # normalize
        proj_x_norm = []
        for i in range(0, len(proj_x), 5):
            part = proj_x[i:i+5]
            avg = sum(part)/5
            for _ in part:
                proj_x_norm.append(avg)

        proj_x = np.array(proj_x_norm)

        avg = sum(proj_x)/len(proj_x)
        avg = avg/2

        parts = []
        part = [0, 0]
        sum_size = 0
        for x, value in enumerate(proj_x):
            if value > avg and not part[0]:
                part[0] = x
            elif value < avg and part[0]:
                part[1] = x
                parts.append(part)
                sum_size += part[1] - part[0]
                part = [0, 0]

        if len(parts) <= 0:
            return []

        avg_size = sum_size/len(parts)

        letters = []
        for part in parts:
            if part[1] - part[0] > avg_size/10:
                letter_img = img[:, part[0]:part[1]]
                h, w = letter_img.shape
                # check image ratio filter images with wrong ratio
                if 1.0 <= h/w <= 4.0:
                    letters.append(letter_img)

        return letters


    def dbg_img_show(self, img):
        cv2.imshow('img', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('n'):
                # cv2.destroyAllWindows()
                break
