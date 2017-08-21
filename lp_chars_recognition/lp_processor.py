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

        return cropped

    def split(self, lp):
        img = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)

        proj_x = np.sum(img, axis=0)
        proj_x = proj_x / max(proj_x)

        print(proj_x)
        self.dbg_img_show(img)


    def dbg_img_show(self, img):
        cv2.imshow('img', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('n'):
                # cv2.destroyAllWindows()
                break
