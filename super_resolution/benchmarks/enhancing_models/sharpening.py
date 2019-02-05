import sys
import os

from enhancing_models.model import Model


class Sharpening(Model):

    def __init__(self):
        super().__init__("Sharpening")

    def enhance(self, img):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ], dtype=np.float32) / 9

        img = cv.resize(img, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
        res = cv.filter2D(img, -1, kernel)
        ret = cv.add(img, res)

        return ret
