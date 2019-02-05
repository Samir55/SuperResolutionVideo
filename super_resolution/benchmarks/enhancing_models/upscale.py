import numpy as np
import cv2 as cv

from enhancing_models.model import Model


class Upscale(Model):

    def __init__(self):
        super(Upscale, self).__init__("Upscale")

    def enhance(self, img):
        return cv.resize(img, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)

