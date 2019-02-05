import numpy as np
import cv2 as cv

from enhancing_models.model import Model


class SuperRes(Model):

    def __init__(self):
        super().__init__("OpenCV Super Resolution")

    def enhance(self, img):
        # ToDo: enhance img
        return cv.resize(img, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
