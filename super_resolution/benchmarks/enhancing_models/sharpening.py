from skimage.transform import rescale, resize, downscale_local_mean
from model import Model

class Sharpening(Model):

    def __init__(self, name):
        super(Sharpening, self).__init__(name)

    def enhance(self, img):
        # ToDo: enhance img
        return rescale(img, 2.0, anti_aliasing=True, multichannel=True)

