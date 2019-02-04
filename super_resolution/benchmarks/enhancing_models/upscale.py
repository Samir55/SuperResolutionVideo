from skimage.transform import rescale, resize, downscale_local_mean
from model import Model

class Upscale(Model):

    def __init__(self, name):
        super(Upscale, self).__init__(name)

    def enhance(self, img):
        return rescale(img, 2.0, anti_aliasing=True, multichannel=True)

