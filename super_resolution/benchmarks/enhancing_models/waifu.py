from skimage.transform import rescale, resize, downscale_local_mean

from model import Model

class Waifu(Model):

    def __init__(self, name):
        super(Waifu, self).__init__(name)

    def enhance(self, img):
        # ToDo: enhance img
        return rescale(img, 2.0, anti_aliasing=True, multichannel=True)

