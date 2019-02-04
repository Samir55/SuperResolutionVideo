from skimage.measure import compare_ssim as ssim

from benchmark import Benchmark


class SSIM(Benchmark):

    def __init__(self, name):
        super(SSIM, self).__init__(name)

    def eval(self, img_orig, img_enhanced):
        return ssim(img_orig, img_enhanced, multichannel=True)
