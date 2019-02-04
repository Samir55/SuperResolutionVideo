from skimage.measure import compare_psnr as psnr

from benchmark import Benchmark


class PSNR(Benchmark):

    def __init__(self, name):
        super(PSNR, self).__init__(name)

    def eval(self, img_orig, img_enhanced):
        return psnr(img_orig, img_enhanced)
