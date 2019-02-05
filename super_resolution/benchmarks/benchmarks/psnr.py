from skimage.measure import compare_psnr as psnr

from benchmarks.benchmark import Benchmark


class PSNR(Benchmark):

    def __init__(self):
        super().__init__("PNSR")

    def eval(self, img_orig, img_enhanced):
        return psnr(img_orig, img_enhanced)
