from skimage.measure import compare_ssim as ssim

from benchmarks.benchmark import Benchmark


class SSIM(Benchmark):

    def __init__(self):
        super().__init__("SSIM")

    def eval(self, img_orig, img_enhanced):
        return ssim(img_orig, img_enhanced, multichannel=True)
