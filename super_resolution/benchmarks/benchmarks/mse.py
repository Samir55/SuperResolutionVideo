from skimage.measure import compare_mse as mse

from benchmarks.benchmark import Benchmark


class MSE(Benchmark):

    def __init__(self,):
        super().__init__("MSE")

    def eval(self, img_orig, img_enhanced):
        return mse(img_orig, img_enhanced)
