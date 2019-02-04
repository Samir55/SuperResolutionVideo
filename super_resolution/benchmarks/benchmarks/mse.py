from skimage.measure import compare_mse as mse

from benchmark import Benchmark


class MSE(Benchmark):

    def __init__(self, name):
        super(MSE, self).__init__(name)

    def eval(self, img_orig, img_enhanced):
        return mse(img_orig, img_enhanced)
