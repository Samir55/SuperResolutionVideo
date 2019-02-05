from abc import ABCMeta, abstractmethod


class Benchmark(metaclass=ABCMeta):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def eval(self, img_orig, img_enhanced):
        """
        Compares the original image with the enhanced image. Both images must have the
        same dimensions.

        :param img_orig:        the original image.
        :param img_enhanced:    the enhanced image.

        :return:                the comparison score.
        """
        pass


