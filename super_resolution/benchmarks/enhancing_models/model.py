from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def enhance(self, img):
        """
        Enhances the input image.

        :param img: the input image.

        :return:    the enhanced image, which is 2x larger than the original image.
        """
        pass


