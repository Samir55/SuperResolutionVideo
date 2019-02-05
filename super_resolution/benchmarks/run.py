import sys
import os

import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

from enhancing_models.upscale import Upscale
from enhancing_models.waifu import Waifu
from enhancing_models.carn import Carn
from enhancing_models.srcnn import SRCNN
from enhancing_models.opencv_superres import SuperRes
from enhancing_models.sharpening import Sharpening

from benchmarks.ssim import SSIM
from benchmarks.psnr import PSNR
from benchmarks.mse import MSE


def run(testset_path):
    # Models
    models = [
        Upscale(),
        # Waifu(),
        # Carn(),
        # SRCNN(),
        # SuperRes(),
        Sharpening(),
    ]

    # Benchmark metrics
    benchmarks = [
        SSIM(),
        PSNR(),
        MSE(),
    ]

    # Initialize scores
    scores_table = {m.name: {b.name: 0 for b in benchmarks} for m in models}

    # List all files in the test set path
    images = os.listdir(testset_path)
    images.sort()

    # Loop over all the images in the test set
    for i in images:
        if i[0] == ".": continue

        print('Testing', i)

        img = cv.imread(testset_path + i)
        img_down_scaled = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

        cv.imshow('GT', img)

        for m in models:
            print('    >> Running', m.name)

            img_enhanced = m.enhance(img_down_scaled)

            cv.imshow(m.name + 'SR', img_enhanced)

            for b in benchmarks:
                b_score = b.eval(img, img_enhanced)
                scores_table[m.name][b.name] += b_score

                print('        - %s: %0.3f' % (b.name, b_score))

        cv.waitKey(0)

    # Normalize
    for m in models:
        for b in benchmarks:
            scores_table[m.name][b.name] /= len(images)

    print(scores_table)


if __name__ == "__main__":
    """
    sys.argv[1] = testset path (includes the trailing '/')
    """
    run(sys.argv[1])
