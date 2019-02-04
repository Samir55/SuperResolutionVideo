import sys
import os
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import rescale

from enhancing_models.upscale import Upscale
from enhancing_models.waifu import Waifu
from enhancing_models.carn import Carn
from enhancing_models.srcnn import SRCNN
from enhancing_models.sharpening import Sharpening
from benchmarks.ssim import SSIM
from benchmarks.psnr import PSNR
from benchmarks.mse import MSE


def run(testset_path):
    # Models
    models = [Upscale("Upscale"), Waifu("Waifu"), Carn("Carn"), SRCNN("SRCNN"), Sharpening("Sharpening")]

    # Benchmarks
    benchmarks = [SSIM("SSIM"), PSNR("PSNR"), MSE("MSE")]

    # Scores
    scores_table = {m.name: {b.name: 0 for b in benchmarks} for m in models}

    # Enhance images
    images = os.listdir(testset_path)

    for i in images:
        img = io.imread(testset_path + i)
        img_down_scaled = rescale(img, 0.5, anti_aliasing=True, multichannel=True)

        for m in models:
            img_enhanced = m.enhance(img_down_scaled)

            for b in benchmarks:
                b_score = b.eval(img, img_enhanced)
                scores_table[m.name][b.name] += b_score

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
