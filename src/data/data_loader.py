import os
import cv2 as cv
import numpy as np


def get_train_file_names(train_path):
    # Walk on data set directory.
    for root, dirs, files in os.walk(train_path):
        #
        # Loop on every file in the directory.
        #
        return files


def read_training(train_path, file_names):
    train_x = []
    train_y = []

    for filename in file_names:
        # Ignore git ignore file.
        if filename[0] == '.':
            continue

        # Read image in gray scale.
        org_img = cv.imread(train_path + '/' + filename)

        # Scale down to hd (ground truth).
        y = cv.resize(org_img, (1280, 720))

        # Scale down to hd (input).
        x = cv.resize(org_img, (640, 360))
        x = cv.resize(x, (1280, 720), interpolation=cv.INTER_CUBIC)

        train_x.append(x)
        train_y.append(y)

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    return (train_x, train_y)
