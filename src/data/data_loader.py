import os
import cv2 as cv
import numpy as np


def delete_bad_images(train_path):
    # Walk on data set directory.
    for root, dirs, files in os.walk(train_path):
        #
        # Loop on every file in the directory.
        #
        for filename in files:
            if filename[0] == '.':
                continue

            # Read image in gray scale.
            org_img = cv.imread(train_path + '/' + filename)
            w, h, c = org_img.shape

            if w < 512 or h < 512:
                os.remove(train_path + '/' + filename)


def get_train_file_names(train_path):
    files_names = []
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
        org_img = cv.imread(train_path + '/' + filename, 0)
        # converted_img = cv.cvtColor(org_img, cv.COLOR_BGR2YCrCb)
        converted_img = org_img / 255.

        sub_image_size = 32
        stride = 14
        for i in range(0, org_img.shape[0] - sub_image_size, stride):
            if i + sub_image_size > org_img.shape[0]:
                break

            for j in range(0, org_img.shape[1] - sub_image_size, stride):
                if j + sub_image_size > org_img.shape[1]:
                    break

                csub_image = converted_img[i:i + sub_image_size, j:j + sub_image_size]
                h, w = csub_image.shape

                # Scale down to (input).
                x = cv.resize(csub_image, (h // 2, w // 2))
                x = cv.resize(x, (h, w), interpolation=cv.INTER_CUBIC)

                train_x.append(np.asarray(x[:, :]).reshape(h, w, 1))
                train_y.append(np.asarray(csub_image[:, :]).reshape(h, w, 1))

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    return train_x, train_y
