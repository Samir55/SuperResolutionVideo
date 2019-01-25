import os
import cv2 as cv
import keras
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
    data = []
    label = []

    for filename in file_names:
        # Ignore git ignore file.
        if filename[0] == '.':
            continue

        # Read image in gray scale.
        hr_img = cv.imread(train_path + '/' + filename)
        hr_img = cv.cvtColor(hr_img, cv.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape

        lr_img = cv.resize(hr_img, (shape[1] // 2, shape[0] // 2))
        lr_img = cv.resize(lr_img, (shape[1], shape[0]))

        BLOCK_STEP = 16
        BLOCK_SIZE = 32
        PATCH_SIZE = 32
        LABEL_SIZE = 32

        width_limit = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
        height_limit = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP

        for w in range(width_limit):
            for h in range(height_limit):
                x = w * BLOCK_STEP
                y = h * BLOCK_STEP

                hr_patch = hr_img[x: x + BLOCK_SIZE, y:y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y:y + BLOCK_SIZE]

                if lr_patch.shape != (PATCH_SIZE, PATCH_SIZE) or hr_patch.shape != (BLOCK_SIZE, BLOCK_SIZE):
                    continue

                lr_patch = lr_patch / 255.
                hr_patch = hr_patch / 255.

                lr = np.zeros((1, PATCH_SIZE, PATCH_SIZE), dtype=np.double)
                hr = np.zeros((1, LABEL_SIZE, LABEL_SIZE), dtype=np.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch

                data.append(np.asarray(lr[0, :, :]).reshape(PATCH_SIZE, PATCH_SIZE, 1))
                label.append(np.asarray(hr[0, :, :]).reshape(LABEL_SIZE, LABEL_SIZE, 1))

    data = np.asarray(data)
    label = np.asarray(label)
    return data, label


def read_training_upconv(train_path, file_names):
    data = []
    label = []

    for filename in file_names:
        # Ignore git ignore file.
        if filename[0] == '.' or filename[-3:] != "jpg":
            continue

        # Read image in gray scale.
        hr_img = cv.imread(train_path + filename)
        shape = hr_img.shape

        lr_img = cv.resize(hr_img, (shape[1] // 2, shape[0] // 2))
        lr_img = cv.resize(hr_img, (shape[1], shape[0]))

        BLOCK_STEP = 96
        BLOCK_SIZE = 96
        PATCH_SIZE = 96
        LABEL_SIZE = 96

        width_limit = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
        height_limit = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP

        for w in range(width_limit):
            for h in range(height_limit):
                x = w * BLOCK_STEP
                y = h * BLOCK_STEP

                hr_patch = hr_img[x: x + BLOCK_SIZE, y:y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y:y + BLOCK_SIZE]

                if lr_patch.shape != (PATCH_SIZE, PATCH_SIZE, 3) or hr_patch.shape != (BLOCK_SIZE, BLOCK_SIZE,3):
                    continue

                lr_patch = lr_patch / 255.
                hr_patch = hr_patch / 255.

                lr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.double)
                hr = np.zeros((1, LABEL_SIZE, LABEL_SIZE, 3), dtype=np.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch

                data.append(np.asarray(lr[0, :, :]).reshape(PATCH_SIZE, PATCH_SIZE, 3))
                label.append(np.asarray(hr[0, :, :]).reshape(LABEL_SIZE, LABEL_SIZE, 3))

    data = np.asarray(data)
    label = np.asarray(label)
    return data, label


class UPCONVDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_path, file_names, batch_size=5, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.filenames = file_names
        self.data_path = data_path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        file_names_list = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(file_names_list)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_names_list):
        # Generates data containing batch_size samples
        X = []
        y = []

        for filename in file_names_list:
            # Ignore git ignore file.
            if filename[0] == '.' or filename[-3:] != 'jpg':
                continue

            # Read image in gray scale.
            org_img = cv.imread(self.data_path + '/' + filename)
            # converted_img = cv.cvtColor(org_img, cv.COLOR_BGR2YCrCb)
            converted_img = org_img / 255.

            sub_image_size = 96
            stride = 96
            for i in range(0, org_img.shape[0] - sub_image_size, stride):
                if i + sub_image_size > org_img.shape[0]:
                    break

                for j in range(0, org_img.shape[1] - sub_image_size, stride):
                    if j + sub_image_size > org_img.shape[1]:
                        break

                    csub_image = converted_img[i:i + sub_image_size, j:j + sub_image_size]
                    h, w, c = csub_image.shape

                    # Scale down to (input).
                    x = cv.resize(csub_image, (h // 2, w // 2))
                    h, w, c = x.shape
                    # x = cv.resize(x, (h, w), interpolation=cv.INTER_CUBIC)

                    X.append(np.asarray(x[:, :]).reshape(h, w, 3))
                    y.append(np.asarray(csub_image[:, :]).reshape(h * 2, w * 2, 3))

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y
