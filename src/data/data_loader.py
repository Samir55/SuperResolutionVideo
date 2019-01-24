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
    train_x = []
    train_y = []

    for filename in file_names:
        # Ignore git ignore file.
        if filename[0] == '.' or filename[-3:] != 'jpg':
            continue

        # Read image in gray scale.
        org_img = cv.imread(train_path + '/' + filename)
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

                train_x.append(np.asarray(x[:, :]).reshape(h, w, 3))
                train_y.append(np.asarray(csub_image[:, :]).reshape(h * 2, w * 2, 3))

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    return train_x, train_y


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
