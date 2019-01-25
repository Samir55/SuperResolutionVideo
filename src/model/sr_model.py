# import the necessary packages
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

from src.data.data_loader import read_training, get_train_file_names, UPCONVDataGenerator, read_training_upconv
import cv2 as cv
import numpy as np
import time

train_path = "/home/ahmedsamir/SuperResolutionVideo/data/raw/"
upconv_train_path = "/home/ahmedsamir/Downloads/ukbench/ukbench/full/"
model_test_path = "/home/ahmedsamir/SuperResolutionVideo/models/"


def sr_loss():
    return None


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


class SRCNN:
    NUM_EPOCHS = 1500

    def sr_model(self):
        input_shape = [None, None, 1]

        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(9, 9), padding="same",
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None),
                         input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Conv2D(filters=32, kernel_size=(1, 1), padding="same",
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)))
        model.add(Activation("relu"))

        model.add(Conv2D(1, kernel_size=(3, 3), padding="same",
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)))

        # Print model layers for debugging.
        print(model.summary())
        plot_model(model, to_file="SRCNN.png", show_shapes=True, show_layer_names=True)

        return model

    def train(self, load_from_checkpoint=False, checkpoint=30):
        model = self.sr_model()

        opt = Adam(lr=0.001)  # decay.
        model.compile(loss="mse", optimizer=opt, metrics=[PSNRLoss])

        if load_from_checkpoint:
            model.load_weights("/home/ahmedsamir/SuperResolutionVideo/models/sr_model" + str(checkpoint) + ".h5")

        checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='train_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')

        callbacks = [checkpoint]

        train_images = get_train_file_names(train_path)

        train_x, train_y = read_training(train_path, train_images)

        model.fit(train_x, train_y, batch_size=128, epochs=SRCNN.NUM_EPOCHS, callbacks=callbacks, verbose=1)

        model.save("/home/ahmedsamir/SuperResolutionVideo/models/sr_model" + ".h5")

    def predict(self, img_path, model_file_name):
        model = self.sr_model()
        model.load_weights(model_test_path + model_file_name + ".h5")

        img = cv.imread(img_path, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        shape = img.shape

        y_img = cv.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv.INTER_CUBIC)
        y_img = cv.resize(y_img, (shape[1], shape[0]), cv.INTER_CUBIC)
        img[:, :, 0] = y_img

        img = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)

        cv.imshow("UP_SCALED", img[:,:,0])
        cv.waitKey(0)

        x = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        x[0, :, :, 0] = y_img.astype(float) / 255.

        y = model.predict(x, batch_size=1) * 255.
        y[y[:] > 255] = 255
        y[y[:] < 0] = 0
        y = y.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        img[:, :, 0] = y[0, :, :, 0]
        img = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)

        cv.imshow("SR", img[:,:,0])
        cv.waitKey(0)


class UPCONV:
    NUM_CHANNELS = 3
    NUM_EPOCHS = 5
    BATCH_SIZE = 32

    def upconv_model(self):
        input_shape = [None, None, UPCONV.NUM_CHANNELS]

        model = Sequential()

        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", input_shape=input_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=3, kernel_size=(3, 3), padding="same"))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

        # Print model layers for debugging.
        print(model.summary())
        plot_model(model, to_file="UPCONV.png", show_shapes=True, show_layer_names=True)
        return model

    def train(self):
        model = self.upconv_model()

        opt = Adam(lr=0.001)  # decay.
        model.compile(loss="mse", optimizer=opt, metrics=[PSNRLoss])

        callbacks = []

        train_images = get_train_file_names(upconv_train_path)

        # training_generator = UPCONVDataGenerator(upconv_train_path, train_images)
        # model.fit_generator(generator=training_generator)

        for e in range(UPCONV.NUM_EPOCHS):
            t = time.clock()
            for i in range(len(train_images) // UPCONV.BATCH_SIZE):
                train_x, train_y = read_training_upconv(upconv_train_path,
                                                 train_images[i * UPCONV.BATCH_SIZE: (i + 1) * UPCONV.BATCH_SIZE])
                model.fit(train_x, train_y, batch_size=32, epochs=1)
            print("ONE EPOCH time is " + str(time.clock() - t))
            model.save("/home/ahmedsamir/SuperResolutionVideo/models/upconv_model" + ".h5")

    def predict(self, img_path, model_file_name):
        model = self.upconv_model()
        model.load_weights(model_test_path + model_file_name + ".h5")

        org_img = cv.imread(img_path)
        h, w, c = org_img.shape

        down_scaled_img = cv.resize(org_img, (w // 2, h // 2), interpolation=cv.INTER_CUBIC)
        up_scaled_img = cv.resize(org_img, (w , h), interpolation=cv.INTER_CUBIC)

        x = np.asarray(np.asarray(up_scaled_img))
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

        y = model.predict(x)

        result_img = y[0, :, :, :]

        cv.imshow("UP_SCALED", up_scaled_img)
        cv.waitKey(0)

        cv.imshow("UP_CONV", result_img)
        cv.waitKey(0)


# SRCNN
# SRCNN().train()
# SRCNN().predict1("/home/ahmedsamir/SuperResolutionVideo/test/d.jpg", "sr_model")

# UPCONV
UPCONV().train()
# UPCONV().predict("/home/ahmedsamir/SuperResolutionVideo/test/c.jpg", "upconv_model")
