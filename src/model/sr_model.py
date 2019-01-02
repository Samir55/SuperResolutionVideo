# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.optimizers import Adam

from src.data.data_loader import read_training, get_train_file_names
import cv2 as cv
import numpy as np
import time

train_path = "/home/ahmedsamir/SuperResolutionVideo/data/raw/train"


def sr_loss():
    return None


def sr_model():
    input_shape = [720, 1280, 1]

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(9, 9), padding="same", kernel_initializer="he_normal",
                     input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(filters=32, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))

    model.add(Conv2D(1, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal"))

    # Print model layers for debugging.
    print(model.summary())

    return model


def train(load_from_checkpoint=False, checkpoint=30):
    model = sr_model()

    opt = Adam(lr=0.001, decay=0.001 / 10)  # decay.
    model.compile(loss="mse", optimizer=opt)

    start = 0
    if load_from_checkpoint:
        model.load_weights("/home/ahmedsamir/SuperResolutionVideo/models/sr_model" + str(checkpoint) + ".h5")
        start = checkpoint

    callbacks = []

    train_images = get_train_file_names(train_path)

    for i in range(start + 1, len(train_images) // 200):
        batch_start = time.clock()
        train_x, train_y = read_training(train_path, train_images[i * 200: i * 200 + 200])
        model.fit(train_x, train_y, batch_size=2, epochs=1, callbacks=callbacks, verbose=1)
        if i > 0 and i % 5 == 0:
            model.save("/home/ahmedsamir/SuperResolutionVideo/models/sr_model" + str(i) + ".h5")

        print("Batch of 200 took ", time.clock() - batch_start, " seconds!")
        print(i * 200 + 200, '/' + str(len(train_images)) + " processed")


def predict(img_path):
    model = sr_model()
    model.load_weights("/home/ahmedsamir/SuperResolutionVideo/models/sr_model" + str(10) + ".h5")

    # Read image in gray scale.
    org_img = cv.imread(img_path)

    # Scale down to hd (input).
    x = cv.resize(org_img, (640, 360))
    x = cv.resize(x, (1280, 720), interpolation=cv.INTER_CUBIC)
    org_img = x.copy()

    converted_x = cv.cvtColor(x, cv.COLOR_BGR2YCrCb)

    x = np.asarray(np.asarray(converted_x[:, :, 0]).reshape(720, 1280, 1))
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    y = model.predict(x)

    converted_x[:, :, 0] = y[0, :, :, 0]
    converted_x = cv.cvtColor(converted_x, cv.COLOR_YCrCb2BGR)

    # cv.imshow("SR", org_img)
    # cv.waitKey(0)

    # cv.imshow("SR", converted_x)
    # cv.waitKey(0)


train()
# train_images = get_train_file_names(train_path)
# predict(train_path + '/' + train_images[0])
# predict("/home/ahmedsamir/SuperResolutionVideo/data/raw/1.jpg")
