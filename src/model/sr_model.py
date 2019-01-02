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
    input_shape = [720, 1280, 3]

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(9, 9), padding="same", kernel_initializer="he_normal",
                     input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(filters=32, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal"))

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
    model.load_weights("/home/ahmedsamir/SuperResolutionVideo/models/sr_model.h5")

    # Read image in gray scale.
    org_img = cv.imread(img_path)

    # Scale down to hd (input).
    x = cv.resize(org_img, (640, 360))
    x = cv.resize(x, (1280, 720), interpolation=cv.INTER_CUBIC)
    x = np.asarray(x)
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    y = model.predict(x)

    cv.imshow("SR", x[0, :, :, :])
    cv.waitKey(0)

    cv.imshow("SR", y[0, :, :, :])
    cv.waitKey(0)


train()
# predict("/home/ahmedsamir/SuperResolutionVideo/test/t1.JPEG")
# predict("/home/ahmedsamir/SuperResolutionVideo/data/raw/1.jpg")
