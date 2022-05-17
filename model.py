# Authors: Aryan Chaudhary (achaud@stanford.edu)
#          Kevin K Yang    (kevvyang@stanford.edu)
#          Brian Wu        (brian.wu@stanford.edu)
#
# Description: This file contains the CNN (Convolutional Neural Network) model used
#

# imports
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.models import Model
# from keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from kt_utils import *

# import keras.backend as K
# K.set_image_data_format('channels_last')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Implement ResNet 50 pretrained weights for this task
# This approach is borrowed from https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
def main():
    # preprocess images
    data_dir = "images"
    img_height,img_width=180,180
    batch_size=32
    # training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # validation set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # load resnet model
    resnet_model = Sequential()
    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=(180,180,3),
                    pooling='avg',classes=5,
                    weights='imagenet')
    for layer in pretrained_model.layers:
            layer.trainable=False
    resnet_model.add(pretrained_model)

    # add output layers
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(1, activation='linear'))
    resnet_model.summary()

    # train model
    resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error',metrics=['accuracy'])
    history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)
    
    print("Welcome to model.py")
    return 0

if __name__ == "__main__":
    main()