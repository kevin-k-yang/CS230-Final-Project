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


# Model.py
def main():
    print("Welcome to model.py")
    data_dir = "images"
    img_height,img_width=180,180
    batch_size=32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return 0

if __name__ == "__main__":
    main()