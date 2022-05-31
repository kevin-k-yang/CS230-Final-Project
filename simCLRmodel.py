# Authors: Aryan Chaudhary (achaud@stanford.edu)
#          Kevin K Yang    (kevvyang@stanford.edu)
#          Brian Wu        (brian.wu@stanford.edu)
#
# Description: This file contains the CNN we built using the SimCLR pretrained model
#

# imports
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Load simCLR pretrained weights for this task
def main():
    # preprocess images
    data_dir = "images"
    img_height,img_width=180,180
    batch_size=5
    # training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width), batch_size=batch_size)
    # validation set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width), batch_size=batch_size)

    # load simCLR model
    hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
    model = hub.KerasLayer(hub_path, trainable=False)
    
    nnclr_model = Sequential()
    nnclr_model.add(model)
    nnclr_model.add(Flatten())
    nnclr_model.add(Dense(512, activation='relu'))
    nnclr_model.add(Dense(1, activation='sigmoid'))

    # # train model
    # resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    # history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)

    # # display results
    # fig1 = plt.gcf()
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.axis(ymin=0.4,ymax=1)
    # plt.grid()
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend(['train', 'validation'])
    # plt.show()
    
    return 0

if __name__ == "__main__":
    main()