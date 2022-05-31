# Authors: Aryan Chaudhary (achaud@stanford.edu)
#          Kevin K Yang    (kevvyang@stanford.edu)
#          Brian Wu        (brian.wu@stanford.edu)
#
# Description: This file contains the CNN we built using the SimCLR pretrained model
#

# imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
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
        label_mode="categorical",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width), batch_size=batch_size)
    # validation set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode="categorical",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width), batch_size=batch_size)

    # load simCLR model
    hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
    model = hub.KerasLayer(hub_path, trainable=False)
    simclr_model = Sequential()
    simclr_model.add(model)
    simclr_model.add(Flatten())
    simclr_model.add(Dropout(0.2, input_shape=(180*180,)))
    simclr_model.add(Dense(512, activation='relu'))
    # simclr_model.add(Dense(1, activation="sigmoid"))
    simclr_model.add(Dense(4, activation='softmax'))

    # train model
    simclr_model.compile(optimizer=Adam(learning_rate=0.001),loss='CategoricalCrossentropy',metrics=['accuracy'])
    # simclr_model.compile(optimizer=Adam(learning_rate=0.001),loss='BinaryCrossentropy',metrics=['accuracy'])
    history = simclr_model.fit(train_ds, validation_data=val_ds, epochs=10)

    # display results
    fig1 = plt.gcf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.axis(ymin=0.4,ymax=1)
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()