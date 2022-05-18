# Authors: Aryan Chaudhary (achaud@stanford.edu)
#          Kevin K Yang    (kevvyang@stanford.edu)
#          Brian Wu        (brian.wu@stanford.edu)
#
# Description: This file contains the CNN (Convolutional Neural Network) model used
#

# imports
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import csv
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# map youtube ID to views with data from viewcounts.csv
# def parse_view_counts(link):
#     view_count_map = {}
#     with open(link) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         first_line = True
#         for row in csv_reader:
#             if not first_line:
#                 view_count_map[row[0]] = int(row[1])
#             else:
#                 first_line = False
#     return view_count_map


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
        image_size=(img_height, img_width), batch_size=batch_size)
    # validation set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width), batch_size=batch_size)

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
    resnet_model.add(Dense(1, activation='sigmoid'))
    resnet_model.summary()

    # train model
    resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)

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
    
    print("Welcome to model.py")
    return 0

if __name__ == "__main__":
    main()