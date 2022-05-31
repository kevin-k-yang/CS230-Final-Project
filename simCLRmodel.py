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
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# import images.image_dataset


# Load SimCLR pretrained weights for this task
# This approach is borrowed from https://github.com/google-research/simclr/blob/master/colabs/load_and_inference.ipynb
def main():
    # preprocess images
    data_dir = "images"
    img_height,img_width=180,180
    batch_size=5
    # training set
    # tfds_dataset, tfds_info = tfds.load(
    # 'tf_flowers', split='train', with_info=True)
    # num_images = tfds_info.splits['train'].num_examples
    # num_classes = tfds_info.features['label'].num_classes

    dataset = tf.data.Dataset.list_files("/images/*")
    print(dataset)

    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width), batch_size=batch_size)
    # # validation set
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width), batch_size=batch_size)


    # x = tfds_dataset.batch(batch_size)
    # x = tf1.data.make_one_shot_iterator(x).get_next()

    # load SimCLR model
    hub_path = "gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/"
    module = hub.Module(hub_path, trainable=False)
    print("done with hub shit")
    key = module(inputs=x['image'], signature="default", as_dict=True)
    logits_t = key['logits_sup'][:, :]
    print(key)

    # resnet_model = Sequential()
    # pretrained_model= tf.keras.applications.ResNet50(include_top=False,
    #                 input_shape=(180,180,3),
    #                 pooling='avg',classes=2,
    #                 weights='imagenet')
    # for layer in pretrained_model.layers:
    #         layer.trainable=False
    # resnet_model.add(pretrained_model)

    # # add output layers
    # resnet_model.add(Flatten())
    # resnet_model.add(Dense(512, activation='relu'))
    # resnet_model.add(Dense(1, activation='sigmoid'))
    # resnet_model.summary()

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