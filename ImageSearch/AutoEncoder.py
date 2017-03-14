from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

import h5py
from keras.models import model_from_json
import Image
import numpy as np
import matplotlib.pyplot as plt


def AutoEncoder():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    input_img = Input(shape=(224, 224, 1))

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator_ = train_datagen.flow_from_directory('..', color_mode="grayscale",
                                                         class_mode=None,
                                                         batch_size=50, target_size=(224, 224))

    def train_generator():
        for x in train_generator_:
            yield x, x

    validation_generator_ = test_datagen.flow_from_directory(
        '..',
        target_size=(224, 224),
        batch_size=50,
        class_mode=None,
        color_mode="grayscale")

    def validation_generator():
        for x in validation_generator_:
            yield x, x

    autoencoder.fit_generator(
        train_generator(),
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator(),
        nb_val_samples=800)

    json_string = autoencoder.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    autoencoder.save_weights('my_model_weights.h5')

    return 0


if __name__ == '__main__':
    AutoEncoder()
