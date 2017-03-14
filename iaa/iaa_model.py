from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import os
import os.path
from PIL import ImageFile
import h5py
from keras.models import model_from_json
import matplotlib.pyplot as plt


def iaa_model():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    input_img = Input(shape=(224, 224, 3))

    base_model = VGG16(weights='imagenet', include_top=True)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    pred = Dense(2, activation='softmax')(x)

    model = Model(base_model.input, pred)
    for layer in base_model.layers:
        if layer.name != 'fc1' and layer.name != 'fc2':
            layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=4,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        'validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)

    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')

    return 0


if __name__ == '__main__':
    iaa_model()
