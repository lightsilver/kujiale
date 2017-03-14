from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import os
import os.path
from PIL import ImageFile
import json


def img2input(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def calcSimilarity(features1, features2):
    a = 0.
    for i in range(4096):
        a += (features1[0][i] - features2[0][i]) * (features1[0][i] - features2[0][i])

    return a / 4096


def s():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model_vgg16 = VGG16(weights='imagenet', include_top=True)
    model = Model(input=model_vgg16.input, output=model_vgg16.get_layer('fc2').output)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    x = img2input('/home/wuxie/Desktop/2421.jpg')
    features = model.predict(x)
    print(features.shape)

    x = img2input('/home/wuxie/Desktop/2348.jpg')
    features2 = model.predict(x)

    print(calcSimilarity(features, features2))
    """
    dir = "img/train"
    fingerprint_file = open('fingerprint_file.json', 'w+')
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            x = img2input(os.path.join(dir, filename))
            features = model.predict(x)
            dump_list = features.tolist()
            img_num = filename.split('.')[0]
            dump_list.append(img_num)

            json.dump(dump_list, fingerprint_file)
            # in_json = json.dumps(dump_list)
            # fingerprint_file.write(in_json)
            # write_str = ' '.join([str(filename), features.tostring(), '\n'])
            # fingerprint_file.write(write_str)
    """


if __name__ == '__main__':
    s()
