from keras.models import model_from_json
import Image
import numpy as np
import matplotlib.pyplot as plt


def model_test():
    autoencoder = model_from_json(open('my_model_architecture.json').read())
    autoencoder.load_weights('my_model_weights.h5')

    im = Image.open('/home/wuxie/PycharmProjects/ImageSearch/img/train/2244.jpg')
    im = im.resize((224, 224), Image.ANTIALIAS)
    im = im.convert('L')
    ax = plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    data = np.reshape(im, [-1, 224, 224, 1])
    decoded_imgs = autoencoder.predict(data)
    ax = plt.subplot(2, 1, 2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(decoded_imgs.reshape(224, 224))
    plt.savefig('im.jpg', format='jpeg')
    plt.show()


if __name__ == '__main__':
    model_test()
