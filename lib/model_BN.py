from keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalAveragePooling2D, Dense, concatenate, \
    BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.core import Activation
from keras.models import Model


def build_model(num_class):
    input_img = Input(shape=(32, 32, 3))
    supervised_label = Input(shape=(10,))
    supervised_flag = Input(shape=(1,))
    unsupervised_weight = Input(shape=(1,))

    kernel_init = 'he_normal'

    net = GaussianNoise(stddev=0.15)(input_img)

    net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = Conv2D(512, (3, 3), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(256, (1, 1), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(128, (1, 1), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.1)(net)

    net = GlobalAveragePooling2D()(net)
    net = Dense(units=num_class, activation=None, kernel_initializer=kernel_init)(net)
    net = BatchNormalization()(net)
    net = Activation('softmax')(net)

    # concate label
    net = concatenate([net, supervised_label, supervised_flag, unsupervised_weight])

    # pred(num_class), unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised_weight(1)
    return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], net)
