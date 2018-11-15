from keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalAveragePooling2D, Dense, concatenate
from keras.layers.noise import GaussianNoise
from keras.layers.core import Activation

from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer

from keras import initializers
from keras import regularizers
from keras import constraints


class MeanOnlyBatchNormalization(Layer):
    def __init__(self,
                 momentum=0.999,
                 moving_mean_initializer='zeros',
                 axis=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.moving_mean_initializer = moving_mean_initializer
        self.axis = axis

    def build(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        # inference
        def normalize_inference():
            return inputs - self.moving_mean

        if training in {0, False}:
            return normalize_inference()

        mean = K.mean(inputs, axis=reduction_axes)
        normed_training = inputs - mean

        self.add_update(K.moving_average_update(self.moving_mean, mean,
                                                self.momentum), inputs)

        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Bias(Layer):
    def __init__(self,
                 filters,
                 data_format=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        self.filters = filters
        self.data_format = data_format
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = K.bias_add(
            inputs,
            self.bias,
            data_format=self.data_format)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def WN_Conv2D(net, filters=None, kernel_size=None, padding='same', kernel_initializer='he_norml'):
    """Convolution layer with Weight Normalization + Mean-Only BatchNormalization"""
    net = Conv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer,
                 use_bias=False)(net)
    net = MeanOnlyBatchNormalization()(net)
    net = Bias(filters)(net)
    net = LeakyReLU(alpha=0.1)(net)

    return net


def WN_Dense(net, units=None, kernel_initializer='he_norml'):
    """Dense layer with Weight Normalization + Mean-Only BatchNormalization"""
    net = Dense(units=units, activation=None, kernel_initializer=kernel_initializer, use_bias=False)(net)
    net = MeanOnlyBatchNormalization()(net)
    net = Bias(units)(net)
    net = Activation('softmax')(net)

    return net


def build_model(num_class):
    input_img = Input(shape=(32, 32, 3))
    supervised_label = Input(shape=(10,))
    supervised_flag = Input(shape=(1,))
    unsupervised_weight = Input(shape=(1,))

    kernel_init = 'he_normal'

    net = GaussianNoise(stddev=0.15)(input_img)

    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = WN_Conv2D(net, 512, (3, 3), padding='valid', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (1, 1), padding='valid', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (1, 1), padding='valid', kernel_initializer=kernel_init)
    net = GlobalAveragePooling2D()(net)
    net = WN_Dense(net, units=num_class, kernel_initializer=kernel_init)

    # concatenate label
    net = concatenate([net, supervised_label, supervised_flag, unsupervised_weight])

    # pred(num_class), unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised_weight(1)
    return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], net)
