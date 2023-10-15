import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Concatenate, LeakyReLU, MaxPool2D, UpSampling2D, Input
from var import *

class LinearTransform(tf.keras.layers.Layer):
    def __init__(self, name="LinearTransform"):
        super(LinearTransform, self).__init__(name=name)
        self.patch_size = patch_size

        self.conv_r = Conv2D(1, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv_g = Conv2D(1, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv_b = Conv2D(1, kernel_size=3, strides=1, padding='same', use_bias=False)

        self.pool_rc = AveragePooling2D(pool_size=(patch_size, patch_size), strides=1)
        self.pool_gc = AveragePooling2D(pool_size=(patch_size, patch_size), strides=1)
        self.pool_bc = AveragePooling2D(pool_size=(patch_size, patch_size), strides=1)

        self.bn = BatchNormalization()
        self.sigmoid = Activation('sigmoid')
        self.softmax = Activation('softmax')

    def call(self, input, training=True):
        r, g, b = input[:, :, :, 0:1], input[:, :, :, 1:2], input[:, :, :, 2:3]

        rs = self.conv_r(r)
        gs = self.conv_g(g)
        bs = self.conv_r(b)

        rc = tf.reshape(self.pool_rc(rs), [-1, 1])
        gc = tf.reshape(self.pool_gc(gs), [-1, 1])
        bc = tf.reshape(self.pool_bc(bs), [-1, 1])

        merge = Concatenate(axis=-1)([rc, gc, bc])
        merge = tf.expand_dims(merge, axis=1)
        merge = tf.expand_dims(merge, axis=1)
        merge = self.softmax(merge)
        merge = tf.repeat(merge, repeats=self.patch_size, axis=2)
        merge = tf.repeat(merge, repeats=self.patch_size, axis=1)

        r = r * (1 + self.sigmoid(rs))
        g = g * (1 + self.sigmoid(gs))
        b = b * (1 + self.sigmoid(bs))

        output = self.bn(merge[:, :, :, 0:1] * r + merge[:, :, :, 1:2] * g + merge[:, :, :, 2:3], training=training)
        return output

    def get_config(self):
        config = super(LinearTransform, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config
		
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, out_ch, residual_path=False, stride=1, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.residual_path = residual_path
        self.stride = stride

        self.conv1 = Conv2D(out_ch, kernel_size=3, strides=stride, padding='same', use_bias=False,
                            data_format="channels_last")
        self.bn1 = BatchNormalization()
        self.relu1 = LeakyReLU()

        self.conv2 = Conv2D(out_ch, kernel_size=3, strides=1, padding='same', use_bias=False,
                            data_format="channels_last")
        self.bn2 = BatchNormalization()

        if residual_path:
            self.conv_shortcut = Conv2D(out_ch, kernel_size=1, strides=stride, padding='same', use_bias=False)
            self.bn_shortcut = BatchNormalization()

        self.relu2 = LeakyReLU()

    def call(self, x, training=True):
        xs = self.relu1(self.bn1(self.conv1(x), training=training))
        xs = self.bn2(self.conv2(xs), training=training)

        if self.residual_path:
            x = self.bn_shortcut(self.conv_shortcut(x), training=training)

        xs = x + xs
        return self.relu2(xs)

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({"out_ch": self.conv1.filters,
                       "residual_path": self.residual_path,
                       "stride": self.stride})
        return config
		
		