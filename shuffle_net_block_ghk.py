import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D

# ShuffleNet block definition
class ShuffleNetBlock(tf.keras.Model):
    def __init__(self, groups, in_channels, out_channels, stride):
        super(ShuffleNetBlock, self).__init__()

        self.stride = stride
        self.groups = groups
        mid_channels = out_channels // 4

        # Adjust the number of channels for the shortcut connection
        if stride == 1:
            in_channels_split = in_channels // self.groups
            self.mid_channels_split = mid_channels // self.groups
            self.out_channels_split = out_channels // self.groups
        else:
            in_channels_split = in_channels // 2
            self.mid_channels_split = mid_channels // 2
            self.out_channels_split = out_channels // 2

        # Define the layers
        self.conv1 = Conv2D(self.mid_channels_split, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.dwconv2 = DepthwiseConv2D((3, 3), strides=(stride, stride), padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(self.out_channels_split, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu2 = Activation('relu')

    def call(self, x):
        # Split the channels into groups
        x_split = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        out_split = []

        for i in range(self.groups):
            x_group = x_split[i]

            out1 = self.relu1(self.bn1(self.conv1(x_group)))
            out2 = self.dwconv2(x_group)
            out = tf.concat([out1, out2], axis=-1)
            out_split.append(out)

        out = tf.concat(out_split, axis=-1)
        out = self.relu2(self.bn3(self.conv3(out)))
        return out

# Create a random input tensor with shape (batch_size, height, width, channels)
batch_size = 1
height, width, channels = 48, 48, 3
input_tensor = tf.random.normal((batch_size, height, width, channels))

# Create a ShuffleNet block
shufflenet_block = ShuffleNetBlock(groups=3, in_channels=channels, out_channels=128, stride=1)

# Pass the input through the ShuffleNet block
output = shufflenet_block(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
