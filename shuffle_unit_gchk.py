!pip install keras.engine.topology
from keras.utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape
#from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense
from keras.models import Model
import keras.backend as K
import os
from keras import backend as K
from keras.models import Model
#from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D


from IPython.display import clear_output
clear_output()

def channel_split(x, groups):
    height, width, in_channels = x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = in_channels // groups

    # Split the input tensor into a list of tensors, one for each group
    split_tensors = []
    for i in range(groups):
        start_channel = i * channels_per_group
        end_channel = start_channel + channels_per_group
        split = x[:, :, :, start_channel:end_channel]
        split_tensors.append(split)

    return split_tensors

class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        self.groups = groups
        
        if stride != 1 or in_channels != out_channels:
            self.residual_path = tf.keras.Sequential([
                Conv2D(out_channels // 4, kernel_size=1, strides=stride, padding='same', use_bias=False),
                BatchNormalization(),
            ])
        else:
            self.residual_path = None
        
        mid_channels = out_channels // 4
        self.shuffle_path = tf.keras.Sequential([
            Conv2D(mid_channels, kernel_size=1, padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False, groups=mid_channels),
            BatchNormalization(),
            Conv2D(out_channels - in_channels, kernel_size=1, padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
        ])
    
    def call(self, x):
        if self.residual_path is not None:
            residual = self.residual_path(x)
        else:
            residual = x
        
        split_tensors = tf.split(residual, self.groups, axis=-1)
        shuffled_tensor = channel_shuffle(split_tensors, self.groups)
        
        x = self.shuffle_path(x)
        
        output = tf.concat([shuffled_tensor, x], axis=-1)
        return output

'''
input_shape = (64, 64, 128)  # Example input shape

# Create a random input tensor with the specified shape
input_tensor = tf.random.normal((1,) + input_shape)  # Batch size of 1

# Create a ShuffleUnit instance (adjust parameters as needed)
in_channels = 128
out_channels = 256
stride = 2
groups = 4
shuffle_unit = ShuffleUnit(in_channels, out_channels, stride, groups)

# Pass the input tensor through the ShuffleUnit
output = shuffle_unit(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
'''


'''
# Example usage:
input_tensor = tf.random.normal((1, 48, 48, 96))  # Example input with 96 channels
split_tensors = channel_split(input_tensor, groups=4)  # Split channels into 4 groups
for i in split_tensors:
  print(i.shape)
'''

def channel_shuffle(split_tensors, groups):
    # Transpose the list of split tensors to group the channels together
    shuffled_tensors = []
    num_splits = len(split_tensors)
    channels_per_group = split_tensors[0].shape[-1]
    
    for group_idx in range(groups):
        for split_idx in range(num_splits):
            start_channel = group_idx * channels_per_group
            end_channel = start_channel + channels_per_group
            shuffled_tensors.append(split_tensors[split_idx][:, :, :, start_channel:end_channel])
    
    # Concatenate the shuffled tensors along the channel axis
    shuffled_tensor = tf.concat(shuffled_tensors, axis=-1)
    
    return shuffled_tensor

'''
groups=4
shuffled_tensor = channel_shuffle(split_tensors, groups)
shuffled_tensor.shape
'''