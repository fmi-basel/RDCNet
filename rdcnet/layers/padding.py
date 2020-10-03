from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer

import tensorflow as tf
import numpy as np


class DynamicPaddingLayer(Layer):
    '''Adds padding to input tensor such that its spatial dimensions
    are divisible by a given factor.

    '''
    def __init__(self, factor, ndim=4, data_format=None, **kwargs):
        '''
        '''
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        self.ndim = ndim
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=self.ndim)]
        self.factor = tuple(
            np.broadcast_to(np.array(factor), ndim - 2).tolist())
        super(DynamicPaddingLayer, self).__init__(**kwargs)

    def get_padded_dim(self, size, dim_factor):
        '''
        '''
        if size is None:
            return size
        if size % dim_factor == 0:
            return size
        return size + dim_factor - size % dim_factor

    def get_paddings(self, size, dim_factor):
        '''
        '''
        # last % operation takes care of the case where size is already divisible by factor
        dx = (dim_factor - size % dim_factor) % dim_factor
        return [dx // 2, dx - dx // 2]

    def compute_output_shape(self, input_shape):
        '''
        '''
        ndim = len(input_shape)
        if self.data_format == 'channels_last':
            return (input_shape[0], ) + tuple(
                self.get_padded_dim(input_shape[dim], self.factor[dim - 1])
                for dim in range(1, ndim - 1)) + (input_shape[-1], )

        return (input_shape[0], input_shape[1]) + tuple(
            self.get_padded_dim(input_shape[dim], self.factor[dim - 2])
            for dim in range(2, ndim))

    def call(self, inputs):
        '''
        '''
        input_shape = tf.shape(inputs)
        ndim = K.ndim(inputs)
        if self.data_format == 'channels_last':
            paddings = [[0, 0]] + [
                self.get_paddings(input_shape[dim], self.factor[dim - 1])
                for dim in range(1, ndim - 1)
            ] + [[0, 0]]
        else:
            paddings = [[0, 0], [0, 0]] + [
                self.get_paddings(input_shape[dim], self.factor[dim - 2])
                for dim in range(2, ndim)
            ]

        return tf.pad(inputs, paddings, 'CONSTANT')

    def get_config(self):
        config = super().get_config()
        config['factor'] = tuple(self.factor)
        config['data_format'] = self.data_format
        config['ndim'] = self.ndim

        return config


class DynamicTrimmingLayer(Layer):
    '''Trims a given tensor to have the same spatial dimensions
    as another.

    '''
    def __init__(self, ndim=4, data_format=None, **kwargs):
        '''
        '''

        self.ndim = ndim
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}
        self.data_format = data_format

        self.input_spec = [
            InputSpec(ndim=self.ndim),
            InputSpec(ndim=self.ndim)
        ]
        super(DynamicTrimmingLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['data_format'] = self.data_format
        config['ndim'] = self.ndim

        return config

    def compute_output_shape(self, input_shape):
        '''
        '''
        # TODO is this ever called?!
        if self.data_format == 'channels_last':
            return input_shape[0][:-1] + input_shape[1][-1:]
        return input_shape[1][:2] + input_shape[0][2:]

    def call(self, inputs):
        '''expects a list of exactly 2 inputs:

        inputs[0] = original_tensor -> for shape
        inputs[1] = output tensor that should be trimmed

        '''
        assert len(inputs) == 2
        output_tensor = inputs[1]
        output_shape = tf.shape(output_tensor)
        original_shape = tf.shape(inputs[0])

        dx = [(x - y) // 2
              for x, y in ((output_shape[idx], original_shape[idx])
                           for idx in range(self.ndim))]

        if self.data_format == 'channels_last':
            starts = [0] + dx[1:-1] + [0]
            ends = [-1] + [
                original_shape[idx] for idx in range(1, self.ndim - 1)
            ] + [-1]
        else:
            starts = [0, 0] + dx[2:]
            ends = [-1, -1
                    ] + [original_shape[idx] for idx in range(2, self.ndim)]

        trimmed_tensor = tf.slice(output_tensor, starts, ends)

        # manually set the shape that has been "lost" by tf.slice
        trimmed_tensor.set_shape(
            self.compute_output_shape([input.shape for input in inputs]))

        return trimmed_tensor
