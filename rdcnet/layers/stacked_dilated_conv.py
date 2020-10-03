import itertools
import tensorflow as tf

from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import conv_utils


class StackedDilatedConv(tf.keras.layers.Layer):
    '''Applies the same filters with different dilation rates to an input,
    concatenates the outputs and reduce it back to "filters" number of channels.
    
    Directly supports group convolutions to take advantage of cudnn implementation.
    '''

    # NOTE with current workaround tf conv3D padding bug, "NCDHW" format is not supported

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 dilation_rates,
                 groups=1,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):

        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.groups = groups
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):

        input_channel = input_shape[-1]
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number'
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))

        kernel_size = conv_utils.normalize_tuple(self.kernel_size, self.rank,
                                                 'kernel_size')
        kernel_initializer = initializers.get(self.kernel_initializer)
        bias_initializer = initializers.get(self.bias_initializer)

        if self.rank == 2:
            self.tf_conv = tf.nn.conv2d
            self.strides = [1, 1, 1, 1]
        elif self.rank == 3:
            self.tf_conv = tf.nn.conv3d
            self.strides = [1, 1, 1, 1, 1]
        else:
            raise ValueError('rank {} not supported, expected 2 or 3'.format(
                self.rank))

        # NOTE: awckward tf interface to cudnn group conv: https://github.com/tensorflow/tensorflow/pull/25818
        # for 2D "the depth of inputs is not necessarily equal to filters.shape[2], but be a multiple of filters.shape[2]"
        kernel_shape = kernel_size + (input_channel // self.groups,
                                      self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=kernel_initializer,
                                      trainable=True,
                                      dtype=self.dtype)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters, ),
                                    initializer=bias_initializer,
                                    trainable=True,
                                    dtype=self.dtype)

        n_dilations = len(self.dilation_rates)
        reduction_kernel_shape = tuple(1 for _ in range(self.rank)) + (
            self.filters * n_dilations // self.groups, self.filters)
        self.reduction_kernel = self.add_weight(
            name='reduction_kernel',
            shape=reduction_kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
        self.reduction_bias = self.add_weight(
            name='reduction_bias',
            shape=(self.filters, ),
            initializer=self.bias_initializer,
            trainable=True,
            dtype=self.dtype)

    def _concat_interleaved_groups(self, dilated_outs):
        dilated_outs = [
            tf.split(out, self.groups, axis=-1) for out in dilated_outs
        ]

        # invert dilation and group axes + flatten
        dilated_outs = list(itertools.chain(*zip(*dilated_outs)))
        return tf.concat(dilated_outs, axis=-1)

    def _pad_input(self, inputs, spatial_dilations):

        spatial_kernel_size = self.kernel.shape[:self.rank]

        block_size = [
            k + (k - 1) * (d - 1)
            for d, k in zip(spatial_dilations, spatial_kernel_size)
        ]
        paddings = [[0, 0]] + [[(bs - 1) // 2, (bs - 1) // 2 + (bs - 1) % 2]
                               for bs in block_size] + [[0, 0]]
        return tf.pad(inputs, paddings, mode='CONSTANT')

    def call(self, inputs, **kwargs):
        dilated_outs = []
        for dilation in self.dilation_rates:
            spatial_dilations = conv_utils.normalize_tuple(
                dilation, self.rank, 'dilation_rate')

            # TODO report/check if fix
            # in tf 2.0, 2.1 tf.nn.Conv3D fails with dilation>1 and padding='SAME'
            # tensorflow.python.framework.errors_impl.NotFoundError: No algorithm worked! [Op:Conv3D]
            padded_inputs = self._pad_input(inputs, spatial_dilations)
            dilation = (1, ) + spatial_dilations + (1, )

            x = self.tf_conv(padded_inputs,
                             self.kernel,
                             padding='VALID',
                             strides=self.strides,
                             dilations=dilation)
            x = x + self.bias
            dilated_outs.append(x)

        out = self._concat_interleaved_groups(dilated_outs)
        out = activations.get(self.activation)(out)
        out = self.tf_conv(out,
                           self.reduction_kernel,
                           padding='VALID',
                           strides=self.strides)
        out = out + self.reduction_bias

        return out

    def get_config(self):
        config = super().get_config()

        config['rank'] = self.rank
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['dilation_rates'] = list(self.dilation_rates)
        config['groups'] = self.groups
        config['activation'] = self.activation
        config['kernel_initializer'] = self.kernel_initializer
        config['bias_initializer'] = self.bias_initializer
        return config
