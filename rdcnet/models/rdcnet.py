import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU

from rdcnet.layers.nd_layers import get_nd_conv, get_nd_spatial_dropout, get_nd_conv_transposed
from rdcnet.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from rdcnet.layers.stacked_dilated_conv import StackedDilatedConv


def delta_loop(output_channels, recurrent_block, n_steps=3):
    '''Recursively applies a given block to refine its output.
    
    Args:
        output_channels: number of output channels.
        recurrent_block: a network taking (input_channels + output_channels) as 
        input and outputting output_channels
        n_steps: number of times the block is applied
    '''
    def block(x, state=None):

        if state is None:
            recurrent_shape = tf.concat(
                [tf.shape(x)[:-1],
                 tf.constant([output_channels])], axis=0)
            state = tf.zeros(recurrent_shape, x.dtype)

        # static unrolling
        for _ in range(n_steps):  # static unrolled loop
            delta = recurrent_block(tf.concat([x, state], axis=-1))
            state = state + delta
        return state

    return block


def rdc_block(n_groups=16,
              dilation_rates=(1, 2, 4, 8, 16),
              channels_per_group=32,
              k_size=3,
              spatial_dims=2,
              dropout=0.1):
    '''Grouped conv with stacked dilated conv in each group and pointwise convolution for mixing
    
    Notes
    -----
    pre-activation to keep the residual path clear as described in:
    
    HE, Kaiming, et al. Identity mappings in deep residual networks.
    In: European conference on computer vision. Springer, Cham, 2016.
    S. 630-645.
    '''

    Conv = get_nd_conv(spatial_dims)
    channels = channels_per_group * n_groups
    sd_conv = StackedDilatedConv(rank=spatial_dims,
                                 filters=channels,
                                 kernel_size=k_size,
                                 dilation_rates=dilation_rates,
                                 groups=n_groups,
                                 activation=LeakyReLU())

    # mixes ch/reduce from input_ch + channels_per_group*n_groups
    reduce_ch_conv = Conv(channels, 1)

    spatial_dropout = get_nd_spatial_dropout(spatial_dims)(dropout)

    def _call(x):

        x = spatial_dropout(x)
        x = LeakyReLU()(x)
        x = reduce_ch_conv(x)
        x = LeakyReLU()(x)
        x = sd_conv(x)

        return x

    return _call


def _format_tuple(val):
    unique_val = tuple(set(val))

    if len(unique_val) == 1:
        return str(unique_val[0])
    else:
        return str(val).replace(', ', '-').replace('(', '').replace(')', '')


def GenericRDCnetBase(input_shape,
                      downsampling_factor,
                      n_downsampling_channels,
                      n_output_channels,
                      n_groups=16,
                      dilation_rates=(1, 2, 4, 8, 16),
                      channels_per_group=32,
                      n_steps=5,
                      dropout=0.1):
    '''delta loop with input/output rescaling and atrous grouped conv recurrent block'''

    spatial_dims = len(input_shape) - 1
    downsampling_factor = tuple(
        np.broadcast_to(np.array(downsampling_factor), spatial_dims).tolist())

    recurrent_block = rdc_block(n_groups,
                                dilation_rates,
                                channels_per_group,
                                spatial_dims=spatial_dims,
                                dropout=dropout)
    n_features = channels_per_group * n_groups
    loop = delta_loop(n_features, recurrent_block, n_steps)

    in_kernel_size = tuple(max(3, f) for f in downsampling_factor)
    out_kernel_size = tuple(max(3, 2 * f) for f in downsampling_factor)

    Conv = get_nd_conv(spatial_dims)
    conv_in = Conv(n_downsampling_channels,
                   kernel_size=in_kernel_size,
                   strides=downsampling_factor,
                   padding='same')

    ConvTranspose = get_nd_conv_transposed(spatial_dims)
    conv_out = ConvTranspose(n_output_channels,
                             kernel_size=out_kernel_size,
                             strides=downsampling_factor,
                             padding='same')

    input_padding = DynamicPaddingLayer(downsampling_factor,
                                        ndim=spatial_dims + 2)
    output_trimming = DynamicTrimmingLayer(ndim=spatial_dims + 2)

    inputs = Input(shape=input_shape)
    x = input_padding(inputs)
    x = conv_in(x)

    x = loop(x)
    x = LeakyReLU()(x)
    x = conv_out(x)
    x = output_trimming([inputs, x])

    name = 'RDCNet-F{}-DC{}-OC{}-G{}-DR{}-GC{}-S{}-D{}'.format(
        _format_tuple(downsampling_factor),
        n_downsampling_channels, n_output_channels, n_groups,
        _format_tuple(dilation_rates), channels_per_group, n_steps, dropout)

    return Model(inputs=inputs, outputs=[x], name=name)
