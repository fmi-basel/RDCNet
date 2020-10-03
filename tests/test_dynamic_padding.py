import tensorflow as tf
import pytest

from rdcnet.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer


@pytest.mark.parametrize(
    'input_shape, factor, data_format, expected_output_shape', [
        ((4, 10, 12, 3), 16, None, (4, 16, 16, 3)),
        ((4, 16, 12, 3), 16, None, (4, 16, 16, 3)),
        ((4, 16, 12, 3), 16, 'channels_last', (4, 16, 16, 3)),
        ((4, 3, 12, 10), 16, 'channels_first', (4, 3, 16, 16)),
        ((1, 31, 32, 33, 1), 16, None, (1, 32, 32, 48, 1)),
        ((1, 17, 17, 17, 1), (4, 8, 16), None, (1, 20, 24, 32, 1)),
        ((1, 1, 17, 17, 17), (4, 8, 16), 'channels_first', (1, 1, 20, 24, 32)),
    ])
def test_dynamic_padding(input_shape, factor, data_format,
                         expected_output_shape):
    '''test dynamic padding layer
    '''

    layer = DynamicPaddingLayer(factor,
                                ndim=len(input_shape),
                                data_format=data_format)

    input_tensor = tf.random.normal(input_shape)
    output_tensor = layer(input_tensor)

    assert layer.compute_output_shape(input_shape) == expected_output_shape
    assert output_tensor.shape == expected_output_shape


@pytest.mark.parametrize('original_shape, data_format, input_shape', [
    ((4, 10, 12, 3), None, (4, 16, 16, 3)),
    ((4, 16, 12, 3), None, (4, 16, 16, 3)),
    ((4, 16, 12, 3), 'channels_last', (4, 16, 16, 3)),
    ((4, 3, 12, 10), 'channels_first', (4, 3, 16, 16)),
    ((1, 31, 32, 33, 1), None, (1, 32, 32, 48, 1)),
    ((1, 17, 17, 17, 1), None, (1, 20, 24, 32, 1)),
    ((1, 1, 17, 17, 17), 'channels_first', (1, 1, 20, 24, 32)),
])
def test_dynamic_trimming(original_shape, data_format, input_shape):
    '''test dynamic padding layer
    '''

    layer = DynamicTrimmingLayer(ndim=len(input_shape),
                                 data_format=data_format)

    input_tensor = tf.random.normal(input_shape)
    orignal_tensor = tf.random.normal(original_shape)
    output_tensor = layer([orignal_tensor, input_tensor])

    assert output_tensor.shape == original_shape


def build_custom_model(shape, data_format):
    '''keras model wrapped with two custom layers from rdcnet.

    '''
    input = tf.keras.layers.Input(shape)
    x = input
    x = DynamicPaddingLayer(factor=2, data_format=data_format)(input)
    x = tf.keras.layers.Conv2D(32,
                               kernel_size=3,
                               activation='relu',
                               padding='same',
                               data_format=data_format)(x)
    x = DynamicTrimmingLayer(data_format=data_format)([input, x])
    x = tf.keras.layers.Conv2D(32,
                               kernel_size=3,
                               activation='relu',
                               padding='same',
                               data_format=data_format)(x)

    return tf.keras.models.Model(inputs=input, outputs=[x])


@pytest.mark.parametrize(
    'tensor_shape, input_shape, data_format, expected_output_shape', [
        ((4, 63, 52, 3), (None, None, 3), 'channels_last', (4, 63, 52, 32)),
    ])
# Note: conv with channels_first not supported on cpu
# ((4, 3, 63, 52), (3, None, None), 'channels_first', (4, 32, 63, 52))])
def test_dynamic_trimming_output_shape(tensor_shape, input_shape, data_format,
                                       expected_output_shape):
    '''Test that a trimming layer can be used in a model with
    additional layer afterwards'''

    model = build_custom_model(input_shape, data_format)
    input_tensor = tf.random.normal(tensor_shape)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == expected_output_shape
