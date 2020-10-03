import itertools

import numpy as np
import tensorflow as tf
import pytest

from rdcnet.layers.stacked_dilated_conv import StackedDilatedConv
from tensorflow.keras.layers import LeakyReLU


# yapf: disable
@pytest.mark.parametrize('input_shape,rank,filters,kernel_size,dilation_rates,groups,activation',[
    ((8, 32, 32, 16), 2, 8, 3, (1,), 1, None),
    ((3, 32, 32, 16), 2, 32, 5, (1,2,4), 8, None),

    ((8, 11, 32, 23, 16), 3, 8, 3, (1,), 1, LeakyReLU()),
    ((7, 11, 32, 23, 16), 3, 16, 5, (2,), 8, LeakyReLU())])
# yapf: enable
def test_stacked_dilated_conv(input_shape, rank, filters, kernel_size,
                              dilation_rates, groups, activation):
    '''Tests that stacked_dilated_conv can be instanciated, called and outputs the correct shape'''

    input = tf.random.normal(shape=input_shape, dtype=tf.float32)

    sd_conv = StackedDilatedConv(rank=rank,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 dilation_rates=dilation_rates,
                                 groups=groups,
                                 activation=activation)

    output = sd_conv(input)

    assert output.shape == input.shape[:-1] + (filters, )


def test__concat_interleaved_groups():

    # group id: 1, 2, 3
    # dilation rate id: 10, 20

    o = np.ones((3, 4, 1), dtype=np.int32)
    dilated_outs = [
        np.stack([o + 10, 2 * o + 10, 3 * o + 10], axis=-1),
        np.stack([o + 20, 2 * o + 20, 3 * o + 20], axis=-1)
    ]

    sd_conv = StackedDilatedConv(rank=2,
                                 filters=4,
                                 kernel_size=3,
                                 dilation_rates=(1, 2, 4),
                                 groups=3)

    first_px_res = sd_conv._concat_interleaved_groups(dilated_outs)[
        0, 0].numpy().squeeze()
    assert all(first_px_res == [11, 21, 12, 22, 13, 23])


def test_raise_group_size():
    '''test that filters/groups raise.

    '''

    with pytest.raises(ValueError):
        input = tf.random.normal(shape=(8, 32, 32, 16), dtype=tf.float32)
        sd_conv = StackedDilatedConv(rank=2,
                                     filters=12,
                                     kernel_size=3,
                                     dilation_rates=(1, ),
                                     groups=7,
                                     activation=None)
        output = sd_conv(input)


@pytest.mark.parametrize('rank', [2, 3])
def test_save_load(tmpdir, rank):
    '''test saving/loading with default as h5.

    '''
    model = tf.keras.models.Sequential([
        StackedDilatedConv(rank=rank,
                           filters=12,
                           kernel_size=3,
                           dilation_rates=(1, 2, 4, 8, 16),
                           input_shape=(None, ) * rank + (16, ))
    ])

    # dump model with layer.
    output_path = tmpdir / 'model.h5'
    assert not output_path.exists()
    tf.keras.models.save_model(model, str(output_path))
    assert output_path.exists()

    # reload it.
    loaded_model = tf.keras.models.load_model(
        output_path,
        custom_objects={StackedDilatedConv.__name__: StackedDilatedConv})

    # compare loaded and original model.
    def assert_identical_models(left, right):
        '''utility to compare two models.
        '''
        assert left.get_config() == right.get_config()
        for left_w, right_w in zip(left.get_weights(), right.get_weights()):
            np.testing.assert_allclose(left_w, right_w)

    assert_identical_models(model, loaded_model)
