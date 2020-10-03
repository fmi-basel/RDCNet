import itertools

import numpy as np
import tensorflow as tf
import pytest

from rdcnet.layers.semi_conv import AdditiveSemiConv2D, AdditiveSemiConv3D
from rdcnet.layers.nd_layers import get_nd_semiconv


def test_get_nd_semiconv():
    '''test get_nd wrapper'''

    assert get_nd_semiconv(2) == AdditiveSemiConv2D
    assert get_nd_semiconv(3) == AdditiveSemiConv3D


@pytest.mark.parametrize('batch_size,spacing,kernel_size',
                         itertools.product([1, 2, 4], [
                             1.25,
                             (0.5, 0.3),
                             (1, 2),
                         ], [1, 3, 5]))
def test_additive_semiconv2d(batch_size, spacing, kernel_size):
    '''test additive scaling layer in 2D.

    '''
    ndim = 2
    model = tf.keras.models.Sequential([
        AdditiveSemiConv2D(spacing=spacing,
                           kernel_size=kernel_size,
                           input_shape=(None, None, 1),
                           padding='same')
    ])

    model.compile(loss='mae')

    vals = np.zeros((batch_size, 11, 13, 1))
    expected_maxima = (np.asarray(vals.shape[1:-1]) - 1) * np.broadcast_to(
        spacing, (ndim, ))

    output = model.predict(vals)

    assert vals.shape[:-1] == output.shape[:-1]
    assert output.shape[-1] == ndim

    # check all the corners of the grid.
    assert np.allclose(output[:, 0, 0, :], 0.)
    assert np.allclose(output[:, -1, 0, :], (expected_maxima[0], 0.))
    assert np.allclose(output[:, 0, -1, :], (0, expected_maxima[1]))
    assert np.allclose(output[:, -1, -1, :], expected_maxima)

    # generate a non-empty input and make sure input is not identical.
    vals = np.random.randn(*vals.shape)
    output_randn = model.predict(vals)
    assert not np.all(output == output_randn)


@pytest.mark.parametrize(
    'batch_size,spacing',
    itertools.product([1, 2, 4], [1.25, (0.5, 0.3, 1.0), (1., 2, 0.23)]))
def test_additive_semiconv3d(batch_size, spacing):
    '''test additive scaling layer in 3D.

    '''
    ndim = 3
    model = tf.keras.models.Sequential([
        AdditiveSemiConv3D(spacing=spacing,
                           kernel_size=3,
                           input_shape=(None, None, None, 1),
                           padding='same')
    ])

    model.compile(loss='mae')

    vals = np.zeros((batch_size, 20, 15, 5, 1))
    expected_maxima = (np.asarray(vals.shape[1:-1]) - 1) * np.broadcast_to(
        spacing, (ndim, ))

    output = model.predict(vals)

    assert vals.shape[:-1] == output.shape[:-1]
    assert output.shape[-1] == ndim

    # check all the corners of the grid.
    assert np.allclose(output[:, 0, 0, 0, :], 0.)
    assert np.allclose(output[:, -1, 0, 0, :], (expected_maxima[0], 0., 0))
    assert np.allclose(output[:, 0, -1, 0, :], (0, expected_maxima[1], 0))
    assert np.allclose(output[:, 0, 0, -1, :], (0, 0, expected_maxima[2]))
    assert np.allclose(output[:, -1, -1, 0, :],
                       (expected_maxima[0], expected_maxima[1], 0))
    assert np.allclose(output[:, -1, 0, -1, :],
                       (expected_maxima[0], 0, expected_maxima[2]))
    assert np.allclose(output[:, 0, -1, -1, :],
                       (0, expected_maxima[1], expected_maxima[2]))
    assert np.allclose(output[:, -1, -1, -1, :], expected_maxima)

    # generate a non-empty input and make sure input is not identical.
    vals = np.random.randn(*vals.shape)
    output_randn = model.predict(vals)
    assert not np.all(output == output_randn)


@pytest.mark.parametrize('spacing', [(2, 1.0, .7), (1, 2, 3, 4)])
def test_raise_spacing_2d(spacing):
    '''test that incompatible spacings raise.

    '''
    with pytest.raises(ValueError):
        AdditiveSemiConv2D(spacing=spacing,
                           kernel_size=3,
                           input_shape=(None, None, 1),
                           padding='same')


@pytest.mark.parametrize('spacing', [(2, 1.0), (1, 2, 3, 4)])
def test_raise_spacing_3d(spacing):
    '''test that incompatible spacings raise.

    '''
    with pytest.raises(ValueError):
        AdditiveSemiConv3D(spacing=spacing,
                           kernel_size=3,
                           input_shape=(None, None, None, 1),
                           padding='same')


@pytest.mark.parametrize('spacing', [1, (1.2, 1.2, 3.), np.asarray((1.))])
def test_save_load(tmpdir, spacing):
    '''test saving/loading with default as h5.

    '''
    model = tf.keras.models.Sequential([
        AdditiveSemiConv3D(spacing=spacing,
                           kernel_size=3,
                           input_shape=(None, None, None, 1),
                           padding='same')
    ])

    # dump model with layer.
    output_path = tmpdir / 'model.h5'
    assert not output_path.exists()
    tf.keras.models.save_model(model, str(output_path))
    assert output_path.exists()

    # reload it.
    loaded_model = tf.keras.models.load_model(
        output_path,
        custom_objects={AdditiveSemiConv3D.__name__: AdditiveSemiConv3D})
    loaded_model.summary()

    # compare loaded and original model.
    def assert_identical_models(left, right):
        '''utility to compare two models.
        '''
        assert left.get_config() == right.get_config()
        for left_w, right_w in zip(left.get_weights(), right.get_weights()):
            np.testing.assert_allclose(left_w, right_w)

    assert_identical_models(model, loaded_model)
