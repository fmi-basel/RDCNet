import tensorflow as tf
import pytest
import numpy as np
import itertools

from rdcnet.models.rdcnet import GenericRDCnetBase, delta_loop
from rdcnet.models.heads import add_instance_seg_heads
from rdcnet.layers.padding import DynamicPaddingLayer, DynamicTrimmingLayer
from rdcnet.layers.stacked_dilated_conv import StackedDilatedConv

CUSTOM_LAYERS = {
    cls.__name__: cls
    for cls in [DynamicPaddingLayer, DynamicTrimmingLayer, StackedDilatedConv]
}


def test_delta_loop():
    '''checks delta loop output vs n_steps with simple increments "predicted"
    by the recurrent block"
    '''

    inputs = tf.zeros((4, 10, 10, 1), dtype=tf.float32)

    def recurrent_block(x):
        # increment internal state by 1,2 on ch0, ch1 respectively
        o = tf.ones((4, 10, 10), dtype=tf.float32)
        return tf.stack([o, o + 1], axis=-1)

    for steps in range(10):
        loop = delta_loop(output_channels=2,
                          recurrent_block=recurrent_block,
                          n_steps=steps)
        outputs = loop(inputs)
        assert np.all(outputs[..., 0] == steps)
        assert np.all(outputs[..., 1] == steps * 2)


@pytest.mark.parametrize('input_shape,\
                          downsampling_factor,\
                          n_output_channels,\
                          channels_per_group',
                         itertools.product([(16, 11, 3), (16, 15, 16, 1)],
                                           [1, 3], [2, 5], [8, 16]))
def test_GenericRDCnetBase(
    input_shape,
    downsampling_factor,
    n_output_channels,
    channels_per_group,
):
    '''Checks that 2D/3D models can be created and output the expected shape'''

    rank = len(input_shape) - 1
    partial_input_shape = (None, ) * rank + input_shape[-1:]

    model = GenericRDCnetBase(partial_input_shape,
                              downsampling_factor,
                              n_downsampling_channels=7,
                              n_output_channels=n_output_channels,
                              n_groups=4,
                              dilation_rates=(1, 2, 4),
                              channels_per_group=channels_per_group,
                              n_steps=5)

    batch_size = 2
    img = np.random.randn(batch_size, *input_shape)
    pred = model.predict(img)

    assert pred.shape == img.shape[:-1] + (n_output_channels, )


@pytest.mark.parametrize('input_shape,spacing',
                         [((16, 11, 3), 1.), ((16, 15, 16, 1), (1, 0.5, 0.5))])
def test_instance_segmentation_head(input_shape, spacing):

    n_classes = 5
    rank = len(input_shape) - 1
    n_output_channels = n_classes + rank
    partial_input_shape = (None, ) * rank + input_shape[-1:]

    model = GenericRDCnetBase(partial_input_shape,
                              downsampling_factor=2,
                              n_downsampling_channels=7,
                              n_output_channels=n_output_channels,
                              n_groups=4,
                              dilation_rates=(1, 2, 4),
                              channels_per_group=8,
                              n_steps=5)

    model = add_instance_seg_heads(model, n_classes, spacing)

    batch_size = 2
    img = np.random.randn(batch_size, *input_shape)
    embeddings, classes = model.predict(img)

    assert embeddings.shape == img.shape[:-1] + (rank, )
    assert classes.shape == img.shape[:-1] + (n_classes, )


def get_dummy_dataset(n_samples, batch_size, repeats=None):
    '''Creates a dummy tensorflow dataset with random noise as input
    and a mask where input>0 as target.'''
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def gen():
        for i in range(n_samples):
            yield tf.random.normal((17, 23, 1))

    return (tf.data.Dataset.from_generator(
                  gen,
                  (tf.float32),
                  output_shapes=(17, 23, 1)).map(lambda img: (img, tf.cast(tf.math.greater(img, 0.), tf.float32)),
                  num_parallel_calls=AUTOTUNE)\
                  .repeat(repeats).batch(batch_size))


@pytest.mark.slow
def test_training_GenericRDCnetBase(tmpdir):
    '''tests GenericRDCnetBase training and saving'''

    input_tensor = tf.random.normal((4, 30, 32, 1))
    target = tf.cast(tf.math.greater(input_tensor, 0.), tf.float32)

    model = GenericRDCnetBase((None, None, 1),
                              downsampling_factor=2,
                              n_downsampling_channels=7,
                              n_output_channels=1,
                              n_groups=4,
                              dilation_rates=(1, 2, 4),
                              channels_per_group=4,
                              n_steps=5)

    loss_fun = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fun,
    )

    outputs_init = model(input_tensor)
    loss_init = loss_fun(target, outputs_init)
    assert outputs_init.shape == (4, 30, 32, 1)

    # train model
    model.fit(
        get_dummy_dataset(16, batch_size=4),
        validation_data=get_dummy_dataset(4, batch_size=4),
        epochs=10,
        steps_per_epoch=4,
        validation_steps=1,
    )

    outputs_trained = model(input_tensor)
    loss_trained = loss_fun(target, outputs_trained)
    assert outputs_trained.shape == (4, 30, 32, 1)

    # save and reload model
    output_path = tmpdir / 'model.h5'
    tf.keras.models.save_model(model, str(output_path))
    loaded_model = tf.keras.models.load_model(output_path,
                                              custom_objects=CUSTOM_LAYERS)

    outputs_reloaded = loaded_model(input_tensor)
    loss_reloaded = loss_fun(target, outputs_reloaded)
    assert outputs_reloaded.shape == (4, 30, 32, 1)

    assert not np.allclose(
        outputs_init.numpy(), outputs_trained.numpy(), rtol=1e-5)
    assert np.allclose(outputs_trained.numpy(),
                       outputs_reloaded.numpy(),
                       rtol=1e-5)

    assert loss_trained.numpy() + 0.1 < loss_init.numpy()
    np.testing.assert_almost_equal(loss_trained.numpy(), loss_reloaded.numpy())
