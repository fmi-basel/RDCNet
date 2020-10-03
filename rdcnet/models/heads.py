import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, Conv3D, Lambda
from tensorflow.keras.models import Model

from rdcnet.layers.semi_conv import generate_coordinate_grid


def add_fcn_output_layers(model,
                          names,
                          n_classes,
                          activation='sigmoid',
                          kernel_size=1):
    '''attaches fully-convolutional output layers to the
    last layer of the given model.

    '''
    last_layer = model.layers[-1].output
    if len(last_layer.shape) == 5:
        Conv = Conv3D
    else:
        Conv = Conv2D

    if isinstance(names, list) and isinstance(n_classes, list):
        assert len(names) == len(n_classes)
    if not isinstance(activation, list):
        activation = len(names) * [
            activation,
        ]
    # TODO handle other cases

    outputs = []
    for name, classes, act in zip(names, n_classes, activation):
        outputs.append(
            Conv(classes,
                 kernel_size=kernel_size,
                 name=name,
                 activation=act,
                 padding='same')(last_layer))
    model = Model(model.inputs, outputs, name=model.name)
    return model


def add_instance_seg_heads(model, n_classes, spacing=1.):
    '''Splits the output of model into instance semi-conv embeddings and semantic class.
    
    Args:
        model: delta_loop base model, should output at least 
            n_classes + n_spatial-dimensions channels
        n_classes: number semantic classes
    '''

    spatial_dims = len(model.inputs[0].shape) - 2
    spacing = tuple(
        float(val) for val in np.broadcast_to(spacing, spatial_dims))
    y_preds = model.outputs[0]

    if y_preds.shape[-1] < n_classes + spatial_dims:
        raise ValueError(
            'model has less than n_classes + n_spatial_dims channels: {} < {} + {}'
            .format(y_preds.shape[-1], n_classes, spatial_dims))

    vfield = y_preds[..., 0:spatial_dims]
    coords = generate_coordinate_grid(tf.shape(vfield), spatial_dims) * spacing
    embeddings = coords + vfield

    semantic_class = y_preds[..., spatial_dims:spatial_dims + n_classes]
    semantic_class = tf.nn.softmax(semantic_class, axis=-1)

    # rename outputs
    embeddings = Lambda(lambda x: x, name='embeddings')(embeddings)
    semantic_class = Lambda(lambda x: x, name='semantic_class')(semantic_class)

    return Model(inputs=model.inputs,
                 outputs=[embeddings, semantic_class],
                 name=model.name)
