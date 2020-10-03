'''provides additive semi-convolutional layers.

'''
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv

from numpy import broadcast_to


def generate_coordinate_grid(shape, ndim):
    '''generates a coordinate meshgrid.

    NOTE that this expects channels_last.

    Parameters
    ----------
    shape : shape tensor or tuple.
        shape of the tensor for which the grid is to be constructed.
    ndim : int
        dimensionality of the coordinate grid.

    Returns
    -------
    coordinate_grid : tensor
        coordinate grid for the spatial dimensions of shape.

    '''
    ranges = tuple(
        tf.cast(tf.range(shape[ii]), tf.float32) for ii in range(1, 1 + ndim))
    coords = tf.stack(tf.meshgrid(*ranges, indexing='ij'),
                      axis=-1,
                      name='coords')
    return coords


class AdditiveSemiConv(Conv):
    '''Additive semiconvolutional layer.

    Implements the following operation in one layer:

        output = Conv(input) + coordinate_grid * spacing

    '''
    def __init__(self, *args, spacing=1., **kwargs):
        '''
        '''
        if 'filters' not in kwargs:
            kwargs['filters'] = kwargs['rank']
        super().__init__(*args, **kwargs)

        if self.rank != self.filters:
            raise ValueError(
                'The AdditiveSemiConv layer expects the number of filters to be equal to the rank. '
                'Got rank={} and filters={} '.format(kwargs['rank'],
                                                     kwargs['filters']))

        self.spacing = tuple(
            float(val) for val in broadcast_to(spacing, self.rank))

    def call(self, inputs):
        '''
        '''
        delta = super().call(inputs)
        return generate_coordinate_grid(tf.shape(delta),
                                        self.rank) * self.spacing + delta

    def get_config(self):
        '''
        '''
        config = super().get_config()
        config['spacing'] = self.spacing
        return config


class AdditiveSemiConv2D(AdditiveSemiConv):
    '''Additive semiconvolutional layer in 2D.

    NOTE Number of filters is fixed to 2.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        spacing: float or tuple of length 2.
            Scaling factor for (each dimension of) the coordinate grid.

        Other parameters: See tensorflow.keras.layers.Conv2D.
        '''
        super().__init__(*args, rank=2, **kwargs)


class AdditiveSemiConv3D(AdditiveSemiConv):
    '''Additive semiconvolutional layer in 3D.

    NOTE Number of filters is fixed to 3.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        spacing: float or tuple of length 3.
            Scaling factor for (each dimension of) the coordinate grid.

        Other parameters: See tensorflow.keras.layers.Conv3D.
        '''
        super().__init__(*args, rank=3, **kwargs)
