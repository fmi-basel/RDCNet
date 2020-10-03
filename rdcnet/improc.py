import tensorflow as tf
import numpy as np


def gaussian_filter(sigma, spatial_rank, truncate=4):
    '''
    Returns a gaussian filter for images (without batch dim)
    
    Args:
        sigma: An float or tuple/list of 2 floats indicating the standard deviation along each axis
        spatial_rank: number of spatial dimensions
        truncate: Truncate the filter at this many standard deviations
        
    Returns:
        callable taking a tensor to fitler as input
    '''

    sigma = np.broadcast_to(np.asarray(sigma), spatial_rank)

    def _gaussian_kernel(n_channels, dtype):
        half_size = np.round(truncate * sigma).astype(int)
        x_1d = [tf.range(-hs, hs + 1, dtype=dtype) for hs in half_size]
        g_1d = [
            tf.math.exp(-0.5 * tf.pow(x / tf.cast(s, dtype), 2))
            for x, s in zip(x_1d, sigma)
        ]

        g_kernel = g_1d[0]
        for g in g_1d[1:]:
            g_kernel = tf.tensordot(g_kernel, g, axes=0)

        g_kernel = g_kernel / tf.reduce_sum(g_kernel)
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel,
                                      (1, ) * spatial_rank + (n_channels, )),
                              axis=-1)

    def _filter(x):

        if len(x.shape) != spatial_rank + 1:
            raise ValueError(
                'Wrong input shape, expected {} spatial dimensions + channel, got {}'
                .format(spatial_rank, len(x.shape)))

        kernel = _gaussian_kernel(tf.shape(x)[-1], x.dtype)
        if spatial_rank == 2:
            return tf.nn.depthwise_conv2d(x[None], kernel,
                                          (1, ) * (spatial_rank + 2),
                                          'SAME')[0]
        elif spatial_rank == 3:
            if x.shape[-1] == 1:
                return tf.nn.conv3d(x[None], kernel,
                                    (1, ) * (spatial_rank + 2), 'SAME')[0]
            else:
                raise NotImplementedError(
                    '3D gaussian filter for more than one channel is not implemented, input shape: {}'
                    .format(x.shape))

    return _filter


def local_max(image, min_distance=1, threshold=1, spacing=1):
    '''Finds local maxima that are above threshold
    
    Args:
        image: greyscale image
        min_distance: scalar defining the min distance between local max
        threshold: absolute intensity threshold to consider a local max
        spacing: pixel/voxel size
    '''

    # implements x==max_pool(x) with pre-blurring to avoid
    # spurious max when neighbors have the same values

    if min_distance < 1:
        raise ValueError('min_distance should be > 1: {}'.format(min_distance))

    rank = len(image.shape)
    spacing = np.broadcast_to(np.asarray(spacing), rank)

    gaussian = gaussian_filter(sigma=np.sqrt(min_distance / spacing),
                               spatial_rank=rank)
    image = tf.cast(image, tf.float32)

    # NOTE needs explicit channel dimension
    blurred_image = gaussian(image[..., None])[..., 0]

    max_filt_size = np.maximum(1, min_distance / spacing * 2 + 1)
    max_image = tf.nn.max_pool(
        blurred_image[None, ..., None],
        ksize=max_filt_size,
        strides=1,
        padding='SAME',
    )[0, ..., 0]

    return tf.where((max_image <= blurred_image) & (image >= threshold))
