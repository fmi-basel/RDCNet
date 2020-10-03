import tensorflow as tf

from rdcnet.losses.utils import hingify


class JaccardLoss(tf.keras.losses.Loss):
    '''
    Differentiable Jaccard/mIoU loss as proposed in:
    
    Rahman, Md Atiqur, and Yang Wang. "Optimizing intersection-over-union 
    in deep neural networks for image segmentation." 
    International symposium on visual computing. Springer, Cham, 2016.
    
    Args:
    eps: epsilon to avoid divison by zero.
    '''
    def __init__(self, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps

    def _remove_unannot(self, y_true, y_pred):
        annot_mask = tf.cast(
            tf.reduce_any(tf.greater_equal(y_true, 0.5),
                          axis=-1,
                          keepdims=True), tf.float32)

        return y_true, y_pred * annot_mask

    def call(self, y_true, y_pred):
        '''
        Args:
        
        y_true: 1 hot masks
        y_pred: probability maps between [0,1] for each label
        '''

        y_true, y_pred = self._remove_unannot(y_true, y_pred)

        spatial_axis = tuple(range(1, len(y_true.shape) - 1))

        intersection = tf.reduce_sum(y_pred * y_true, axis=spatial_axis)
        union = tf.reduce_sum(
            (y_pred + y_true), axis=spatial_axis) - intersection
        jaccard = 1. - (intersection + self.eps) / (union + self.eps)

        return tf.math.reduce_mean(jaccard)


class HingedJaccardLoss(JaccardLoss):
    '''Similar to JaccardLoss, but with unbounded predictions
    
    hinge_thresh: Threshold over which pixels in prediction
        are replaced by the groundtruth (i.e. no backpropagation)
    '''
    def __init__(self, hinge_thresh=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hinge_thresh < 0.:
            raise ValueError('hinge_thresh {}<0.'.format(hinge_thresh))

        if hinge_thresh >= 0.5:
            raise ValueError('hinge_thresh {}>=0.5'.format(hinge_thresh))

        self.hinge_low = hinge_thresh
        self.hinge_high = 1. - hinge_thresh

    def call(self, y_true, y_pred):
        y_pred = hingify(y_true, y_pred, self.hinge_low, self.hinge_high)
        return super().call(y_true, y_pred)


class BinaryJaccardLoss(JaccardLoss):
    '''
    Single class variant of JaccardLoss.
    
    Expects integer labels instead of one hot. Negative values in y_true are excluded.
    
    Args:
    symmetric: if true returns the mean of losses over target class and its inverse 
    '''
    def __init__(self, symmetric=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.symmetric = symmetric

    def _annot_to_hot(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.minimum(y_true, 1)
        y_true = tf.cast(tf.one_hot(tf.squeeze(y_true, -1), depth=2),
                         tf.float32)
        y_pred = tf.concat([1. - y_pred, y_pred], axis=-1)

        return y_true, y_pred

    def _remove_unannot(self, y_true, y_pred):

        y_true, y_pred = super()._remove_unannot(y_true, y_pred)

        if not self.symmetric:
            # remove complementary binary channel, AFTER computing unannot mask
            y_true = y_true[..., -1:]
            y_pred = y_pred[..., -1:]

        return y_true, y_pred

    def call(self, y_true, y_pred):

        y_true, y_pred = self._annot_to_hot(y_true, y_pred)

        return super().call(y_true, y_pred)


class HingedBinaryJaccardLoss(BinaryJaccardLoss, HingedJaccardLoss):
    pass
