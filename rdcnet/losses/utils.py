import tensorflow as tf


def hingify(y_true, y_pred, hinge_low=0., hinge_high=1.):
    '''Replaces y_pred values under low_hinge threshold by groundtruth 
    values if groundtruth<=0. and vice versa'''

    y_pred = tf.where(~tf.cast(y_true, tf.bool) & (y_pred < hinge_low), y_true,
                      y_pred)
    y_pred = tf.where(
        tf.cast(y_true, tf.bool) & (y_pred > hinge_high), y_true, y_pred)

    return y_pred
