import pytest
import tensorflow as tf
import numpy as np

from rdcnet.losses.jaccard_loss import JaccardLoss, HingedJaccardLoss, BinaryJaccardLoss, HingedBinaryJaccardLoss

np.random.seed(11)

RAND_5CLASSSES_LABELS = np.random.choice([0, 1, 2, 3, 4], size=(4, 10, 10, 1))


# reference implementations ############################################
def single_ch_numpy_iou(one_hot, probs, mask, eps=0.0):

    intersection = (one_hot[mask] * probs[mask]).sum()
    union = (one_hot[mask] + probs[mask]).sum() - intersection
    return 1. - (eps + intersection) / (eps + union)


def numpy_iou(one_hot, probs, mask, eps=0.0):
    losses = [
        single_ch_numpy_iou(one_hot[..., i], probs[..., i],
                            mask.squeeze(axis=-1), eps)
        for i in range(one_hot.shape[-1])
    ]
    return np.array(losses).mean()


def batched_numpy_iou(one_hot, probs, masks, eps=0.0):
    '''Reference numpy implementation taking hot encoded inputs'''

    losses = [
        numpy_iou(hot, prob, mask, eps)
        for hot, prob, mask in zip(one_hot, probs, masks)
    ]
    return np.array(losses).mean()


def keras_IoU(yt, yp, num_classes=2):

    mask = yt >= 0
    m = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    m.update_state(yt[mask], tf.cast(yp[mask], tf.int32))
    return 1. - m.result().numpy()


def batched_keras_IoU(yt, yp, num_classes=2):
    '''reference keras multi-class implementation taking labels as inputs'''

    losses = [keras_IoU(t, p, num_classes) for t, p in zip(yt, yp)]
    return np.array(losses).mean()


########################################################################


# yapf: disable
@pytest.mark.parametrize(
    "n_classes,yt,yp",[(5,np.random.choice([0,1,2,3,4], size=(4, 10, 10, 1)), np.random.choice([0,1,2,3,4], size=(4, 10, 10, 1))), # random 5 labels
                       (5,np.random.choice([1,3], size=(4, 10, 10, 1)), np.random.choice([0,1,2,3,4], size=(4, 10, 10, 1))), # biased target
                       (5,np.random.choice([0,1,2,3,4], size=(4, 10, 10, 1)), np.random.choice([1,4], size=(4, 10, 10, 1))), # biased pred
                       (5,np.random.choice([0,1,2,3,4], size=(4, 10, 20, 10, 1)), np.random.choice([0,1,2,3,4], size=(4, 10, 20, 10, 1))), # 3D
                       (5,RAND_5CLASSSES_LABELS,RAND_5CLASSSES_LABELS), # perfect prediction
                       (2,np.random.choice([0,1], size=(4, 10, 10, 1)), np.random.choice([0,1], size=(4, 10, 10, 1))), # symmetric binary outputs
                       (5,np.random.choice([-1,0,1,2,3,4], size=(4, 10, 10, 1)), np.random.choice([0,1,2,3,4], size=(4, 10, 10, 1))), # partial annotation
                       (3,np.random.choice([-1,-2,-10,0,1,2], size=(4, 10, 10, 1)), np.random.choice([0,1,2], size=(4, 10, 10, 1))), # partial annotation
                       ],
)
# yapf: enable
def test_JaccardLoss(n_classes, yt, yp):
    '''Verifies that the soft Jaccard loss behaves as keras MeanIoU when 
    probabilities are either 0 or 1 and is >0 otherwise
    '''

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)
    loss = JaccardLoss(eps=0)(one_hot, probs)

    expected_loss_keras = batched_keras_IoU(yt, yp, num_classes=n_classes)
    expected_loss_numpy = batched_numpy_iou(one_hot.numpy(), probs.numpy(),
                                            yt >= 0)

    np.testing.assert_almost_equal(loss, expected_loss_keras)
    np.testing.assert_almost_equal(loss, expected_loss_numpy)

    # replace hard prediction by 0.1, 0.9 --> imperfect loss
    probs = 0.8 * probs + 0.1
    loss = JaccardLoss(eps=0)(one_hot, probs)
    assert loss > 0.


@pytest.mark.parametrize(
    "eps",
    [0., 1e-6, 0.001, 1., 10.],
)
def test_JaccardLoss_epsilon(eps):
    '''Verifies that the eps param as the same effect as in numpy implementation
    '''

    yt = np.random.choice([0, 1, 2, 3, 4], size=(4, 10, 10, 1))
    yp = np.random.choice([0, 1, 2, 3, 4], size=(4, 10, 10, 1))

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), 5), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), 5), tf.float32)

    expected_loss_numpy = batched_numpy_iou(one_hot.numpy(), probs.numpy(),
                                            yt >= 0, eps)
    loss = JaccardLoss(eps=eps)(one_hot, probs)

    np.testing.assert_almost_equal(loss, expected_loss_numpy)

    # check that eps does "something"
    assert not np.allclose(loss, JaccardLoss(eps=eps + 0.1)(one_hot, probs))


def test_HingedJaccardLoss():
    '''Test that hinge loss falls back to hard prediction when pred is 
    over the hinge threshold'''

    np.random.seed(25)
    n_classes = 5

    yt = np.random.choice(range(n_classes), size=(4, 10, 10, 1))
    yp = yt

    one_hot = tf.cast(tf.one_hot(tf.squeeze(yt, -1), n_classes), tf.float32)
    probs = tf.cast(tf.one_hot(tf.squeeze(yp, -1), n_classes), tf.float32)

    # yp == to groundtruth --> perfect loss
    loss = HingedJaccardLoss(eps=0, hinge_thresh=0.)(one_hot, probs)
    np.testing.assert_almost_equal(loss, 0.)

    # replace hard prediction by 0.1, 0.9 --> imperfect loss
    probs = 0.8 * probs + 0.1
    loss = HingedJaccardLoss(eps=0, hinge_thresh=0.)(one_hot, probs)
    assert loss > 0.

    # hinged loss with  0.2 threshold should fall back to hard prediction
    loss = HingedJaccardLoss(eps=0, hinge_thresh=0.2)(one_hot, probs)
    np.testing.assert_almost_equal(loss, 0., decimal=3)


# yapf: disable
@pytest.mark.parametrize(
    "symmetric,yt,yp",[(False, np.random.choice([0, 1], size=(4, 10, 10, 1)), np.random.choice([0, 1], size=(4, 10, 10, 1))),
                       (False, np.random.choice([-1, -2, 0, 1], size=(4, 10, 10, 1)), np.random.choice([0, 1], size=(4, 10, 10, 1))),
                       (True, np.random.choice([0, 1], size=(4, 10, 10, 1)), np.random.choice([0, 1], size=(4, 10, 10, 1))),
                       (True, np.random.choice([-1, -2, 0, 1], size=(4, 10, 10, 1)), np.random.choice([0, 1], size=(4, 10, 10, 1))),
                      ],
)
# yapf: enable
def test_BinaryJaccardLoss(symmetric, yt, yp):
    '''Verifies that the soft Jaccard loss on binary classification task
    behaves as IoU calculated in numpy (asymetric) or keras mean IoU (symmetric).
    '''

    if symmetric:
        expected_loss = batched_keras_IoU(yt, yp, num_classes=2)
    else:
        expected_loss = batched_numpy_iou(yt, yp, yt >= 0)

    loss = BinaryJaccardLoss(eps=0, symmetric=symmetric)(yt,
                                                         yp.astype(np.float32))

    np.testing.assert_almost_equal(loss, expected_loss)


# yapf: disable
@pytest.mark.parametrize(
    "symmetric,yt",[(False, np.random.choice([0, 1], size=(4, 10, 10, 1))),
                       (False, np.random.choice([-1, -2, 0, 1], size=(4, 10, 10, 1))),
                       (True, np.random.choice([0, 1], size=(4, 10, 10, 1))),
                       (True, np.random.choice([-1, -2, 0, 1], size=(4, 10, 10, 1))),
                      ],
)
# yapf: enable
def test_HingedBinaryJaccardLoss(symmetric, yt):
    '''Test that hinge binary loss falls back to hard prediction when pred is 
    over the hinge threshold'''

    yp = yt.astype(np.float32)

    # yp == to groundtruth --> perfect loss
    loss = HingedBinaryJaccardLoss(eps=0, hinge_thresh=0.,
                                   symmetric=symmetric)(yt, yp)
    np.testing.assert_almost_equal(loss, 0.)

    # replace hard prediction by 0.1, 0.9 --> imperfect loss
    yp = 0.8 * yp + 0.1
    loss = HingedBinaryJaccardLoss(eps=0, hinge_thresh=0.,
                                   symmetric=symmetric)(yt, yp)
    assert loss > 0.

    # hinged loss with  0.2 threshold should fall back to hard prediction
    loss = HingedBinaryJaccardLoss(eps=0,
                                   hinge_thresh=0.2,
                                   symmetric=symmetric)(yt, yp)
    np.testing.assert_almost_equal(loss, 0., decimal=3)


def test_JaccardLoss_training():
    '''Verifies that the JaccardLoss can be used to learn a simple thresholding operation.'''

    np.random.seed(25)
    raw = np.random.normal(size=(1, 10, 10, 1)).astype(np.float32)
    yt = (raw > 0.0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors((raw, yt))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1,
                               kernel_size=1,
                               padding='same',
                               activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10.),
                  loss=JaccardLoss())

    loss_before = model.evaluate(dataset)
    model.fit(dataset, epochs=100)
    loss_after = model.evaluate(dataset)

    assert loss_before * 0.95 >= loss_after
    assert loss_after < 0.001


def test_BinaryJaccardLoss_training():
    '''Verifies that the BinaryJaccardLoss can be used to learn a simple thresholding operation.'''

    np.random.seed(25)
    raw = np.random.normal(size=(1, 10, 10, 1)).astype(np.float32)
    yt = (raw > 0.0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors((raw, yt))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1,
                               kernel_size=1,
                               padding='same',
                               activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10.),
                  loss=BinaryJaccardLoss())

    loss_before = model.evaluate(dataset)
    model.fit(dataset, epochs=100)
    loss_after = model.evaluate(dataset)

    assert loss_before * 0.95 >= loss_after
    assert loss_after < 0.001
