import pytest
import numpy as np
import tensorflow as tf

from rdcnet.losses.utils import hingify


def test_hingify():

    yt = np.array([[0., 0., 0., 0., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 0., 0., 0., 0.]])

    yp_hinged = hingify(yt,
                        np.array([[-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3],
                                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]),
                        hinge_low=0.,
                        hinge_high=1.)
    np.testing.assert_almost_equal(
        yp_hinged,
        np.array([[0., 0., 0.1, 0.3, 0.7, 0.9, 1., 1.],
                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]))

    yp_hinged = hingify(yt,
                        np.array([[-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3],
                                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]),
                        hinge_low=0.25,
                        hinge_high=1.)
    np.testing.assert_almost_equal(
        yp_hinged,
        np.array([[0., 0., 0., 0.3, 0.7, 0.9, 1., 1.],
                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]))

    yp_hinged = hingify(yt,
                        np.array([[-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3],
                                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]),
                        hinge_low=0.,
                        hinge_high=0.75)
    np.testing.assert_almost_equal(
        yp_hinged,
        np.array([[0., 0., 0.1, 0.3, 0.7, 1., 1., 1.],
                  [-0.3, 0., 0.1, 0.3, 0.7, 0.9, 1., 1.3]]))
