import pytest
import numpy as np
import itertools

from scipy.ndimage import gaussian_filter as np_gaussian_filter
from rdcnet.improc import gaussian_filter, local_max


@pytest.mark.parametrize('sigma,truncate',
                         itertools.product([1, 3, 15, (2, 7)], [4, 2]))
def test_2D_gaussian_filter(sigma, truncate):
    '''Checks tensorflow implementation of gaussian filter against scipy reference'''

    x = np.zeros((101, 101), dtype=np.float32)
    x[50, 50] = 1.2
    x[90, 30] = -0.7
    x[20:30, 10:40] = 0.21

    y_ref = np_gaussian_filter(x, sigma, truncate=truncate, mode='nearest')
    gaussian = gaussian_filter(sigma=sigma, spatial_rank=2, truncate=truncate)
    y = gaussian(x[..., None]).numpy().squeeze()

    np.testing.assert_almost_equal(y, y_ref)


@pytest.mark.parametrize('sigma,truncate',
                         itertools.product([1, 3, (2, 2, 7)], [4, 2]))
def test_3D_gaussian_filter(sigma, truncate):
    '''Checks tensorflow implementation of gaussian filter against scipy reference'''

    x = np.zeros((101, 101, 50), dtype=np.float32)
    x[50, 50, 27] = 1.2
    x[90, 30, 2] = -0.7
    x[20:30, 10:40, 30:45] = 0.21

    y_ref = np_gaussian_filter(x, sigma, truncate=truncate, mode='nearest')
    gaussian = gaussian_filter(sigma=sigma, spatial_rank=3, truncate=truncate)
    y = gaussian(x[..., None]).numpy().squeeze()

    np.testing.assert_almost_equal(y, y_ref, decimal=6)


def peaks_to_set(peaks):

    peaks = peaks.tolist()
    peaks = [tuple(p) for p in peaks]
    return set(peaks)


def test_2D_local_max():
    '''Checks 2D local max extraction'''

    peak_img = np.zeros((100, 300), dtype=np.float32)
    peak_img[50, 17] = 15
    peak_img[90, 30] = 22
    peak_img[14:17, 200:203] = 100

    peak_img[1, 1] = 150
    peak_img[7, 7] = 150
    peak_img[1, 7] = 150
    peak_img[7, 1] = 150

    ref_peaks = np.array([[50, 17], [90, 30], [15, 201]])
    peaks = local_max(peak_img, min_distance=7, threshold=1, spacing=1).numpy()
    assert peaks_to_set(peaks) == peaks_to_set(ref_peaks)

    ref_peaks = np.array([[15, 201]])
    peaks = local_max(peak_img, min_distance=7, threshold=50,
                      spacing=1).numpy()
    assert peaks_to_set(peaks) == peaks_to_set(ref_peaks)

    ref_peaks = np.array([[50, 17], [90, 30], [15, 201], [1, 1], [7, 7],
                          [1, 7], [7, 1]])
    peaks = local_max(peak_img, min_distance=1, threshold=1, spacing=1).numpy()
    assert peaks_to_set(peaks) == peaks_to_set(ref_peaks)


def test_2D_local_max():
    '''Checks local max extraction in isotropic volume'''

    peak_img = np.zeros((16, 128, 128), dtype=np.float32)
    peak_img[10, 23, 17] = 15
    peak_img[12, 23, 17] = 22

    ref_peaks = np.array([])
    peaks = local_max(peak_img, min_distance=7, threshold=1, spacing=1).numpy()
    assert peaks_to_set(peaks) == peaks_to_set(ref_peaks)

    ref_peaks = np.array([[10, 23, 17], [12, 23, 17]])
    peaks = local_max(peak_img,
                      min_distance=7,
                      threshold=1,
                      spacing=(16, 2, 2)).numpy()
    assert peaks_to_set(peaks) == peaks_to_set(ref_peaks)
