import pytest
import numpy as np

from rdcnet.postprocessing.voting import count_votes, embeddings_to_labels


def test_3D_count_votes():
    '''Tests 3D Hough voting'''

    np.random.seed(42)

    embeddings = np.zeros((20, 20, 20, 3), dtype=np.float32)
    embeddings[2:5, 8:12, 13:19] = [3, 9, 15]
    embeddings[10:13, 1:4, 2:5] = [11, 3, 2]

    votes = count_votes(embeddings[embeddings[..., 0] > 0],
                        (20, 20, 20)).numpy()

    assert votes[3, 9, 15] == 72
    assert votes[11, 3, 2] == 27
    assert votes.sum() == 99


def test_2D_count_votes():
    '''Tests 2D Hough voting'''

    np.random.seed(42)

    embeddings = np.zeros((20, 20, 2), dtype=np.float32)
    embeddings[2:5, 8:12] = [3, 9]
    embeddings[10:13, 1:4] = [11, 3]

    votes = count_votes(embeddings[embeddings[..., 0] > 0], (20, 20)).numpy()

    assert votes[3, 9] == 12
    assert votes[11, 3] == 9
    assert votes.sum() == 21


def test_embeddings_to_labels():
    '''Test conversion of spatial embeddings to labels using a Hough voting scheme'''

    np.random.seed(42)

    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 30:70] = True
    mask[80:90, 80:90] = True

    embeddings = np.zeros((100, 100, 2), dtype=np.float32)
    embeddings[10:20, 30:70] = [12, 49]
    embeddings[80:90, 80:90] = [85, 82]
    embeddings += np.random.rand(100, 100, 2) * 2

    labels = embeddings_to_labels(embeddings,
                                  mask,
                                  peak_min_distance=10,
                                  spacing=1.,
                                  min_count=5).numpy()

    assert np.all(labels[10:20, 30:70] == 1) or np.all(labels[10:20,
                                                              30:70] == 2)
    assert np.all(labels[80:90, 80:90] == 1) or np.all(labels[80:90,
                                                              80:90] == 2)
    assert np.all(labels[~mask] == 0)
    assert np.unique(labels).tolist() == [0, 1, 2]
