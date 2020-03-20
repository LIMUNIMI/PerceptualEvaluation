import numpy as np
from fastcluster import linkage_vector, linkage
from scipy.cluster.hierarchy import fcluster


def farthest_points(samples, k, optimize=False):
    """
    Returns a list of `k` indices referred to the `k` rows in `samples` whcih
    are farthest from each other, computed using wang-linkage and picking the
    point from each cluster which is the most distant one from the other
    clusters.

    Arguments
    ---
    `samples` : np.ndarray
        the set of samples with dimensions (N, D), where N is the number of
        samples and D is the number of features for each sample

    `k` : int
        the number of points to find

    `optimize` : bool
        if using optimized algorithm which takes O(ND) instead of O(N^2).

    Returns
    ---

    np.ndarray
        array of `np.int64` representing the indices of the farthest samples
    """
    assert samples.shape[0] > k, "samples cardinality < k"

    if optimize:
        linkage_func = linkage_vector
    else:
        linkage_func = linkage
    Z = linkage_func(samples, method='ward')
    clusters = fcluster(Z, k, criterion='maxclust')

    out = np.empty(k, dtype=np.int64)
    for i in range(k):
        kth_cluster = clusters[clusters == k]
        others = clusters[clusters != k]

        others_centroid = np.mean(others, axis=0)

        out[i] = np.argmax(np.abs(kth_cluster - others_centroid))

    return out


def midi_pitch_to_f0(midi_pitch):
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch-69)/12)
