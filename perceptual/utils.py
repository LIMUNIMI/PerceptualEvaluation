import numpy as np
from fastcluster import linkage_vector, linkage
from scipy.cluster.hierarchy import fcluster
import essentia.standard as es

PLOT = False


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
        kth_cluster = samples[clusters == i + 1]
        others = samples[clusters != i + 1]

        others_centroid = np.mean(others, axis=0)

        kth_cluster_point = np.argmax(
            np.mean(np.abs(kth_cluster - others_centroid), axis=1))
        kth_cluster_indices = np.where(clusters == i + 1)[0]

        out[i] = kth_cluster_indices[kth_cluster_point]

    if PLOT:
        from scipy.spatial.distance import pdist, squareform
        import visdom
        vis = visdom.Visdom()
        mat = squareform(pdist(samples))
        m = np.argsort(np.sum(mat, axis=1))[-k:]
        vis.heatmap(mat)
        win = vis.scatter(X=samples, Y=clusters, opts={"markersize": 5})
        vis.scatter(X=samples[out], update="append", name="chosen", win=win, opts={
            "markersymbol": "cross-open", "markersize": 10})
        vis.scatter(X=samples[m], update="append", name="selfsim", win=win, opts={
            "markersymbol": "square", "markersize": 10})

    return out


def midi_pitch_to_f0(midi_pitch):
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch-69)/12)


def find_start_stop(audio, sample_rate=44100, seconds=False):
    """
    Returns a tuple containing the start and the end of sound in an audio array.

    ARGUMENTS:
    `audio` : essentia.array
        an essentia array or numpy array containing the audio
    `sample_rate` : int
        sample rate
    `seconds` : boolean
        if True, results will be expressed in seconds (float)

    RETURNS:
    `start` : int or float
        the sample where sound starts or the corresponding second

    `end` : int or float
        the sample where sound ends or the corresponding second
    """
    processer = es.StartStopSilence(threshold=-60)
    for frame in es.FrameGenerator(audio, frameSize=512, hopSize=128,
                                   startFromZero=True):
        start, stop = processer(frame)

    if seconds:
        start = specframe2sec(start, sample_rate, 128, 512)
        stop = specframe2sec(stop, sample_rate, 128, 512)
    else:
        start = int(specframe2sample(start, 128, 512))
        stop = int(specframe2sample(stop, 128, 512))

    if start == 256:
        start = 0

    return start, stop


def specframe2sec(frame, sample_rate=44100, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return specframe2sample(frame, hop_size, win_len) / sample_rate


def specframe2sample(frame, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return frame*hop_size + win_len / 2
