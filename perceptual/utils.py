import numpy as np
import pretty_midi as pm
from fastcluster import linkage_vector, linkage
from scipy.cluster.hierarchy import fcluster
import essentia.standard as es

PLOT = False


def farthest_points(samples, k, n, optimize=False):
    """
    Computes `n` sets of `k` indices of farthest points.

    First, it clusterizes `samples` with ward-linkage to find `k` clusters.
    Then, for each cluster, it looks for the `n` points which have the largest
    distance from the centroid of all the other points. The order of the `n`
    points in each cluster is randomized. This method ensure that each of the
    `n` sets contains one point in each cluster.

    Arguments
    ---------

    `samples` : np.ndarray
        the set of samples with dimensions (N, D), where N is the number of
        samples and D is the number of features for each sample

    `k` : int
        the number of clusters to find

    `n` : int
        the number of points per cluster to find

    `optimize` : bool
        if using optimized algorithm which takes O(ND) instead of O(N^2).

    Returns
    -------

    np.ndarray
        array of `np.int64` representing the indices of the farthest samples
        shape: (k, n)
    """
    assert samples.shape[0] > k, "samples cardinality < k"

    if optimize:
        linkage_func = linkage_vector
    else:
        linkage_func = linkage
    Z = linkage_func(samples, method='ward')
    clusters = fcluster(Z, k, criterion='maxclust')

    out = np.empty((k, n), dtype=np.int64)
    for i in range(k):
        kth_cluster = samples[clusters == i + 1]
        kth_cluster_indices = np.where(clusters == i + 1)[0]
        others_centroid = np.empty_like(kth_cluster)

        for l, j in enumerate(kth_cluster_indices):
            others = np.concatenate((samples[:j], samples[j + 1:]))
            others_centroid[l] = np.mean(others, axis=0)

        # np.partition returns an array with the `-n`-th element in the
        # position it would be if sorted and all larger value after it (not
        # sorted, though
        kth_cluster_points = np.argpartition(
            np.mean(np.abs(kth_cluster - others_centroid), axis=1), -n)[-n:]

        out[i] = kth_cluster_indices[kth_cluster_points]
    np.random.shuffle(out.T)

    if PLOT:
        from scipy.spatial.distance import pdist, squareform
        import visdom
        vis = visdom.Visdom()
        mat = squareform(pdist(samples))
        m = np.argsort(np.sum(mat, axis=1))[-k:]
        vis.heatmap(mat)
        win = vis.scatter(X=samples, Y=clusters, opts={"markersize": 5})
        vis.scatter(X=samples[out[:, 0]],
                    update="append",
                    name="chosen",
                    win=win,
                    opts={
                        "markersymbol": "cross-open",
                        "markersize": 10
        })
        vis.scatter(X=samples[m],
                    update="append",
                    name="selfsim",
                    win=win,
                    opts={
                        "markersymbol": "square",
                        "markersize": 10
        })

    return out


def midi_pitch_to_f0(midi_pitch):
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch - 69) / 12)


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
    for frame in es.FrameGenerator(audio,
                                   frameSize=512,
                                   hopSize=128,
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

    return frame * hop_size + win_len / 2


def midipath2mat(path):
    """
    Open a midi file  with one instrument track and construct a mat like asmd:

    pitch, start (sec), end (sec), velocity
    """

    out = []
    notes = pm.PrettyMIDI(midi_file=path).instruments[0].notes

    for note in notes:
        out.append([note.pitch, note.start, note.end, note.velocity])

    return np.array(out)


def make_pianoroll(mat,
                   res=0.25,
                   velocities=True,
                   basis=1,
                   attack=1,
                   basis_l=1):
    """
    return a pianoroll starting from a mat score from asmd

    if velocities are available, it will be filled with velocity values; to
    turn this off use `velocities=False`

    `basis` is the number of basis for the nmf; `attack` is the attack
    duration, all other basis will be long `basis_l` column except the last one
    that will last till the end if needed
    """

    L = int(np.max(mat[:, 2]) / res) + 1

    pr = np.zeros((128, basis, L))

    for note in mat:
        pitch = int(note[0])
        vel = int(note[3])
        start = int(np.round(note[1] / res))
        end = int(np.round(note[2] / res)) + 1
        if velocities:
            vel = max(1, vel)
        else:
            vel = 1

        # the attack basis
        pr[pitch, 0, start:start + attack] = vel

        start += attack

        # all the other basis
        END = False
        for b in range(basis):
            for k in range(basis_l):
                t = start + b * basis_l + k
                if t < end:
                    pr[pitch, b, t] = vel
                else:
                    END = True
                    break
            if END:
                break

        # the ending part
        if start + (basis - 1) * basis_l < end:
            pr[pitch, basis - 1, start + (basis - 1) * basis_l:end] = vel

    # collapse pitch and basis dimension
    pr = pr.reshape((128 * basis, -1), order='C')
    return pr


def stretch_pianoroll(pr, out_length):
    """
    Stretch a pianoroll along the second dimension.
    """
    ratio = pr.shape[1] / out_length
    return np.array(
        list(
            map(lambda i: pr[:, min(round(i * ratio), pr.shape[1] - 1)],
                range(out_length)))).T


def evaluate2d(estimate, ground_truth):
    """
    Evaluate two 2D arrays in which rows are notes and columns are `pitch`,
    `onset` and `offset`.

    This function first compare the number of notes in the two arrays for all
    the pitches and removes notes in excess, so that the two arrays have the
    same number of pitches.  Then, it returns two arrays with onsets and
    offsets relative errors, computed as `estimate - ground_truth` for all
    correspondend pitches, after having sorted by pitch and onset. Ordering is
    performed so that the input
    arrays don't need to be ordered in the same way.


    Arguments
    ---

    `estimate` : np.array
        The array of estimated timings. 2D array where rows are notes and
        columns are `pitch`, `onsets`, `offsets`

    `ground_truth` : np.array
        The array of ground_truth timings. 2D array where rows are notes and
        columns are `pitch`, `onsets`, `offsets`

    Returns
    ---

    `np.array` :
        A 1D array where the i element is the relative error computed as for
        the `i`-th estimated note onset, after having removed mismatching notes
        and having ordered by pitch and onset. Ordering is performed so that
        the input arrays don't need to be ordered in the same way.


    `np.array` :
        Same as the first output but for offsets.
    """
    ###########
    # removing last k pitches that create mismatch
    # after this operation all the pitches have the same cardinality in both
    # lists
    pitches_est = np.unique(estimate[:, 0])
    pitches_gt = np.unique(ground_truth[:, 0])
    pitches = np.union1d(pitches_est, pitches_gt)
    for pitch in pitches:
        # computing how many notes for this pitch there are in estimate and
        # ground_truth
        pitch_est = np.count_nonzero(estimate[:, 0] == pitch)
        pitch_gt = np.count_nonzero(ground_truth[:, 0] == pitch)

        # deciding from which we should remove notes of this pitch
        if pitch_est > pitch_gt:
            remove_from, not_remove_from = estimate, ground_truth
            pitch_not_remove_from = pitch_gt
        elif pitch_est < pitch_gt:
            remove_from, not_remove_from = ground_truth, estimate
            pitch_not_remove_from = pitch_est
        else:
            continue

        # taking indices of notes with this pitch in remove_from that are not
        # in not_remove_from
        remove_from_idx = np.where(remove_from[:, 0] == pitch)[
            0][pitch_not_remove_from:]

        # remove from remove_from
        remove_from = np.delete(remove_from, remove_from_idx, 0)

        # reassigning names
        if pitch_est > pitch_gt:
            estimate, ground_truth = remove_from, not_remove_from
        elif pitch_est < pitch_gt:
            ground_truth, estimate = remove_from, not_remove_from

    ###########
    # sorting according to pitches and then onsets
    est_sorted = np.lexsort((estimate[:, 1], estimate[:, 0]))
    gt_sorted = np.lexsort((ground_truth[:, 1], ground_truth[:, 0]))

    # computing errors
    _err_ons = estimate[est_sorted, 1] - ground_truth[gt_sorted, 1]
    _err_offs = estimate[est_sorted, 2] - ground_truth[gt_sorted, 2]

    # sorting errors according to input
    err_ons = np.empty_like(_err_ons)
    err_offs = np.empty_like(_err_offs)
    err_ons[est_sorted] = _err_ons
    err_offs[est_sorted] = _err_offs

    return err_ons, err_offs
