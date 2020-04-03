import numpy as np
from librosa.sequence import dtw
import fastdtw
from .. import utils
import pickle
from asmd import audioscoredataset as asd
import os
from . import cdist
import random

N_JOBS = -4
DISTS = [
    'euclidean', 'cosine', 'canberra', 'chebyshev', 'braycurtis',
    'correlation', 'manhattan'
]
RADIUS_DTW = [0.01, 0.2, 0.5, 1.0]
RADIUS_FASTDTW = [1, 5, 10, 20]
RES = 0.1
FASTDTW = True
NUM_SONGS = 160


# hack to let fastdtw accept float32
def _my_prep_inputs(x, y, dist):
    return x, y


def evaluate(i, dataset, dist, radius, use_fastdtw=False):

    aligned = dataset.get_score(i, score_type=['precise_alignment', 'broad_alignment'])
    misaligned = dataset.get_score(i, score_type=['non_aligned'])
    errors = np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))

    pr_aligned = utils.make_pianoroll(aligned, res=RES,
                                      velocities=False).astype(np.float32)
    pr_misaligned = utils.make_pianoroll(misaligned, res=RES,
                                         velocities=False).astype(np.float32)

    if not use_fastdtw:
        # computing distance matrix with float32 and thread parallelization
        # (cython)
        dist_matrix = cdist.cdist(pr_aligned, pr_misaligned, metric=dist)
        # this isn't thread parallelizing...
        _D, path = dtw(C=dist_matrix, global_constraints=True, band_rad=radius)
        path = path[::-1]
    else:
        # hack to let fastdtw accept float32
        fastdtw._fastdtw.__prep_inputs = _my_prep_inputs
        dist = getattr(cdist, dist)
        _D, path = fastdtw.fastdtw(pr_aligned.T,
                                   pr_misaligned.T,
                                   dist=dist,
                                   radius=radius)
        path = np.array(path)

    # converting indices to seconds
    path = path * RES

    # interpolating
    misaligned[:, 1] = np.interp(misaligned[:, 1], path[:, 1], path[:, 0])
    misaligned[:, 2] = np.interp(misaligned[:, 2], path[:, 1], path[:, 0])

    # evaluating
    errors -= np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))
    return errors


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    best = -9999
    for dist in DISTS:
        dataset = asd.Dataset().filter(datasets=['MusicNet'],
                                       instruments=['piano'])
        random.seed(1992)
        dataset.paths = random.sample(dataset.paths, NUM_SONGS)
        RADIUS = RADIUS_FASTDTW if FASTDTW else RADIUS_DTW
        for radius in RADIUS:
            print(f"Testing {dist} - {radius}")
            print(f"Number of songs: {len(dataset)}")
            data = dataset.parallel(evaluate,
                                    dist,
                                    radius,
                                    n_jobs=N_JOBS,
                                    backend="multiprocessing",
                                    use_fastdtw=FASTDTW)
            # removing Nones...
            # l1 = len(data)
            # data = [i for i in data if i is not None]
            # num_none = len(data) - l1

            # logging!
            m = log(data)
            if m > best:
                best = m
                best_params = [(dist, radius)]
            elif m == best:
                best_params.append((dist, radius))

            fname = os.path.join('results', f'dtw-{dist}-{radius:.2f}.pkl')
            pickle.dump(
                {
                    # 'num_none': num_none,
                    'data': data
                },
                open(fname, 'wb'))
            del data

    print(
        f"Best parameters for midi2midi over piano are (dist, radius): {best_params}"
    )


def log(data):
    data = np.hstack(data)
    m1 = np.mean(data[0])
    s1 = np.std(data[0])
    print(f'Error improvement ons (avg, std): {m1:.4E}, {s1:.4E}')
    m2 = np.mean(data[1])
    s2 = np.std(data[1])
    print(f'Error improvement offs (avg, std): {m2:.4E}, {s2:.4E}')
    m3 = np.mean(data)
    s3 = np.std(data)
    print(f'Error improvement mean (avg, std): {m3:.4E}, {s3:.4E}')
    return m3


if __name__ == "__main__":
    main()
