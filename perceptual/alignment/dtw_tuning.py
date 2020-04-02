import numpy as np
from librosa.sequence import dtw
from .. import utils
import pickle
from asmd import audioscoredataset as asd
import os
from .cdist import cdist

N_JOBS = 2
DISTS = [
    'euclidean', 'cosine', 'canberra', 'chebyshev', 'braycurtis',
    'correlation', 'manhattan'
]
RADIUS = [1.00, 0.75, 0.5]
RES = 0.02


def evaluate(i, dataset, dist, radius):
    aligned = dataset.get_score(i, score_type='precise_alignment')
    misaligned = dataset.get_score(i, score_type='non_aligned')
    errors = np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))

    pr_aligned = utils.make_pianoroll(aligned, res=RES,
                                      velocities=False).astype(np.float32)
    pr_misaligned = utils.make_pianoroll(misaligned, res=RES,
                                         velocities=False).astype(np.float32)

    # computing distance matrix with float32 and thread parallelization (cython)
    dist_matrix = cdist(pr_aligned, pr_misaligned, metric=dist)
    # this isn't thread parallelizing...
    _D, path = dtw(C=dist_matrix, global_constraints=True, band_rad=radius)

    # converting indices to seconds
    path = path[::-1] * RES

    # interpolating
    misaligned[:, 1] = np.interp(misaligned[:, 1], path[:, 0], path[:, 1])
    misaligned[:, 2] = np.interp(misaligned[:, 2], path[:, 0], path[:, 1])

    # evaluating
    errors = np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))
    return errors


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    best = 99999
    for dist in DISTS:
        dataset = asd.Dataset().filter(datasets=['MusicNet'],
                                       instruments=['piano'])
        for radius in RADIUS:
            print(f"Testing {dist} - {radius}")
            print(f"Number of songs: {len(dataset)}")
            data = dataset.parallel(evaluate,
                                    dist,
                                    radius,
                                    n_jobs=N_JOBS,
                                    backend="multiprocessing")
            # removing Nones...
            # l1 = len(data)
            # data = [i for i in data if i is not None]
            # num_none = len(data) - l1

            # logging!
            m = log(data)
            if m < best:
                best = m
                best_params = [(dist, radius)]
            elif m == best:
                best_params.append((dist, radius))

            fname = os.path.join('results', f'dtw-{dist}-{radius:.2f}.pkl')
            pickle.dump({
                # 'num_none': num_none,
                'data': data
            }, open(fname, 'wb'))
            del data

    print(
        f"Best parameters for midi2midi over piano are (dist, radius): {best_params}"
    )


def log(data):
    data = np.hstack(data)
    m1 = np.mean(data[0])
    s1 = np.std(data[0])
    print(f'Error ons (avg, std): {m1:.4E}, {s1:.4E}')
    m2 = np.mean(data[1])
    s2 = np.std(data[1])
    print(f'Error offs (avg, std): {m2:.4E}, {s2:.4E}')
    m3 = np.mean(data)
    s3 = np.std(data)
    print(f'Error mean (avg, std): {m2:.4E}, {s3:.4E}')
    return m3


if __name__ == "__main__":
    main()
