#!/usr/bin/env python3

import sys
import csv
from .. import utils
import numpy as np

from asmd.asmd import audioscoredataset
if len(sys.argv) < 1:
    print("Error: missing algorithm, add `ewert` or `amt`")
    sys.exit(2)
if sys.argv[1] == "ewert":
    from .ewert.align import audio_to_score_alignment
    FNAME = "results/ewert.csv"
elif sys.argv[1] == "amt":
    from .align_with_amt import audio_to_score_alignment
    FNAME = "results/amt.csv"
else:
    print("Error: missing algorithm, provie `ewert` or `amt`")
    sys.exit(3)

SR = 22050
RES = 0.001
NJOBS = 10
EPS = 1e-15


def path_processing(i, data):
    # print(f"    Running Alignment on {data.paths[i][2][0]}")
    aligned = data.get_score(i, score_type='precise_alignment')
    misaligned = data.get_score(i, score_type='non_aligned')
    audio, sr = data.get_mix(i, sr=SR)
    start_errors = np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))

    new_ons, new_offs = audio_to_score_alignment(misaligned,
                                                 audio,
                                                 sr,
                                                 res=RES)

    misaligned[:, 1] = new_ons
    misaligned[:, 2] = new_offs

    end_errors = np.abs(np.vstack(utils.evaluate2d(misaligned, aligned)))

    # interleaving lists
    err = np.empty((2 * end_errors.shape[1], ))
    err[::2] = end_errors[0]
    err[1::2] = end_errors[1]
    # print(
    #     f"{np.mean(err_ons):.2E}, {np.mean(err_offs):.2E}" +
    #     f", {np.std(err_ons):.2E}, {np.std(err_offs):.2E}")
    return err, start_errors / (end_errors + EPS)


if __name__ == "__main__":
    data = audioscoredataset.Dataset().filter(datasets=['SMD'])
    results = data.parallel(path_processing, n_jobs=NJOBS)

    ratios = []
    errors = []
    for err, ratio in results:
        ratios.append(ratio)
        errors.append(err)

    ratios = np.hstack(ratios)
    print(
        f"Average ratio error after/before: {np.mean(ratios):.6E}, std: {np.std(ratios):.6E}"
    )

    with open(FNAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(errors)
