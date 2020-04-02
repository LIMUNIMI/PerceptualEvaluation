#!/usr/bin/env python3

import sys
import csv

import numpy as np

from asmd import audioscoredataset
if len(sys.argv) < 1:
    print("Error: missing algorithm, provie `ewert` or `amt`")
    sys.exit(2)
if sys.argv[1] == "ewert":
    from .ewert.align import audio_to_score_alignment
    FNAME = "ewert.csv"
elif sys.argv[1] == "amt":
    from .align_with_amt import audio_to_score_alignment
    FNAME = "amt.csv"
else:
    print("Error: missing algorithm, provie `ewert` or `amt`")
    sys.exit(3)

SR = 22050
RES = 0.02


def path_processing(i, data):
    # print(f"    Running Alignment on {data.paths[i][2][0]}")
    gt = data.get_score(i, score_type='precise_alignment')
    mat = data.get_score(i, score_type='non_aligned')
    audio, sr = data.get_mix(i, sr=SR)
    new_ons, new_offs = audio_to_score_alignment(mat, audio, sr, res=RES)

    # computing errors
    err_ons = new_ons - gt[:, 1]
    err_offs = new_offs - gt[:, 2]

    # interleaving lists
    err = np.empty((2*err_ons.shape[0],))
    err[::2] = err_ons
    err[1::2] = err_offs
    del gt, mat, audio, new_ons, new_offs
    # print(
    #     f"{np.mean(err_ons):.2E}, {np.mean(err_offs):.2E}" +
    #     f", {np.std(err_ons):.2E}, {np.std(err_offs):.2E}")
    return err


if __name__ == "__main__":
    data = audioscoredataset.Dataset()
    data.filter(datasets=["SMD"])
    errors = data.parallel(path_processing, n_jobs=1)

    with open(FNAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(errors)
