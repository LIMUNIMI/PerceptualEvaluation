#!/usr/bin/env python3

import sys
import csv

import numpy as np

from asmd import audioscoredataset


class Tester:

    def __init__(self):
        if len(sys.argv) < 1:
            print("Error: missing algorithm, provie `ewert` or `amt`")
            return 2

        if sys.argv[1] == "ewert":
            from .ewert.align import audio_to_score_alignment
            self.fname = "ewert.csv"
        elif sys.argv[1] == "amt":
            from .align_with_amt import audio_to_score_alignment
            self.fname = "amt.csv"
        else:
            print("Error: missing algorithm, provie `ewert` or `amt`")
            return 3

        self.align = audio_to_score_alignment

        self.data = audioscoredataset.Dataset()
        self.data.filter(datasets=["SMD"])

    def run(self):
        errors = self.data.parallel(self.path_processing, n_jobs=-2)

        with open(self.fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(errors)

    def path_processing(self, i, data):
        print(f"    Running Alignment on {data.paths[i][2][0]}")
        mat = data.get_score(i, score_type='non_aligned')
        gt = data.get_score(i, score_type='precise_alignment')
        audio, sr = data.get_audio(i)

        new_ons, new_offs = self.align(mat, audio, sr)

        # computing errors
        err_ons = new_ons - gt[:, 1]
        err_offs = new_offs - gt[:, 2]

        # interleaving lists
        err = np.empty((2*err_ons.shape[0],))
        err[::2] = err_ons
        err[1::2] = err_offs
        print(
            f"{np.mean(err_ons):.2E}, {np.mean(err_offs):.2E}" +
            f", {np.std(err_ons):.2E}, {np.std(err_offs):.2E}")
        return err


if __name__ == "__main__":
    t = Tester()
    t.run()
