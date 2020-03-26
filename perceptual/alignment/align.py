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
            def ewert_alignment(i, data):
                """
                this is needed because google sucks
                """
                from .ewert.align import audio_to_score_alignment
                mat = data.get_score(i, score_type='non_aligned')
                audio, sr = data.get_audio(i)
                return audio_to_score_alignment(mat, audio, sr)
            self.align = ewert_alignment
            self.fname = "ewert.csv"
        elif sys.argv[1] == "amt":
            def amt_alignment(i, data):
                """
                this is needed because google sucks and wants wav_data instead
                of audio in a system package
                """
                from .align_with_amt import audio_to_score_alignment
                mat = data.get_pianoroll(
                    i, score_type='non_aligned', resolution=0.25)
                path_file = data.paths[i][0]
                return audio_to_score_alignment(mat, path_file)
            self.align = amt_alignment
            self.fname = "amt.csv"
        else:
            print("Error: missing algorithm, provie `ewert` or `amt`")
            return 3

        self.data = audioscoredataset.Dataset()
        self.data.filter(datasets=["SMD"])

    def run(self):
        errors = self.data.parallel(self.path_processing, n_jobs=-2)

        with open(self.fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(errors)

    def path_processing(self, i, data):
        print(f"    Running Alignment on {data.paths[i][2][0]}")
        gt = data.get_score(i, score_type='precise_alignment')
        new_ons, new_offs = self.align(i, data)

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
