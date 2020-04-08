from pprint import pprint
from .utils import midipath2mat, midi_pitch_to_f0
import mir_eval
import numpy as np


def compare_midi(fname1, fname2):
    t1, p1, v1 = open_midi(fname1)
    t2, p2, v2 = open_midi(fname2)

    # remove initial silence
    t1 -= np.min(t1)
    t2 -= np.min(t2)

    if t1.shape == t2.shape:
        print(f"MAE timing error: {np.mean(np.abs(t1 - t2)): .4}")

    evaluation = mir_eval.transcription.evaluate(t1, p1, t2, p2)

    print("\nEvaluation without velocity: ")
    pprint(evaluation)

    evaluation = mir_eval.transcription_velocity.evaluate(
        t1, p1, v1, t2, p2, v2, rcond=None)

    print("\nEvaluation with velocity: ")
    pprint(evaluation)


def open_midi(fname):
    mat = midipath2mat(fname)
    # sorting according to pitches and then onsets
    mat = mat[np.lexsort((mat[:, 1], mat[:, 0]))]
    times = mat[:, (1, 2)]
    pitches = midi_pitch_to_f0(mat[:, 0])
    vel = mat[:, 3]
    return times, pitches, vel


if __name__ == "__main__":
    import sys

    def show_usage():
        print("Usage: " + sys.argv[0] + " [reference] [est]")

    if len(sys.argv) != 3:
        show_usage()
    else:
        try:
            compare_midi(sys.argv[1], sys.argv[2])
        except Exception as e:
            print(e)
            show_usage()
