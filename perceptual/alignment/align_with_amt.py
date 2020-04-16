import fastdtw
import numpy as np
from copy import copy
from .. import utils
from . import cdist
from ..magenta_transcription import transcribe

START_NOTE = 21
EPS = np.finfo(np.float64).eps
#: how many realignment do
NUM_REALIGNMENT = 3
#: how many seconds for each hop size in fine alignment
FINE_HOP = [5, 2.5, 0.5]
# FINE_HOP = [90 / (2**i) for i in range(NUM_REALIGNMENT)]
#: how many seconds for each window in fine alignment
FINE_WIN = [10, 5, 1]


def _my_prep_inputs(x, y, dist):
    """
    Fastdtw sucks too and convicts you to use float64...
    """
    return x, y


def transcription(audio, sr, res=0.001, cuda=False):

    predicted_mat = transcribe(audio, sr, cuda)

    pianoroll = utils.make_pianoroll(predicted_mat, res=res,
                                     velocities=False) + EPS
    pianoroll += utils.make_pianoroll(predicted_mat,
                                      res=res,
                                      velocities=False,
                                      only_onsets=True)

    return pianoroll


def dtw_align(pianoroll, audio_features, misaligned, res, radius):
    """
    perform alignment and return new times
    """

    # parameters for dtw were chosen with midi2midi on musicnet (see dtw_tuning)
    # hack to let fastdtw accept float32
    fastdtw._fastdtw.__prep_inputs = _my_prep_inputs
    _D, path = fastdtw.fastdtw(pianoroll.astype(np.float32).T,
                               audio_features.astype(np.float32).T,
                               dist=cdist.cosine,
                               radius=radius)

    path = np.array(path) * res
    new_ons = np.interp(misaligned[:, 1], path[:, 0], path[:, 1])
    new_offs = np.interp(misaligned[:, 2], path[:, 0], path[:, 1])

    return new_ons, new_offs


def get_usable_features(misaligned, res, audio_features):
    """
    compute pianoroll and remove extra columns
    """
    pianoroll = utils.make_pianoroll(misaligned, res=res,
                                     velocities=False) + EPS
    pianoroll += utils.make_pianoroll(misaligned,
                                      res=res,
                                      velocities=False,
                                      only_onsets=True)

    # force exactly the same shape
    L = min(pianoroll.shape[1], audio_features.shape[1])
    pianoroll = pianoroll[:, :L]
    audio_features = audio_features[:, :L]
    return pianoroll, audio_features


def audio_to_score_alignment(misaligned, audio, sr, res=0.001):
    # removing trailing silence
    misaligned = copy(misaligned)
    start, stop = utils.find_start_stop(audio, sample_rate=sr)
    audio = audio[start:stop]

    misaligned[:, 1:3] -= np.min(misaligned[:, 1])

    # force input duration to last as audio
    audio_duration = (stop - start) / sr
    misaligned_duration = np.max(misaligned[:, 2])
    misaligned[:, 1:3] *= (audio_duration / misaligned_duration)

    # computing features
    audio_features = transcription(audio, sr, res) + EPS
    pianoroll, audio_features = get_usable_features(misaligned, res,
                                                    audio_features)

    # first alignment
    new_ons, new_offs = dtw_align(
        pianoroll, audio_features, misaligned, res, 10)
    misaligned[:, 1] = new_ons
    misaligned[:, 2] = new_offs

    # realign segment by segment
    for j in range(NUM_REALIGNMENT):
        pianoroll, audio_features = get_usable_features(misaligned, res,
                                                        audio_features)
        hop_size = int(FINE_HOP[j] // res)
        win_size = int(FINE_WIN[j] // res)
        num_win = int(pianoroll.shape[1] // hop_size)
        for i in range(num_win):
            start = i * hop_size
            end = min(i * hop_size + win_size, pianoroll.shape[1])
            indices_of_notes_in_win = np.argwhere(
                np.logical_and(misaligned[:, 1] >= start * res,
                               misaligned[:, 2] <= end * res))
            if indices_of_notes_in_win.shape[0] > 1:
                indices_of_notes_in_win = indices_of_notes_in_win[0]
            else:
                continue
            pr_win = pianoroll[:, start:end]
            au_win = audio_features[:, start:end]
            mis_win = misaligned[indices_of_notes_in_win]
            ons_win, offs_win = dtw_align(pr_win, au_win, mis_win, res, 1)
            mis_win[:, 1] = ons_win
            mis_win[:, 2] = offs_win

    misaligned[:, (1, 2)] += (start / sr)

    return misaligned[:, 1], misaligned[:, 2]
