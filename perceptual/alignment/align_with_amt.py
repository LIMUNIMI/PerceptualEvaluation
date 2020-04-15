import fastdtw
import numpy as np
from .. import utils
from . import cdist
from .. import magenta_transcription


START_NOTE = 21
EPS = np.finfo(np.float64).eps


def _my_prep_inputs(x, y, dist):
    """
    Fastdtw sucks too and convicts you to use float64...
    """
    return x, y


def transcription(audio, sr, res=0.02, cuda=False):

    prediction_list = magenta_transcription.transcribe(audio, sr, res, cuda)

    frame_predictions = prediction_list[0]['frame_predictions'][0].astype(
        np.float)
    frame_predictions += prediction_list[0]['onset_predictions'][0].astype(
        np.float)

    # convert to pianoroll with resolution `res`
    n_cols = round(len(audio) / sr / res)
    frame_predictions = utils.stretch_pianoroll(frame_predictions.T, n_cols)
    pr = np.zeros((128, n_cols))
    pr[START_NOTE:START_NOTE + 88] = frame_predictions
    return pr


def audio_to_score_alignment(misaligned, audio, sr, res=0.02):
    # removing trailing silence
    start, stop = utils.find_start_stop(audio, sample_rate=sr)
    audio = audio[start:stop]

    misaligned[:, 1:3] -= np.min(misaligned[:, 1])

    # force input duration to last as audio
    audio_duration = (stop - start) / sr
    misaligned_duration = np.max(misaligned[:, 2])
    misaligned[:, 1:3] *= (audio_duration / misaligned_duration)

    # computing features
    audio_features = transcription(audio, sr, res) + EPS
    pianoroll = utils.make_pianoroll(misaligned, res=res, velocities=False) + EPS
    pianoroll += utils.make_pianoroll(
        misaligned, res=res, velocities=False, only_onsets=True)

    # force exactly the same shape
    L = min(pianoroll.shape[1], audio_features.shape[1])
    pianoroll = pianoroll[:, :L]
    audio_features = audio_features[:, :L]

    # parameters for dtw were chosen with midi2midi on musicnet (see dtw_tuning)
    # hack to let fastdtw accept float32
    fastdtw._fastdtw.__prep_inputs = _my_prep_inputs
    _D, path = fastdtw.fastdtw(pianoroll.astype(np.float32).T,
                               audio_features.astype(np.float32).T,
                               dist=cdist.cosine,
                               radius=10)
    path = np.array(path) * res

    # interpolating
    new_ons = np.interp(misaligned[:, 1], path[:, 0], path[:, 1])
    new_offs = np.interp(misaligned[:, 2], path[:, 0], path[:, 1])

    new_ons += (start / sr)
    new_offs += (start / sr)

    return new_ons, new_offs
