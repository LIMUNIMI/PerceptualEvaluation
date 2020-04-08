import tensorflow as tf
from magenta.music import audio_io
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from magenta.music.protobuf import music_pb2
import os
import logging
import fastdtw
import numpy as np
from .. import utils
from . import cdist

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Define model and load checkpoint
# Only needs to be run once.
CHECKPOINT_DIR = './perceptual/magenta/train/'

START_NOTE = 21
EPS = np.finfo(np.float64).eps


def google_sucks(samples, sr):
    """
    Google sucks and want to use audio path (raw wav) instead of decoded
    samples loosing in decoupling between file format and DSP.
    This hack overwrites their stupid loader which writed data to a temprorary
    file and reopen it
    """

    return samples


def _my_prep_inputs(x, y, dist):
    """
    Fastdtw sucks too and convicts you to use float64...
    """
    return x, y


def transcription(audio, sr, res=0.02, cuda=False):
    """
    Google sucks and want to use audio path (raw wav) instead of decoded
    samples loosing in decoupling between file format and DSP
    """

    # simple hack because google sucks... in this way we can accept audio data
    # already loaded and mantain our reasonable interface (and decouple i/o
    # from processing)
    original_google_sucks = audio_io.wav_data_to_samples
    audio_io.wav_data_to_samples = google_sucks
    audio = np.array(audio)
    config = configs.CONFIG_MAP['onsets_frames']
    hparams = config.hparams
    hparams.use_cudnn = cuda
    hparams.batch_size = 1
    examples = tf.placeholder(tf.string, [None])

    dataset = data.provide_batch(examples=examples,
                                 preprocess_examples=True,
                                 params=hparams,
                                 is_training=False,
                                 shuffle_examples=False,
                                 skip_n_initial_records=0)

    estimator = train_util.create_estimator(config.model_fn, CHECKPOINT_DIR,
                                            hparams)

    iterator = dataset.make_initializable_iterator()
    next_record = iterator.get_next()

    example_list = list(
        audio_label_data_utils.process_record(wav_data=audio,
                                              sample_rate=sr,
                                              ns=music_pb2.NoteSequence(),
                                              example_id="fakeid",
                                              min_length=0,
                                              max_length=-1,
                                              allow_empty_notesequence=True,
                                              load_audio_with_librosa=False))
    assert len(example_list) == 1
    to_process = [example_list[0].SerializeToString()]

    sess = tf.Session()

    sess.run([
        tf.initializers.global_variables(),
        tf.initializers.local_variables()
    ])

    sess.run(iterator.initializer, {examples: to_process})

    def transcription_data(params):
        del params
        return tf.data.Dataset.from_tensors(sess.run(next_record))

    # put back the original function (it still writes and reload... stupid
    # though
    audio_io.wav_data_to_samples = original_google_sucks
    input_fn = infer_util.labels_to_features_wrapper(transcription_data)

    prediction_list = list(
        estimator.predict(input_fn, yield_single_examples=False))

    assert len(prediction_list) == 1

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
    pianoroll = utils.make_pianoroll(misaligned, res=res,
                                      velocities=False) + EPS
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

    # DEBUG
    # misaligned[:, 1] = new_ons
    # misaligned[:, 2] = new_offs
    # pr_after = utils.make_pianoroll(
    #     misaligned, res=res, velocities=False, only_onsets=True) + EPS
    # pr_after += utils.make_pianoroll(misaligned, res=res, velocities=False) + EPS
    # import visdom
    # vis = visdom.Visdom()
    # vis.heatmap(pianoroll, opts={"title": "pianoroll"})
    # vis.heatmap(pr_after, opts={"title": "pr_after"})
    # vis.heatmap(audio_features, opts={"title": "audio"})
    # vis.heatmap(pr_after - audio_features, opts={"title": "pr_after-audio"})
    # vis.heatmap(pianoroll - audio_features, opts={"title": "pianoroll - audio"})
    # vis.heatmap(pianoroll - pr_after, opts={"title": "pianoroll - pr_after"})
    # __import__('ipdb').set_trace()

    new_ons += (start / sr)
    new_offs += (start / sr)

    return new_ons, new_offs


def match_pianorolls(pr1, pr2, max_dist, row_cost, lookup_range):
    """
    Maybe in future... something like object tracking

    Arguments
    ---------

    `pr1` : np.ndarray
        pianoroll 1

    `pr2` : np.ndarray
        pianoroll 2

    `max_dist` : int
        maximum distance for looking for a match (real maximum distance will be
        max_dist + lookup_range)

    `row_cost` : int
        the cost with wich the distance will be divided to decide if max_dist
        is reached along the row dimension

    `lookup_range` : int
        the maximum number of columns in pr2 that all together can create a
        correspondance to a single column in pr1

    """

    for col1 in range(pr1.shape[1]):
        if np.any(pr1[:, col1]):
            # look for the nearest cols in pr2 which is equal to pr1
            # previous match must be taken into account
            for i in range(max_dist):
                for col2 in [col1 + i, col1 - 1]:
                    diff = pr1[:, col1] - pr2[col2:col2 + lookup_range]
                    if not np.any(diff):
                        # this is the match!
                        pass
                        break
            for i in range(max_dist):
                for col2 in [col1 + i, col1 - 1]:
                    if np.all(
                            np.where(diff == 1)[0] -
                            np.where(diff == -1)[0] <= max_dist / row_cost):
                        # this is the match!
                        pass
                        break
            # chose which match is the best

    # return a map or similar
