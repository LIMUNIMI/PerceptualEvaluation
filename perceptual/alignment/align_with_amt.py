import sys
import tensorflow as tf
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
from magenta.music.protobuf import music_pb2
import os
import logging
import librosa
import numpy as np

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Define model and load checkpoint
# Only needs to be run once.
CHECKPOINT_DIR = './perceptual/magenta/train/'


def transcription(wav_data, res):
    """
    Google sucks and want to use raw wav instead of decoded samples loosing in
    decoupling...
    """

    # almost everything here is copied from
    # https://colab.research.google.com/notebooks/magenta/onsets_frames_transcription/onsets_frames_transcription.ipynb
    config = configs.CONFIG_MAP['onsets_frames']
    hparams = config.hparams
    hparams.use_cudnn = False
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
        audio_label_data_utils.process_record(wav_data=wav_data,
                                              ns=music_pb2.NoteSequence(),
                                              example_id="0",
                                              min_length=0,
                                              max_length=-1,
                                              allow_empty_notesequence=True))
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

    input_fn = infer_util.labels_to_features_wrapper(transcription_data)

    prediction_list = list(
        estimator.predict(input_fn, yield_single_examples=False))

    assert len(prediction_list) == 1

    frame_predictions = prediction_list[0]['frame_predictions'][0]
    # convert to pianoroll with resolurion `res`
    pianoroll = None
    return pianoroll


def audio_to_score_alignment(misaligned, pianoroll, path, res):
    wav_data = tf.gfile.Open(sys.argv[1], 'rb').read()
    audio_features = transcription(wav_data, res)

    # parameters for dtw were chosen with midi2midi on musicnet (see stw_test
    # in biseqnet)
    _D, path = librosa.sequence.dtw(X=pianoroll,
                                    Y=audio_features,
                                    metric='cosine',
                                    global_constraints=True,
                                    band_rad=0.66)
    path = path[::-1] * res

    # interpolating
    new_ons = np.interp(misaligned[:, 1], path[:, 0], path[:, 1])
    new_offs = np.interp(misaligned[:, 2], path[:, 0], path[:, 1])

    return new_ons, new_offs


if __name__ == "__main__":
    wav_data = tf.gfile.Open(sys.argv[1], 'rb').read()
    transcription(wav_data)
