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
import essentia.standard as esst
import numpy as np
from .utils import mat2midipath

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Define model and load checkpoint
# Only needs to be run once.
CHECKPOINT_DIR = './perceptual/magenta/train/'


def google_sucks(samples, sr):
    """
    Google sucks and want to use audio path (raw wav) instead of decoded
    samples loosing in decoupling between file format and DSP.
    This hack overwrites their stupid loader which writed data to a temprorary
    file and reopen it
    """

    return samples


def transcribe(audio, sr, cuda=False):
    """
    Google sucks and want to use audio path (raw wav) instead of decoded
    samples loosing in decoupling between file format and DSP

    input audio and sample rate, output mat like asmd with (pitch, ons, offs, velocity)
    """

    # simple hack because google sucks... in this way we can accept audio data
    # already loaded and keep our reasonable interface (and decouple i/o
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

    notes = music_pb2.NoteSequence.FromString(
        prediction_list[0]['sequence_predictions'][0]).notes

    out = np.empty((len(notes), 4))
    for i, note in enumerate(notes):
        out[i] = [note.pitch, note.start_time, note.end_time, note.velocity]
    return out


def transcribe_from_paths(audio_path, topath, sr=44100, cuda=False):
    audio = esst.EasyLoader(filename=audio_path, sampleRate=sr)()
    mat = transcribe(audio, sr, cuda=cuda)
    mat2midipath(mat, topath)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [input audio file] [output midi file]")

    cuda = False
    if "--cuda" in sys.argv:
        cuda = True

    transcribe_from_paths(sys.argv[1], sys.argv[2], sr=44100, cuda=cuda)
