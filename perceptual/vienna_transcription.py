from madmom.processors import SequentialProcessor
from madmom.features.notes import ADSRNoteTrackingProcessor, _cnn_pad
import essentia.standard as esst
import numpy as np
from .utils import mat2midipath

VIENNA_MODEL_PATH = ['vienna_model.pkl']


class ViennaMaestroModel(SequentialProcessor):
    """
    This is stolen from madmom, but we are using another model...

    """

    def __init__(self, sr=44100, **kwargs):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)
        from madmom.ml.nn import NeuralNetworkEnsemble
        sr_ratio = 44100 / sr
        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=sr)
        frames = FramedSignalProcessor(frame_size=4096 // sr_ratio,
                                       fps=50 // sr_ratio)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)
        spec = LogarithmicSpectrogramProcessor(add=1)
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor(
            (sig, frames, stft, filt, spec, _cnn_pad))
        # process the pre-processed signal with a NN
        nn = NeuralNetworkEnsemble.load(VIENNA_MODEL_PATH)
        # instantiate a SequentialProcessor
        super().__init__((pre_processor, nn))

        self.adsr = ADSRMaestro()

    def get_mat(self, data):
        """
        Process audio in `fname`:
            * create activations
            * analyze activations with `ADSRMaestro`
            * returns array with columns (seconds, pitch, duration)
        """
        return self.adsr(self.process(data))

    def get_pr(self, data):
        """
        Process audio in `fname`:
            * return activations
        """
        return self.process(data)


class ADSRMaestro(ADSRNoteTrackingProcessor):
    """
    Just a wrapper with new parameters
    """

    def __init__(self, onset_note_prob=0.9, offset_prob=0.7, threshold=0.5):
        super().__init__(onset_prob=onset_note_prob,
                         note_prob=onset_note_prob,
                         offset_prob=offset_prob,
                         attack_length=0.04,
                         decay_length=0.04,
                         release_length=0.02,
                         complete=True,
                         onset_threshold=threshold,
                         note_threshold=threshold,
                         fps=50,
                         pitch_offset=21)

    def process(self, activations, **kwargs):
        return super().process(activations, clip=1e-2)


def transcribe(audio, sr=44100, cuda=False):
    """
    Transcribe an audio array and returns the mat like asmd (no velocities!)

    `cuda` has no effect at now
    """
    processor = ViennaMaestroModel(sr=sr)
    mat = processor.get_mat(audio)
    out = np.empty((mat.shape[0], 4))
    out[:, 0] = mat[:, 1]
    out[:, 1] = mat[:, 0]
    out[:, 2] = mat[:, 0] + mat[:, 2]
    out[:, 3] = -255

    return out


def transcribe_from_paths(audio_path, topath, sr=22050, cuda=False):
    """
    `cuda` has no effect at now
    """
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
