from madmom.processors import SequentialProcessor, Processor
from madmom.features.notes import ADSRNoteTrackingProcessor

VIENNA_MODEL_PATH = ''


class ViennaMaestroModel(SequentialProcessor):
    """
    This is stolen from madmom, but we are using another model...

    """

    def __init__(self, **kwargs):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                         LogarithmicSpectrogramProcessor)
        from madmom.ml.nn import NeuralNetworkEnsemble
        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=4096, fps=50)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)
        spec = LogarithmicSpectrogramProcessor(add=1)
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor(
            (sig, frames, stft, filt, spec, _cnn_pad))
        # process the pre-processed signal with a NN
        nn = NeuralNetworkEnsemble.load(VIENNA_MODEL_PATH)
        # instantiate a SequentialProcessor
        super(CNNPianoNoteProcessor, self).__init__((pre_processor, nn))


class ADSRMaestro(ADSRNoteTrackingProcessor):
    """
    Just a wrapper with new parameters
    """

    def __init__(self):
        super().__init__(#something)
        )
