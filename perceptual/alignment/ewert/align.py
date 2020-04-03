import numpy as np
import pretty_midi
from math import floor, log2
import librosa
import librosa.display
from .dlnco.DLNCO import dlnco
import os
import essentia.standard as esst


def multiple_audio_alignment(audio1,
                             sr1,
                             audio2,
                             sr2,
                             hopsize,
                             n_fft=4096,
                             merge_dlnco=True):
    """
    Aligns two audio files and returns a list of lists containing the map
    between the audio frames.

    Parameters
    ----------

    audio1 : np.array
        Numpy array representing the signal.
    sr1 : int
        Sampling rate of **audio1**
    audio2 : np.array
        Numpy array representing the signal.
    sr2 : int
        Sampling rate of **audio2**
    hopsize : int
        The hopsize for the FFT. Consider to use something like `n_fft/4`
    n_fft : int
        The length of the FFT. Consider to use something like `4*hopsize`
    merge_dlnco : bool
        Unknown


    Returns
    -------
     numpy.ndarray
        A 2d array, mapping frames from :attr: `audio1` to frames in
        :attr: `audio2`. `[[frame in audio 1, frame in audio 2]]`

    """

    # chroma and DLNCO features
    # print("Computing features")
    audio1_chroma = librosa.feature.chroma_stft(y=audio1,
                                                sr=sr1,
                                                tuning=0,
                                                norm=2,
                                                hop_length=hopsize,
                                                n_fft=n_fft)
    audio1_dlnco = dlnco(audio1, sr1, n_fft, hopsize)

    audio2_chroma = librosa.feature.chroma_stft(y=audio2,
                                                sr=sr2,
                                                tuning=0,
                                                norm=2,
                                                hop_length=hopsize,
                                                n_fft=n_fft)
    audio2_dlnco = dlnco(audio2, sr2, n_fft, hopsize)

    audio1_merge = audio1_chroma + audio1_dlnco if merge_dlnco else audio1_chroma
    audio2_merge = audio2_chroma + audio2_dlnco if merge_dlnco else audio2_chroma

    # print("Starting DTW")
    D, wp = librosa.sequence.dtw(X=audio1_merge,
                                 Y=audio2_merge,
                                 metric='cosine')  # D (153, 307)
    del D
    return wp


def audio_to_midi_alignment(midi,
                            audio,
                            sr,
                            hopsize,
                            n_fft=4096,
                            merge_dlnco=True,
                            sf2_path=None):
    """
    Synthesize midi file, align it to :attr: `audio` and returns a mapping
    between midi times and audio times.

    Parameters
    ----------

    midi : :class: `pretty_midi.PrettyMIDI`
        The midi file that will be aligned
    audio : np.array
        Numpy array representing the signal.
    sr : int
        Sampling rate of **audio1**
    hopsize : int
        The hopsize for the FFT. Consider to use something like `n_fft/4`
    n_fft : int
        The length of the FFT. Consider to use something like `4*hopsize`
    merge_dlnco : bool
        Unknown
    sf2_path : string
        The path to a sf2 file. If `None`, then `TimGM6mb.sf2` is used.


    Returns
    -------
    numpy.ndarray
        A 2d array, mapping time from :attr: `midi` to times in
        :attr: `audio`. `[[midi time, audio time]]`

    """
    # print("Synthesizing MIDI")
    fname = str(os.getpid())
    midi.write(fname+".mid")
    os.system(
        f"fluidsynth -ni SalamanderGrandPianoV3Retuned-renormalized.sf2 {fname}.mid -F {fname}.wav -r {sr} > /dev/null 2>&1"
    )
    audio1 = esst.EasyLoader(filename=fname+".wav", sampleRate=sr)()
    os.remove(fname+".mid")
    os.remove(fname+".wav")

    audio2 = audio
    # print("Starting alignment")
    wp = multiple_audio_alignment(audio1, sr, audio2, sr, hopsize, n_fft=n_fft)
    wp_times = np.asarray(wp) * hopsize / sr  # (330, 2) wrapping path
    # print("Finished alignment")

    return wp_times[::-1]


def audio_to_score_alignment(mat,
                             audio,
                             sr,
                             res=0.02,
                             n_fft=4096,
                             merge_dlnco=True,
                             sf2_path=None):
    """
    Synthesize a matrix of notes, align it to :attr: `audio` and returns a mapping
    between midi times and audio times.

    Parameters
    ----------

    mat : numpy.ndarray
        A matrix representing notes from which a midi file is contructed. Each
        row is a note and columns are: pitches, onsets, offsets, *something*,
        program, *anything else*.
    audio : np.array
        Numpy array representing the signal.
    sr : int
        Sampling rate of **audio1**
    time_precision : float
        The width of each column of the spectrogram. This will define the
        hopsize as 2**floor(log2(sr*time_precision)).
    n_fft : int
        The length of the FFT. Consider to use something like `4*hopsize`
    merge_dlnco : bool
        Unknown
    sf2_path : string
        The path to a sf2 file. If `None`, then `TimGM6mb.sf2` is used.


    Returns
    -------
    list :
        a list of floats representing the new computed onsets
    list :
        a list of floats representing the new computed offsets

    """
    # creating one track per each different program
    programs = np.unique(mat[:, 4])
    tracks = {}
    is_drum = False
    for program in programs:
        if program == 128:
            program = 0
            is_drum = True
        tracks[program] = pretty_midi.Instrument(program=int(program),
                                                 is_drum=is_drum)

    # creating pretty_midi.PrettyMIDI object and inserting notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments = tracks.values()
    for row in mat:
        program = row[4]
        if program == 128:
            program = 0
        tracks[program].notes.append(
            pretty_midi.Note(100, int(row[0]), float(row[1]), float(row[2])))

    # aligning midi to audio
    hopsize = 2**floor(log2(sr * res))
    mapping_times = audio_to_midi_alignment(midi, audio, sr, hopsize, n_fft,
                                            merge_dlnco, sf2_path)

    # interpolating
    new_ons = np.interp(mat[:, 1], mapping_times[:, 0], mapping_times[:, 1])
    new_offs = np.interp(mat[:, 2], mapping_times[:, 0], mapping_times[:, 1])

    return new_ons, new_offs
