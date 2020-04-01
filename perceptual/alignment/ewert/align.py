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
        f"fluidsynth -ni Arachno\\ SoundFont\\ -\\ Version\\ 1.0.sf2 {fname}.mid -F {fname}.wav -r {sr} > /dev/null 2>&1"
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


# def main():
#     fn1 = "./examples/KissTheRain_2_t_short.wav"
#     fn2 = "./examples/KissTheRain_2_s_short.wav"
#     audio1, sr1 = librosa.load(fn1)
#     audio2, sr2 = librosa.load(fn2)

#     nfft = 4096
#     hopsize = int(nfft / 4)
#     wp = multiple_audio_alignment(audio1, sr1, audio2, sr2, hopsize=hopsize, n_fft=nfft)
#     wp_s = np.asarray(wp) * hopsize / sr1 # (330, 2) wrapping path

#     # wav length in time
#     len_audio1 = len(audio1)/sr1
#     len_audio2 = len(audio2)/sr2

#     # print(wp_s[:, 1])
#     # print(wp_s[:, 0])

#     # fig = plt.figure(figsize=(5, 5))
#     # ax = fig.add_subplot(111)
#     # librosa.display.specshow(D, x_axis='time', y_axis='time',
#     #                          cmap='gray_r', hop_length=hopsize)
#     # imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
#     #                  origin='lower', interpolation='nearest', aspect='auto')
#     # ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
#     # plt.title('Warping Path on Acc. Cost Matrix $D$')
#     # plt.colorbar()
#     # plt.tight_layout()
#     # plt.show()

#     # load score of sonic visualizer
#     score_t = sv_score_parser("./examples/KissTheRain_2_t_short.txt")
#     score_s = sv_score_parser("./examples/KissTheRain_2_s_short.txt")

#     # use wrapping path to align the notes
#     # step1: sort the overlapping note area from high to low
#     # step2: abs(note_t - note_s) <= 2
#     wp_stu = wp_s[:, 1][::-1]
#     wp_t = wp_s[:, 0][::-1]

#     list_score_aligned = []
#     non_aligned_score_s = []
#     for note_s in score_s: # search teacher's corresponding notes using student's notes
#         found_s = False # bool student's note found or not
#         idx_note_start = np.argmin(np.abs(wp_stu - note_s[0]))
#         idx_note_end = np.argmin(np.abs(wp_stu - (note_s[0] + note_s[2])))
#         time_note_start_t = wp_t[idx_note_start] # note time start corresponding in teacher's wrapping time
#         time_note_end_t = wp_t[idx_note_end] # note time end

#         list_dur_pitch_t = [] # calculate overlapping area and note difference
#         for ii_note_t, note_t in enumerate(score_t):
#             i0 = max(time_note_start_t, note_t[0])
#             i1 = min(time_note_end_t, note_t[0] + note_t[2])
#             diff_dur = i1-i0 if i0 < i1 else 0.0
#             diff_pitch = np.abs(note_t[1] - note_s[1])
#             list_dur_pitch_t.append([diff_dur, diff_pitch, ii_note_t])
#         list_dur_pitch_t = sorted(list_dur_pitch_t, key=itemgetter(0), reverse=True)

#         for ldp_t in list_dur_pitch_t:
#             if ldp_t[0] > 0 and ldp_t[1] <=2: # find the most overlapped note and pitch diff <= 2
#                 list_score_aligned.append([score_t[ldp_t[2]], note_s])
#                 # print(len(score_t))
#                 score_t.pop(ldp_t[2])
#                 # print(len(score_t))
#                 found_s = True
#                 break

#         if not found_s:
#             non_aligned_score_s.append(note_s)

#     if len(score_t):
#         for st in score_t:
#             list_score_aligned.append([st, []])

#     if len(non_aligned_score_s):
#         for ss in non_aligned_score_s:
#             list_score_aligned.append([[], ss])

#     # plot the alignment, red aligned notes, black extra or missing notes
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     for note_pair in list_score_aligned:
#         if len(note_pair[0]) and len(note_pair[1]):
#             face_color = 'r'
#         elif len(note_pair[0]):
#             face_color = 'k'
#         else:
#             continue
#         rect = patches.Rectangle((note_pair[0][0], note_pair[0][1]-0.5), note_pair[0][2], 1.0, linewidth=1,edgecolor=face_color,facecolor=face_color)
#         ax1.add_patch(rect)

#     ax1.set_ylabel('Teacher')
#     ax1.set_xlim(0, len_audio1)
#     ax1.set_ylim(0, 88)

#     for note_pair in list_score_aligned:
#         if len(note_pair[0]) and len(note_pair[1]):
#             face_color = 'r'
#             con = ConnectionPatch(xyA=(note_pair[1][0], note_pair[1][1]-0.5), xyB=(note_pair[0][0], note_pair[0][1]-0.5), coordsA="data", coordsB="data",
#                                   axesA=ax2, axesB=ax1, color="b")
#             ax2.add_artist(con)
#         elif len(note_pair[1]):
#             face_color = 'k'
#         else:
#             continue
#         rect = patches.Rectangle((note_pair[1][0], note_pair[1][1]-0.5), note_pair[1][2], 1.0, linewidth=1,edgecolor=face_color,facecolor=face_color)
#         ax2.add_patch(rect)

#     ax2.set_ylabel('Student')
#     ax2.set_xlim(0, len_audio2)
#     ax2.set_ylim(0, 88)
#     ax2.set_xlabel('time (s)')

#     # plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()
