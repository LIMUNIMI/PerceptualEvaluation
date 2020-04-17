import numpy as np
import fastdtw
from .alignment import cdist
import essentia.standard as esst
from .utils import farthest_points, find_start_stop, midipath2mat, mat2midipath
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from asmd.audioscoredataset import Dataset
from . import proposed, magenta_transcription
from .make_template import TEMPLATE_PATH, SR
import os
import pickle
from . import utils

#: duration of each pianoroll column in seconds
RES = 0.005
#: duration of excerpts in seconds
DURATION = 20
#: percentage of hop-size for excerpt search
HOP = 0.5
#: number of parallele processes
NJOBS = -1
#: the number of excerpts per each question (without the medoid)
NUM_EXCERPTS = 4
#: the number of questions
QUESTIONS = 1
#: sample rate for the output excerpts
OUT_SR = 44100
#: fade in and fade out duration in seconds
FADE = 0.5
#: file format for the excerpts: essentia still has some problem in writing
#: mp3 in the python version
FORMAT = 'flac'

audio_win_len = int(DURATION * SR)
hop_audio = int(audio_win_len * HOP)
score_win_len = int(DURATION / RES)
hop_score = int(score_win_len * HOP)


def main():
    dataset = Dataset()
    dataset.filter(instruments=['piano'], datasets=['vienna_corpus'])

    print("\nAnalysis:")

    parallel_out = dataset.parallel(process_song, n_jobs=NJOBS)
    songs = []
    wins = []
    samples = []
    positions = []
    for i in range(len(parallel_out)):
        samples += parallel_out[i][0]
        positions += parallel_out[i][1]
        wins += list(range(len(parallel_out[i][0])))
        songs += [i] * len(parallel_out[i][0])

    samples = np.array(samples)
    samples = StandardScaler().fit_transform(samples)
    samples = PCA(n_components=15).fit_transform(samples)
    points = farthest_points(samples, NUM_EXCERPTS, QUESTIONS)
    print("\nChosen songs:")
    for question in range(QUESTIONS):
        print(f"\nQuestion {question+1}:")
        for j, point in enumerate(points[:, question]):
            path = dataset.paths[songs[point]][0]
            time = positions[point]
            print(f"Song {path}, seconds audio \
{time[0][0]:.2f} - {time[0][1]:.2f} ...... midi \
{time[1][0]:.2f} - {time[1][1]:.2f}")
            path = os.path.join(dataset.install_dir, path[0])
            create_excerpt(path, time, f'q{question}_n{j}_')

    distmat = squareform(pdist(samples))
    medoid = np.argmin(np.sum(distmat, axis=1))
    path = dataset.paths[songs[medoid]][0]
    time = positions[medoid]
    print(f"The medoid of the whole set is: {path}, seconds audio \
{time[0][0]:.2f} - {time[0][1]:.2f} ...... midi \
{time[1][0]:.2f} - {time[1][1]:.2f}")
    path = os.path.join(dataset.install_dir, path[0])
    create_excerpt(path, time, f'q{question}_medoid_')
    print(f"Total number of samples: {samples.shape[0]}")


def _my_prep_inputs(x, y, dist):
    """
    Fastdtw sucks too and convicts you to use float64...
    """
    return x, y


def remap_original_in_other(original, other, timings):
    """
    perform fasdtw between original and other and returns timings mapped
    """
    EPS = 1e-15
    pr_original = utils.make_pianoroll(
        original, res=RES, velocities=False, only_onsets=True) + EPS
    pr_original += utils.make_pianoroll(original, res=RES,
                                        velocities=False) + EPS
    pr_other = utils.make_pianoroll(
        other, res=RES, velocities=False, only_onsets=True) + EPS
    pr_other += utils.make_pianoroll(other, res=RES, velocities=False) + EPS

    # hack to let fastdtw accept float32
    fastdtw._fastdtw.__prep_inputs = _my_prep_inputs
    _D, path = fastdtw.fastdtw(pr_original.astype(np.float32).T,
                               pr_other.astype(np.float32).T,
                               dist=cdist.cosine,
                               radius=1)

    # converting indices to seconds
    path = np.array(path) * RES

    # interpolating
    return np.interp(timings, path[:, 0], path[:, 1])


def create_excerpt(audio_path, time, name):
    """
    Given audio path and times, transcribes it and creates new midis and wav
    files for the given excerpts. `name` is the file name without extension and
    transcription number.
    """

    full_audio = esst.EasyLoader(filename=audio_path, sampleRate=SR)()
    start_audio, _ = find_start_stop(full_audio, sample_rate=SR, seconds=True)
    original = midipath2mat(audio_path[:-4] + '.mid')

    # compute score path
    score_path = './my_scores/' + os.path.basename(audio_path)[:-8] + '.mid'
    score = midipath2mat(score_path)

    # transcribe
    data = pickle.load(open(TEMPLATE_PATH, 'rb'))
    # transcription_0, _, _, _ = proposed.transcribe(
    #     full_audio,
    #     data,
    #     score=score)

    transcription_1 = magenta_transcription.transcribe(full_audio, SR)

    # transcription_2, _, _, _ = proposed.transcribe(
    #     full_audio,
    #     data,
    #     score=None)

    # chose another interpretation
    performance = '01'
    if audio_path[-6:-4] == '01':
        performance = '02'
    other = midipath2mat(audio_path[:-6] + performance + '.mid')

    # segment all the scores and audios
    full_audio = esst.EasyLoader(filename=audio_path, sampleRate=OUT_SR)()
    original_audio = full_audio[round(time[0][0] * OUT_SR):round(time[0][1] *
                                                                 OUT_SR)]
    other_time = remap_original_in_other(original, other, time[0])
    original = segment_mat(original, time[0][0], time[0][1], start_audio)
    other = segment_mat(other, other_time[0], other_time[1], start_audio)
    # transcription_0 = segment_mat(transcription_0, time[0][0], time[0][1],
    #                               start_audio)
    transcription_1 = segment_mat(transcription_1, time[0][0], time[0][1],
                                  start_audio)
    # transcription_2 = segment_mat(transcription_2, time[0][0], time[0][1],
    #                               start_audio)

    # write scores to `to_be_synthesized` and audios to `excerpts`
    if not os.path.exists('to_be_synthesized'):
        os.mkdir('to_be_synthesized')
    midi_path = os.path.join('to_be_synthesized', name)
    mat2midipath(original, midi_path + 'orig.mid')
    mat2midipath(other, midi_path + 'other.mid')
    # mat2midipath(transcription_0, midi_path + 'proposed.mid')
    mat2midipath(transcription_1, midi_path + 'magenta.mid')
    # mat2midipath(transcription_2, midi_path + 'vienna.mid')

    if not os.path.exists('audio'):
        os.mkdir('audio')
    audio_path = os.path.join('audio', name)

    # apply fade in and fade out (maybe this is better to be performed at
    # questionnaire creation stage
    # fade_len = int(FADE * OUT_SR)
    # fade_array = np.arange(0, 1, 1 / fade_len)
    # original_audio[:fade_len] *= fade_array
    # original_audio[-fade_len:] *= fade_array[::-1]

    # write audio
    esst.MonoWriter(filename=audio_path + 'target.' + FORMAT,
                    sampleRate=OUT_SR,
                    format=FORMAT,
                    bitrate=256)(original_audio)


def segment_mat(mat, start, end, start_audio=0):
    """
    returns a new mat (list of lists) with only notes included between start
    and end (included)

    if `start_audio` is the reference for audio time shift
    (use it to provide `start` and `end` referred to audio)
    """

    # realign mat with audio
    mat[:, (1, 2)] = mat[:, (1, 2)] - np.min(mat[:, (1, 2)]) + start_audio

    # filter notes included in [start, end]
    mat = np.array(
        [note for note in mat if note[1] >= start and note[2] <= end])

    # make the notes start at 0
    mat[:, (1, 2)] -= np.min(mat[:, (1, 2)])

    return mat


def process_song(i, dataset):
    """
    Returns features for all windows of a given song in the dataset
    (to be run in parallel)
    """
    score = dataset.get_pianoroll(
        i, score_type=['precise_alignment', 'broad_alignment'], resolution=RES)
    audio, sr = dataset.get_audio(i)

    audio = esst.Resample(inputSampleRate=sr, outputSampleRate=SR)(audio)
    return get_song_win_features(score, audio)


def get_song_win_features(score, audio):
    """
    returns all features of all windows for a song
    """

    # looking for start and end in audio
    start, stop = find_start_stop(
        audio,
        sample_rate=SR,
    )
    audio = audio[start:stop]

    # looking for start and end in midi
    for i in range(score.shape[1]):
        if np.any(score[:, i]):
            break
    score_start = i
    score = score[:, i:]

    for i in reversed(range(score.shape[1])):
        if np.any(score[:, i]):
            break
    score = score[:, :i + 1]

    num_win = (audio.shape[0] - audio_win_len) // hop_audio
    num_win = min(num_win, (score.shape[1] - score_win_len) // hop_score)
    dur_win = audio_win_len / SR
    dur_hop = hop_audio / SR
    features = []
    times = []
    for i in range(int(num_win)):
        audio_win = audio[i * hop_audio:i * hop_audio + audio_win_len]
        score_win = score[:, i * hop_score:i * hop_score + score_win_len]
        features.append(score_features(score_win) + audio_features(audio_win))
        # TODO: something is wrong in timings!!
        times.append(
            ((start / SR + dur_hop * i, start / SR + dur_hop * i + dur_win),
             (score_start * RES + i * hop_score * RES,
              score_start * RES + i * hop_score * RES + score_win_len * RES)))

    return features, times


def audio_features(audio_win):
    """
    returns audio features for a win
    """
    spectrum = esst.Spectrum(size=audio_win.shape[0])(audio_win)
    _bands, mfcc = esst.MFCC(inputSize=spectrum.shape[0],
                             sampleRate=SR)(spectrum)

    rhythm = esst.RhythmDescriptors()(audio_win)
    return mfcc.tolist() + [rhythm[2]] + list(rhythm[5:11])


def score_features(score_win):
    """
    returns symbolic features for a win
    """
    pixels = np.nonzero(score_win)
    avg_pitch = np.mean(pixels[0])
    std_pitch = np.std(pixels[0])
    avg_vel = np.mean(score_win[pixels])
    std_vel = np.std(score_win[pixels])

    vertical_notes = np.bincount(pixels[1])
    avg_vert = np.mean(vertical_notes)
    std_vert = np.std(vertical_notes)

    interv = []
    for i, v in enumerate(vertical_notes):
        if v > 1:
            pitches = pixels[0][np.where(pixels[1] == i)[0]]
            pitches.sort()
            interv += (pitches[1:] - pitches[0]).tolist()

    avg_interv = np.mean(interv)
    std_interv = np.std(interv)

    score_win[pixels] = 1
    A_ext = np.diff(np.hstack(([[0]] * 128, score_win, [[0]] * 128)))
    # Find interval of non-zeros lengths
    duration_win = np.where(A_ext == -1)[1] - np.where(A_ext == 1)[1]
    avg_dur = np.mean(duration_win)
    std_dur = np.std(duration_win)

    return [
        avg_pitch, std_pitch, avg_vel, std_vel, avg_vert, std_vert, avg_interv,
        std_interv, avg_dur, std_dur
    ]


if __name__ == "__main__":
    main()
