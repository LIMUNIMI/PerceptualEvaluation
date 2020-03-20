import numpy as np
import essentia.standard as es
from .utils import farthest_points, midi_pitch_to_f0
# TODO: add asmd and change this
from .asmd.audioscoredataset import Dataset

RES = 0.01
SR = 22050
DURATION = 10  # in seconds
audio_win_len = int(DURATION * SR)
score_win_len = int(DURATION / RES)


def main():
    dataset = Dataset()
    dataset.filter(
        instruments=['piano'],
        datasets=['vienna_corpus'])

    samples = []
    for i in range(len(dataset)):
        score = dataset.get_pianoroll(
            i, score_type=['precise_alignment', 'broad_alignment'],
            resolution=RES)
        audio, sr = dataset.get_audio(i)

        resampler = es.Resample(inputSampleRate=sr, outputSampleRate=SR)
        audio = resampler(audio)
        samples += get_win_features(score, audio)


def get_win_features(score, audio):

    # looking for start and end in midi
    for i in range(score.shape[1]):
        if np.any(score[:, i]):
            break
    score = score[:, i:]
    audio = audio[int(i*RES*SR):]

    for i in reversed(range(score.shape[1])):
        if np.any(score[:, i]):
            break
    score = score[:, :i+1]
    audio = audio[:int(i*RES*SR)+1]

    num_win = (len(audio) / SR) / DURATION
    features = []
    for i in range(int(num_win)):
        audio_win = audio[i*audio_win_len:(i+1)*audio_win_len]
        score_win = score[:, i*score_win_len:(i+1)*score_win_len]
        features.append(
            score_features(score_win) + audio_features(audio_win)
        )
    return features


def audio_features(audio_win):
    return []


def score_features(score_win):
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
    A_ext = np.diff(np.hstack(([[0]]*128, score_win, [[0]]*128)))
    # Find interval of non-zeros lengths
    duration_win = np.where(A_ext == -1)[1] - np.where(A_ext == 1)[1]
    avg_dur = np.mean(duration_win)
    std_dur = np.std(duration_win)

    return [avg_pitch, std_pitch, avg_vel, std_vel, avg_vert,
            std_vert, avg_interv, std_interv, avg_dur, std_dur]


if __name__ == "__main__":
    main()
