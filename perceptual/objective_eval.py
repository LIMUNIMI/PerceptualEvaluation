from .utils import midipath2mat, midi_pitch_to_f0
from asmd.audioscoredataset import Dataset
import mir_eval
import numpy as np
import pickle
from . import proposed, magenta_transcription
from .make_template import TEMPLATE_PATH, SR
import os

DATASET = "SMD"
N_JOBS = 8
EXCERPTS_DIR = "to_be_synthesized"

EXCERPTS = {'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3, 'medoid': 4}
METHODS = {
    # hr
    'orig': 0,
    # nr
    'other': 1,
    # si
    'proposed': 3,
    # o&f
    'magenta': 2
}


def compare_midi(fname_targ, fname_pred, stdout=True):
    mp = mat2mir_eval(midipath2mat(fname_pred))
    mt = mat2mir_eval(midipath2mat(fname_targ))

    res = evaluate(mt, mp)
    if stdout:
        print("Hello bro, values are in this order:")
        print("  [w/o velocity: ['Precision', 'Recall', 'Fmeasure']")
        print("   w/ velocity:  ['Precision', 'Recall', 'Fmeasure']]")
        print(res)
        return res
    else:
        return res


def evaluate(targ, pred):
    targ = mat2mir_eval(targ)
    pred = mat2mir_eval(pred)
    t1, p1, v1 = targ
    t2, p2, v2 = pred
    # remove initial silence
    t1 -= np.min(t1)
    t2 -= np.min(t2)

    out = np.empty((2, 3))
    evaluation = mir_eval.transcription.evaluate(t1, p1, t2, p2)
    out[0, 0] = evaluation['Precision']
    out[0, 1] = evaluation['Recall']
    out[0, 2] = evaluation['F-measure']

    try:
        evaluation = mir_eval.transcription_velocity.evaluate(t1,
                                                              p1,
                                                              v1,
                                                              t2,
                                                              p2,
                                                              v2,
                                                              rcond=None)
        out[1, 0] = evaluation['Precision']
        out[1, 1] = evaluation['Recall']
        out[1, 2] = evaluation['F-measure']
    except Exception:
        out[1] = [0, 0, 0]
    return out


def mat2mir_eval(mat):
    # sorting according to pitches and then onsets
    mat = mat[np.lexsort((mat[:, 1], mat[:, 0]))]
    times = mat[:, (1, 2)]
    pitches = midi_pitch_to_f0(mat[:, 0])
    vel = mat[:, 3]
    return times, pitches, vel


def excerpts_test(path=EXCERPTS_DIR, ordinal=False, evaluate=evaluate):
    # load all midi files in excerpts
    excerpts = {}
    print("Hello bro, values are in this order:")
    print("  [w/o velocity: ['Precision', 'Recall', 'Fmeasure']")
    print("   w/ velocity:  ['Precision', 'Recall', 'Fmeasure']]")
    for root, dirs, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext not in ['.mid', '.midi']:
                continue
            try:
                q, n, t = name.split('_')
            except:
                print(f"skipping {file}")
                continue
            if n not in excerpts:
                excerpts[n] = {t: os.path.join(root, file)}
            else:
                excerpts[n][t] = os.path.join(root, file)

    # compare midis with the one ending with type 'orig'
    out = np.zeros((len(excerpts.keys()), len(excerpts[n].keys()), 2, 3))
    for n in excerpts.keys():
        target = excerpts[n]['orig']
        target = midipath2mat(target)
        for t in excerpts[n].keys():
            # store results based on name
            exc = midipath2mat(excerpts[n][t])
            out[EXCERPTS[n], METHODS[t]] = evaluate(target, exc)
            print(f"name: {n}, type: {t}")
            print(out[EXCERPTS[n], METHODS[t]])

    # change values if asked ordinal
    if ordinal:
        for excerpt in range(out.shape[0]):
            for measure_i in range(out.shape[2]):
                for measure_j in range(out.shape[3]):
                    values = out[excerpt, :, measure_i, measure_j]
                    order = np.argsort(values)
                    values[order] = list(range(out.shape[1]))
    return out


def dataset_test(dataset):
    dataset = Dataset().filter(datasets=[dataset])
    out = dataset.parallel(process, n_jobs=N_JOBS)
    out = np.stack(out)

    # printing results
    print("\nHello bro, values are in this order (averages):")
    print("  [w/o velocity: ['Precision', 'Recall', 'Fmeasure']")
    print("   w/ velocity:  ['Precision', 'Recall', 'Fmeasure']]")
    np.mean(out)

    print("\nHello bro, values are in this order (std):")
    print("  [w/o velocity: ['Precision', 'Recall', 'Fmeasure']")
    print("   w/ velocity:  ['Precision', 'Recall', 'Fmeasure']]")
    np.std(out)

    # saving results
    if not os.path.exists('results'):
        os.mkdir('results')
    pickle.dump(out, open('results/transcription_' + DATASET + '.pkl', 'wb'))


def process(i, dataset):
    audio, sr = dataset.get_audio(i)
    score = dataset.get_score(i, score_type=['non_aligned'])

    # transcribe
    data = pickle.load(open(TEMPLATE_PATH, 'rb'))
    transcription_0, _, _, _ = proposed.transcribe(audio, data, score=score)

    transcription_1 = magenta_transcription.transcribe(audio, SR)

    transcription_2, _, _, _ = proposed.transcribe(audio, data, score=None)

    # evaluating
    gt = dataset.get_score(i,
                           score_type=['precise_alignment', 'broad_alignment'])

    res0 = evaluate(gt, transcription_0)
    res1 = evaluate(gt, transcription_1)
    res2 = evaluate(gt, transcription_2)

    return np.stack([res0, res1, res2])


if __name__ == "__main__":
    import sys

    def show_usage():
        print("Usage: " + sys.argv[0] + " [reference] [est]")
        print("        to compare two midi files")
        print("Usage: " + sys.argv[0] + " " + DATASET)
        print(f"        to test and evaluate over the {DATASET} dataset")
        print(f"Usage: {sys.argv[0]} {EXCERPTS_DIR}")
        print("        to evaluate excerpts extracted")

    if len(sys.argv) == 3:
        try:
            compare_midi(sys.argv[1], sys.argv[2])
        except Exception as e:
            print(e)
            show_usage()
    elif len(sys.argv) == 2:
        if sys.argv[1] == DATASET:
            dataset_test(DATASET)
        elif sys.argv[1] == EXCERPTS_DIR:
            excerpts_test(EXCERPTS_DIR)
        else:
            show_usage()
    else:
        show_usage()
