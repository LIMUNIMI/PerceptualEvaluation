import os
import essentia.standard as esst
import numpy as np
import plotly.express as px
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV, LassoCV, BayesianRidge, ARDRegression, LassoLarsCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from copy import copy
from . import excerpt_search
from . import objective_eval
from . import subjective_eval
from . import utils

AUDIO_PATH = 'excerpts/'
MIDI_PATH = 'to_be_synthesized/'
SAVES_PATH = subjective_eval.PATH

SR = 22050


def _fill_out_targets(out, targets, features, splits):
    if splits[2] == 'target':
        targets[objective_eval.EXCERPTS[splits[1]]] = features
    elif splits[2] not in objective_eval.METHODS:
        return
    else:
        out[objective_eval.EXCERPTS[splits[1]]][objective_eval.METHODS[
            splits[2]]] = features


def load_audio_excerpts(path=AUDIO_PATH, num_features=9):
    """
    Extracts `num_features+1` MFCC coeffcients from each audio and discards the
    first coefficients (tied to energy).
    """

    targets = np.zeros((3, 5, num_features))
    out = np.zeros((3, 5, 4, num_features))
    for file in tqdm(os.listdir(path)):
        if file.endswith(excerpt_search.FORMAT):
            audio = esst.EasyLoader(filename=os.path.join(path, file),
                                    sampleRate=SR)()
            if audio.shape[0] % 2 == 1:
                audio = audio[:-1]
            spectrum = esst.Spectrum(size=audio.shape[0])(audio)
            _bands, features = esst.MFCC(inputSize=spectrum.shape[0],
                                         sampleRate=SR,
                                         numberCoefficients=num_features +
                                         1)(spectrum)
            splits = file.replace('.flac', '').split('_')
            question = int(splits[0][1])
            _fill_out_targets(out[question], targets[question], features[1:],
                              splits)
    return out - targets[..., np.newaxis, :]


def load_midi_scores(path=MIDI_PATH):
    num_features = 16
    targets = np.zeros((5, num_features))
    out = np.zeros((5, 4, num_features))
    for file in tqdm(os.listdir(path)):
        if file.endswith(('.mid', '.midi')):
            if not file.startswith('q0'):
                continue
            mat = utils.midipath2mat(os.path.join(path, file))
            midi = utils.make_pianoroll(mat, res=0.005)
            features = excerpt_search.score_features(midi)
            # features = features[2:6] + symbolic_bpms(mat)
            features += subjective_eval.symbolic_bpms(mat)
            splits = file.replace('.mid', '').split('_')
            _fill_out_targets(out, targets, features, splits)
    return out - targets[:, np.newaxis, :]


def leave_one_out(x, y, model):
    predictions = []
    for train_idx, test_idx in LeaveOneOut().split(x):
        model.fit(copy(x[train_idx]), y[train_idx])
        predictions.append(model.predict(x[test_idx]))

    return np.array(predictions)[:, 0]


def main():
    print("Loading audio features")
    mfcc = 13
    audios = load_audio_excerpts(num_features=mfcc)

    print("Loading symbolic features")
    midis = load_midi_scores()[np.newaxis]
    midis = np.broadcast_to(midis, (*audios.shape[:-1], midis.shape[-1]))
    samples = np.concatenate([audios, midis], axis=-1)
    old_shape = samples.shape
    samples = samples.reshape((-1, samples.shape[-1]))

    sub_eval = subjective_eval.sqlite2pandas(
        subjective_eval.xml2sqlite(SAVES_PATH),
        variable=None,
        min_listen_time=5,
        cursor_moved=True,
        ordinal=False)
    sub_eval = sub_eval.groupby(['question', 'excerpt_num', 'method']).median()
    sub_eval = sub_eval['rating'].values  # .reshape((*old_shape[:-1], -1))

    obj_eval = objective_eval.excerpts_test(ordinal=False)[..., 1, 2]
    obj_eval = np.broadcast_to(obj_eval[np.newaxis, :, :, np.newaxis],
                               (*old_shape[:-1], 1)).reshape(-1)

    peamt = subjective_eval.get_peamt()
    peamt_eval = objective_eval.excerpts_test(
        ordinal=False, evaluate=peamt.evaluate_from_midi)[..., 1, 2]
    peamt_eval = np.broadcast_to(peamt_eval[np.newaxis, :, :, np.newaxis],
                                 (*old_shape[:-1], 1)).reshape(-1)

    samples = np.concatenate([samples, obj_eval[:, None]], axis=-1)
    scaler = StandardScaler().fit(samples)
    samples = scaler.transform(samples)

    for model_type in [
            'BayesianRidge', 'ARDRegression', 'LassoLarsCV', 'ElasticNetCV',
            'LassoCV', 'RidgeCV', 'LinearRegression'
    ]:
        predictions = []
        for i in range(2):
            # sort = np.argsort(tasks[:, i])
            # task = tasks[sort, i]
            # samples = samples[sort]
            model = eval(model_type)()
            model.max_iter = 1e7
            if i == 1:
                # discarding audio
                predictions.append(
                    leave_one_out(samples[:, mfcc:], sub_eval, model))
            else:
                predictions.append(leave_one_out(samples, sub_eval, model))
            l1_err = np.mean(np.abs(predictions[i] - sub_eval))
            print(f"Mean error for {model_type}, task {i}: {l1_err:.2f}")

    ###############################################
    # Plotting coefficients:
    # fitting using audio features
    model = ElasticNetCV(max_iter=1e5)
    model.fit(copy(samples), sub_eval)
    px.bar(y=model.coef_, title="audio").show()

    # fitting without audio features
    model = ElasticNetCV(max_iter=1e5)
    model.fit(copy(samples[:, mfcc:]), sub_eval)
    px.bar(y=model.coef_, title="noaudio").show()
    features = [
        mfcc + i for i in range(len(model.coef_)) if abs(model.coef_[i]) > 0.1
    ]

    # fitting with the selected features only
    model = ElasticNetCV(max_iter=1e5)
    model.fit(copy(samples[:, features]), sub_eval)
    print(f"weights: {model.coef_}")
    print(f"intercept: {model.intercept_}")
    print(f"symbolic features: {[i - mfcc for i in features]}")
    print(f"scale: {scaler.scale_[features]}")
    print(f"mean: {scaler.mean_[features]}")

    #################################################
    # Comparison:
    # leave one out for comparison
    model = ElasticNetCV(max_iter=1e5)
    predictions = leave_one_out(copy(samples[:, features]), sub_eval, model)
    new_l1_err = np.mean(np.abs(predictions - sub_eval))
    obj_l1_err = np.mean(np.abs(obj_eval - sub_eval))
    peamt_l1_err = np.mean(np.abs(peamt_eval - sub_eval))
    print(f"Average error with selected features (new): {new_l1_err:.2f}")
    print(f"Average error with selected features (obj): {obj_l1_err:.2f}")
    print(f"Average error with selected features (peamt): {peamt_l1_err:.2f}")
    px.bar(y=model.coef_, title="noaudio").show()

    # correlation coeffients
    prev_correls = [
        pearsonr(sub_eval, obj_eval),
        spearmanr(sub_eval, obj_eval)
    ]
    new_correls = [
        pearsonr(sub_eval, predictions),
        spearmanr(sub_eval, predictions)
    ]
    peamt_correls = [
        pearsonr(sub_eval, peamt_eval),
        spearmanr(sub_eval, peamt_eval)
    ]

    print(
        f"Prev correlations: (pearson, spearman) {prev_correls[0][0]:.2f} {prev_correls[1][0]:.2f}"
    )
    print(
        f"New correlations: (pearson, spearman) {new_correls[0][0]:.2f} {new_correls[1][0]:.2f}"
    )
    print(
        f"Peamt correlations: (pearson, spearman) {peamt_correls[0][0]:.2f} {peamt_correls[1][0]:.2f}"
    )


if __name__ == "__main__":
    main()
