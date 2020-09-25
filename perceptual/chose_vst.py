from .excerpt_search import audio_features
from .utils import farthest_points
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
import essentia.standard as esst
import numpy as np
import os
import shutil
import subprocess

SR = 44100
VST = 2
DIST = 'euclidean'
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.aif')


def load_audio_files_in_dir(path):

    vsts = []
    original = []
    paths = []
    for root, dirs, files in os.walk(path):
        FOUND_VST = False
        vst = []
        for file in files:
            if file.endswith(AUDIO_EXTS) and 'scales' not in file:
                if 'target' in file:
                    audios = original
                else:
                    FOUND_VST = True
                    audios = vst
                audios.append(
                    esst.EasyLoader(filename=os.path.join(root, file),
                                    sampleRate=SR)())
        if FOUND_VST:
            vsts.append(vst)
            paths.append(root)
    return vsts, original, paths


def audio_features_from_set(set):
    """
    Computes features averaged over each item in the set
    """

    audio = []
    for i in set:
        audio.append(audio_features(i))

    return np.mean(audio, axis=0)


def move_files(path, q='', n='', type='', dir='./excerpts', filter=''):

    if not os.path.exists(dir):
        os.makedirs(dir)

    params = (q, n, type)
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isfile(file) and 'scales' not in file:
            root, base = os.path.split(file)
            name, ext = os.path.splitext(base)
            q_, n_, type_ = name.split('_')
            if type_ != filter and filter:
                continue
            if not q:
                q = q_
            if not n:
                n = n_
            if not type:
                type = type_
            shutil.copy(file, os.path.join(dir, f"{q}_{n}_{type}{ext}"))
            q, n, type = params


def main(path):

    # load all files
    vsts, original, paths = load_audio_files_in_dir(path)

    # substituting audio arrays with their features
    for i in range(len(vsts)):
        vsts[i] = audio_features_from_set(vsts[i])

    original = audio_features_from_set(original)

    # looking for the vst farthest from original
    dist = getattr(distance, DIST)
    max_d = -1
    for i, vst in enumerate(vsts):
        d = dist(vst, original)
        if d > max_d:
            max_d = d
            chosen = i

    # taking path of the correct VST
    q0 = paths[chosen]

    # removing this vst from the set
    del paths[chosen]
    del vsts[chosen]

    # looking for the medoid
    distmat = squareform(pdist(vsts))
    medoid = np.argmin(np.sum(distmat, axis=1))
    q2 = paths[medoid]

    # removing this vst from the set
    del paths[medoid]
    del vsts[medoid]

    # looking for farthest VSTs
    vsts = np.array(vsts)
    vsts = StandardScaler().fit_transform(vsts)
    pca = PCA(n_components=10)
    vsts = pca.fit_transform(vsts)
    print("Explained variance: ", pca.explained_variance_ratio_,
          pca.explained_variance_ratio_.sum())
    points = farthest_points(vsts, VST, 1)

    # taking paths of the correct VSTs
    q1 = [paths[i] for i in points[:, 0]]

    # moving audio files to the excerpts dir
    # move all the files in q0, q1[0] and q2 to name q0_[], q1_[] etc
    print(f"path q0: {q0}")
    print(f"path q1: {q1}")
    print(f"path q2: {q2}")
    move_files(q0, q='q0')
    move_files(q1[0], q='q1')
    move_files(q2, q='q2')

    # the target is the one which ends with 'orig'
    # for q0 we use had produced excerpts with the correct name
    move_files(path, q='q0', filter='target')
    move_files(q1[1], q='q1', type='target', filter='orig')
    move_files(q2, q='q2', type='target', filter='orig')


def post_process(in_dir, out_dir, options=[]):
    """
    Recursively walk across a directory and apply sox command with `options` to
    all the audio files with extension `ext`.
    """
    # check that sox is installed:
    if shutil.which('sox') is None:
        raise RuntimeError(
            "Sox is needed, install it or run with '--no-postprocess' option")

    for root, dirs, files in os.walk(in_dir):
        if 'reverb' in root:
            # skipping directories created by this script
            continue

        # computing the directory name in which new files are stored
        r = root[len(dir) + 1:]
        if len(r) > 0:
            r += '-'
        new_dir = r + '_'.join(options)
        new_out_dir = os.path.join(out_dir, new_dir)
        for file in files:
            if file.endswith(
                    AUDIO_EXTS
            ) and 'target' not in file and 'scales' not in file:
                if not os.path.exists(new_out_dir):
                    os.makedirs(new_out_dir)

                # run sox
                proc = subprocess.Popen(
                    ['sox', os.path.join(root, file)] +
                    [os.path.join(new_out_dir, file)] + options)
                proc.wait()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != '--no-postprocess':
        dir = sys.argv[1]
    else:
        dir = './audio'

    if '--no-postprocess' not in sys.argv:
        post_process(dir, dir, ['norm', '-20', 'reverb', '50', 'norm'])
        post_process(dir, dir, ['norm', '-20', 'reverb', '100', 'norm'])

    main(dir)
