from copy import copy
from .nmf import NMF
from .make_template import TEMPLATE_PATH, HOP_SIZE, SR
from .make_template import BASIS, FRAME_SIZE, ATTACK, BINS
from .utils import make_pianoroll, find_start_stop, midipath2mat
from .utils import stretch_pianoroll, mat2midipath
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from asmd import audioscoredataset
import numpy as np
import sys
import random
from tqdm import tqdm

MINI_SPEC_PATH = 'mini_specs.pkl'
MINI_SPEC_SIZE = 20
DEVICE = 'cuda'
VELOCITY_MODEL_PATH = 'velocity_model.pkl'
COST_FUNC = 'EucDist'
NJOBS = 8
EPS_ACTIVATIONS = 1e-8
NUM_SONGS_FOR_TRAINING = 200
EPOCHS = 500
BATCH_SIZE = 100
EARLY_STOP = 10
# TODO channels
BRANCHES = 16
DATASET_LEN = 1  # use this for debugging


def spectrogram(audio, frames=FRAME_SIZE, hop=HOP_SIZE):

    import essentia.standard as esst
    import essentia as es
    spectrogram = []
    spec = esst.SpectrumCQ(numberBins=BINS, sampleRate=SR, windowType='hann')
    for frame in esst.FrameGenerator(audio, frameSize=frames, hopSize=hop):
        spectrogram.append(spec(frame))

    return es.array(spectrogram).T


def transcribe(audio,
               score,
               data,
               velocity_model=None,
               res=0.001,
               sr=SR,
               align=True,
               return_mini_specs=False):
    """
    Takes an audio mono file and the non-aligned score mat format as in asmd.
    Align them and perform NMF with default templates.
    Returns new score with velocities and timings updated

    `res` is only used for alignment

    `velocity_model` is a callable wich takes a minispectrogram and returns the
    velocity (e.g. a PyTorch nn.Module)

    `align` False can be used for testing other alignment procedures; in that
    case `audio_path` and `res` can be ignored

    """
    from .alignment.align_with_amt import audio_to_score_alignment

    initW, minpitch, maxpitch = data
    initW = copy(initW)
    score = copy(score)
    if align:
        # align score
        new_ons, new_offs = audio_to_score_alignment(score, audio, sr, res=res)
        score[:, 1] = new_ons
        score[:, 2] = new_offs

    # prepare initial matrices

    # remove stoping and starting silence in audio
    start, stop = find_start_stop(audio, sample_rate=sr)
    audio = audio[start:stop]
    V = spectrogram(audio)

    # compute the needed resolution for pianoroll
    res = len(audio) / sr / V.shape[1]
    pr = make_pianoroll(score,
                        res=res,
                        basis=BASIS,
                        velocities=False,
                        attack=ATTACK)

    # remove trailing zeros in initH
    nonzero_cols = pr.any(axis=0).nonzero()[0]
    start = nonzero_cols[0]
    stop = nonzero_cols[-1]
    pr = pr[:, start:stop + 1]

    # stretch pianoroll
    initH = stretch_pianoroll(pr, V.shape[1])

    # check shapes
    assert V.shape == (initW.shape[0], initH.shape[1]),\
        "V, W, H shapes are not comparable"
    assert initH.shape[0] == initW.shape[1],\
        "W, H have different ranks"

    initW = initW[:, minpitch * BASIS:(maxpitch + 1) * BASIS]
    initH = initH[minpitch * BASIS:(maxpitch + 1) * BASIS, :]
    initH[initH == 0] = EPS_ACTIVATIONS

    # perform nfm
    NMF(V, initW, initH, B=BASIS, num_iter=5, cost_func=COST_FUNC)

    NMF(V, initW, initH, B=BASIS, num_iter=5, cost_func=COST_FUNC, fixW=True)

    # use the updated H and W for computing mini-spectrograms
    # and predict velocities
    mini_specs = []
    npitch = maxpitch - minpitch + 1
    initH = initH.reshape(npitch, BASIS, -1)
    initW = initW.reshape((-1, npitch, BASIS), order='C')
    # removing existing velocities
    score[:, 3] = -255
    for note in score:
        # extract mini-spectrogram

        # look for the maximum value in initH in the note
        start = max(0, int(note[1] / res))
        end = min(initH.shape[2], int(note[2] / res))

        if end - start < 1:
            note[3] = 63
            mini_specs.append(None)
            continue

        m = np.argmax(
            np.max(initH[int(note[0] - minpitch), :, start:end],
                   axis=0)) + start

        # select the sorrounding space in initH
        start = max(0, m - MINI_SPEC_SIZE // 2)
        end = min(start + MINI_SPEC_SIZE, initH.shape[2])

        if end - start < MINI_SPEC_SIZE:
            note[3] = 63
            mini_specs.append(None)
            continue

        # compute the mini_spec
        mini_spec = initW[:, int(note[0] - minpitch), :] @\
            initH[int(note[0] - minpitch), :, start:end]

        # normalizing with rms
        mini_spec /= (mini_spec**2).mean()**0.5

        mini_specs.append(mini_spec)

    if return_mini_specs:
        return mini_specs
    else:
        # remove nans...
        mini_specs = [i for i in mini_specs if i is not None]
        # numpy to torch and add channel dimensions
        mini_specs = torch.tensor(mini_specs).to(DEVICE).to(
            torch.float).unsqueeze(1)
        with torch.no_grad():
            vels = velocity_model(mini_specs)
        score[score[:, 3] != 63, 3] = vels.cpu().numpy()
        return score, V, initW, initH


def transcribe_from_paths(audio_path,
                          midi_score_path,
                          data,
                          velocity_model,
                          tofile='out.mid'):
    """
    Load a midi and an audio file and call `transcribe`. If `tofile` is not
    empty, it will also write a new MIDI file with the provided path.
    The output midi file will contain only one track with piano (program 0)
    """
    import essentia.standard as esst
    audio = esst.EasyLoader(filename=audio_path, sampleRate=SR)()
    score = midipath2mat(midi_score_path)
    new_score, _, _, _ = transcribe(audio, score, data, velocity_model)

    # writing to midi
    mat2midipath(new_score, tofile)
    return new_score


def processing(i, dataset, data):
    audio, sr = dataset.get_mix(i, sr=SR)
    score = dataset.get_score(i, score_type=['non_aligned'])
    velocities = dataset.get_score(i, score_type=['precise_alignment'])[:, 3]
    return transcribe(audio, score, data,
                      return_mini_specs=True), velocities.tolist()


def create_mini_specs(data):
    """
    Perform alignment and NMF but not velocity estimation; instead, saves all
    the mini_specs of each note in the Maestro dataset for successive training
    """
    from .maestro_split_indices import maestro_splits
    train, validation, test = maestro_splits()
    dataset = audioscoredataset.Dataset().filter(datasets=["Maestro"])
    random.seed(1750)
    train = random.sample(train, NUM_SONGS_FOR_TRAINING)
    dataset.paths = np.array(dataset.paths)[train].tolist()

    data = dataset.parallel(processing, data, n_jobs=NJOBS)

    mini_specs, velocities = [], []
    for d in data:
        specs, vels = d
        # removing nones
        for i in range(len(specs)):
            spec = specs[i]
            vel = vels[i]
            if spec is not None and vel is not None:
                mini_specs.append(spec)
                velocities.append(vel)

    pickle.dump((mini_specs, velocities), open(MINI_SPEC_PATH, 'wb'))
    print(
        f"number of (inputs, targets) in training set: {len(mini_specs)}, {len(velocities)}"
    )


class VelocityEstimation(nn.Module):
    def __init__(self, in_numel=2000, branches=BRANCHES, k=2):
        super().__init__()

        # TODO Dropout
        self.preprocess = nn.Sequential(nn.BatchNorm2d(1), nn.Dropout(0.4))

        self.in_numel = in_numel
        self.branches = branches
        self.k = k

        self.process = nn.ModuleList()

        for i in range(branches):
            self.process.append(
                nn.Sequential(nn.Linear(in_numel, k, bias=True), nn.SELU()))

        self.finalize = nn.Sequential(
            # TODO
            # nn.BatchNorm1d(branches * k),
            # TODO 3
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False),
            nn.SELU(),
            # TODO Sigmoid -> SELU
            nn.Linear(branches * k, 1, bias=False),
            nn.Sigmoid())

        # self.apply(lambda x: init_weights(x, nn.init.kaiming_uniform_))

    def forward(self, x):

        # preprocess
        x = self.preprocess(x).reshape(x.shape[0], -1)

        # process each velocity range
        y = torch.zeros(x.shape[0], self.branches,
                        self.k).to(x.dtype).to(x.device)
        for i in range(self.branches):
            y[:, i, :] = self.process[i](x)

        # apply softmax so that only the first output is a probability (classification)
        middle_out = F.softmax(y[:, :, 0], dim=1)

        # finalize takes as input the concatenation of all the features of previous layers
        if self.k > 1:
            x = torch.cat([y[..., i] for i in range(1, self.k)], dim=1)
            x = torch.cat([middle_out, x], dim=1)
        else:
            x = middle_out
        x = self.finalize(x)[:, 0] * 127

        return x, middle_out

    def predict(self, x):
        x = self.forward(x)[0]
        return x
        # return torch.argmax(x, dim=1)


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, branches=BRANCHES):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float).to(DEVICE)
        self.targets = torch.tensor(targets).to(torch.float).to(DEVICE)
        self.targets_middle = torch.zeros(len(targets),
                                          branches).to(torch.float).to(DEVICE)
        self.targets_middle[torch.arange(len(targets)), targets % branches] = 1
        assert len(self.inputs) == len(self.targets),\
            "inputs and targets must have the same length!"

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i], self.targets_middle[i]

    def __len__(self):
        return len(self.inputs)


def train(data):

    print("Loading dataset...")
    mini_spec = open(MINI_SPEC_PATH, 'rb')
    inputs, targets = pickle.load(mini_spec)
    mini_spec.close()

    print("Building model...")
    model = VelocityEstimation().to(DEVICE)
    print(model)

    # shuffle and split
    indices = list(range(len(inputs) // DATASET_LEN))
    random.seed(1998)
    random.shuffle(indices)
    inputs = np.array(inputs)
    targets = np.array(targets)
    train_size = int(len(indices) * 0.7)
    test_size = valid_size = int(len(indices) * 0.15)
    train_x = inputs[indices[:train_size]]
    valid_x = inputs[indices[train_size:train_size + valid_size]]
    test_x = inputs[indices[-test_size:]]
    train_y = targets[indices[:train_size]]
    valid_y = targets[indices[train_size:train_size + valid_size]]
    test_y = targets[indices[-test_size:]]

    # creating loaders
    trainloader = torch.utils.data.DataLoader(Dataset(train_x, train_y,
                                                      BRANCHES),
                                              batch_size=BATCH_SIZE)
    validloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_y,
                                                      BRANCHES),
                                              batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(Dataset(test_x, test_y, BRANCHES),
                                             batch_size=BATCH_SIZE)

    # TODO lr
    optim = torch.optim.Adadelta(model.parameters(), lr=1e-2)

    best_epoch = 0
    best_params = None
    best_loss = 9999
    for epoch in range(EPOCHS):
        print(f"-- Epoch {epoch} --")
        trainloss, validloss = [], []
        print("-> Training")
        model.train()
        for inputs, targets, targets_middle in tqdm(trainloader):
            inputs = inputs.to(DEVICE).unsqueeze(1)
            targets = targets.to(DEVICE)
            targets_middle = targets_middle.to(DEVICE)

            optim.zero_grad()
            out, middle_out = model(inputs)
            bce_loss = F.binary_cross_entropy(middle_out, targets_middle)
            l1_loss = F.l1_loss(out, targets)
            # loss = l1_loss
            loss = bce_loss + l1_loss
            loss.backward()
            optim.step()
            trainloss.append(l1_loss.detach().cpu().numpy())

        print(f"training loss : {np.mean(trainloss)}")

        print("-> Validating")
        with torch.no_grad():
            model.eval()
            for inputs, targets, _ in tqdm(validloader):
                inputs = inputs.unsqueeze(1)
                targets = targets.to(DEVICE)
                # targets = torch.argmax(targets, dim=1).to(torch.float)

                out = model.predict(inputs).to(torch.float)
                loss = torch.abs(targets - out)
                validloss += loss.tolist()

        validloss = np.mean(validloss)
        print(f"validation loss : {validloss}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            best_params = model.state_dict()
        elif epoch - best_epoch > EARLY_STOP:
            print("-- Early stop! --")
            break

    # saving params
    model.load_state_dict(best_params)
    pickle.dump(model.to('cpu').state_dict(), open(VELOCITY_MODEL_PATH, 'wb'))
    model.to(DEVICE)

    # testing
    print("-> Testing")
    testloss = []
    with torch.no_grad():
        model.eval()
        for inputs, targets, _ in tqdm(testloader):
            inputs = inputs.unsqueeze(1)
            targets = targets.to(DEVICE)
            # targets = torch.argmax(targets, dim=1).to(torch.float)

            out = model.predict(inputs).to(torch.float)
            loss = torch.abs(targets - out)
            testloss += loss.tolist()

        print(
            f"testing absolute error (mean, std): {np.mean(testloss)}, {np.std(testloss)}"
        )


def get_default_predict_func():
    velocity_model = VelocityEstimation().to(DEVICE)
    velocity_model.load_state_dict(pickle.load(open(VELOCITY_MODEL_PATH,
                                                    'rb')))
    velocity_model.eval()
    return velocity_model.predict


def show_usage():
    print(
        f"Usage: {sys.argv[0]} [audio_path] [midi_score_path] [midi_output_path] [--cpu]"
    )
    print(f"Usage: {sys.argv[0]} create_mini_specs")
    print(f"Usage: {sys.argv[0]} train")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_usage()
    elif sys.argv[1] == 'create_mini_specs':
        data = pickle.load(open(TEMPLATE_PATH, 'rb'))
        create_mini_specs(data)
    elif sys.argv[1] == 'train':
        data = pickle.load(open(TEMPLATE_PATH, 'rb'))
        train(data)
    elif len(sys.argv) < 4:
        show_usage()
    else:
        if '--cpu' in sys.argv:
            DEVICE = 'cpu'
        data = pickle.load(open(TEMPLATE_PATH, 'rb'))
        transcribe_from_paths(sys.argv[1],
                              sys.argv[2],
                              data,
                              get_default_predict_func(),
                              tofile=sys.argv[3])
