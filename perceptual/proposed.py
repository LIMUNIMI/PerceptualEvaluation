from copy import copy
from .nmf import NMF
from .alignment.align_with_amt import audio_to_score_alignment
from .make_template import TEMPLATE_PATH, HOP_SIZE, SR
from .make_template import BASIS, FRAME_SIZE, ATTACK, BINS
import essentia.standard as esst
import essentia as es
from .utils import make_pianoroll, find_start_stop, midipath2mat
from .utils import stretch_pianoroll
import pickle
import lzma
import torch
import torch.nn.functional as F
from torch import nn
from asmd import audioscoredataset
import numpy as np
import sys
from .maestro_split_indices import maestro_splits
import pretty_midi
import random
from tqdm import tqdm

MINI_SPEC_PATH = 'mini_specs.pkl.xz'
MINI_SPEC_SIZE = 30
DEVICE = 'cuda:0'
VELOCITY_MODEL_PATH = 'velocity_model.pkl'
COST_FUNC = 'EucDist'
NJOBS = 4
EPS_ACTIVATIONS = 1e-4
NUM_SONGS_FOR_TRAINING = 20
EPOCHS = 100
BATCH_SIZE = 100


def spectrogram(audio, frames=FRAME_SIZE, hop=HOP_SIZE):

    spectrogram = []
    spec = esst.SpectrumCQ(numberBins=BINS, sampleRate=SR, windowType='hann')
    for frame in esst.FrameGenerator(audio, frameSize=frames, hopSize=hop):
        spectrogram.append(spec(frame))

    return es.array(spectrogram).T


def transcribe(audio,
               score,
               data,
               velocity_model=None,
               res=0.02,
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
    NMF(V, initW, initH, B=BASIS, num_iter=10, cost_func=COST_FUNC)

    NMF(V, initW, initH, B=BASIS, num_iter=2, cost_func=COST_FUNC, fixW=True)

    # use the updated H and W for computing mini-spectrograms
    # and predict velocities
    mini_specs = []
    npitch = maxpitch - minpitch + 1
    initH = initH.reshape(npitch, BASIS, -1)
    initW = initW.reshape((-1, npitch, BASIS), order='C')
    for note in score:
        # extract mini-spectrogram
        mini_spec = np.zeros((initW.shape[0], MINI_SPEC_SIZE))
        start = max(0, int((note[1] - 0.05) / res))
        end = min(initH.shape[2], int((note[2] + 0.05) / res) + 1)
        if end - start <= 1:
            mini_specs.append(None)
            continue
        _mini_spec = initW[:, int(note[0] - minpitch), :] @\
            initH[int(note[0] - minpitch), :, start:end]

        # looking for the frame with maximum energy (MEF)
        m = np.argmax(np.sum(_mini_spec, axis=0))

        # segment the window so that the position of the MEF is significant in
        # the input mini-spec
        start = m - MINI_SPEC_SIZE // 2
        if start < 0:
            begin_pad = -start
            start = 0
        else:
            begin_pad = 0
        end = min(m + MINI_SPEC_SIZE // 2 + 1, _mini_spec.shape[1])
        end_pad = end - start + begin_pad
        if end_pad > MINI_SPEC_SIZE:
            end = end - (end_pad - MINI_SPEC_SIZE)
            end_pad = MINI_SPEC_SIZE
        mini_spec[:, begin_pad:end_pad] += _mini_spec[:, start:end]

        if return_mini_specs:
            mini_specs.append(mini_spec)
        else:
            # numpy to torch and add batch dimension
            mini_spec = torch.tensor(mini_spec).to(DEVICE).unsqueeze(0)
            vel = velocity_model(mini_spec)
            note[3] = vel.cpu().value
    if return_mini_specs:
        return mini_specs
    else:
        return score, V, initW, initH


def build_model(spec_size):
    # spec_size is (100, 20) -> 2000
    n_in, n_out = spec_size[0] * spec_size[1], 1
    model = nn.Sequential(
        nn.Linear(n_in, n_in // 2),
        nn.ReLU(),  # 1000
        nn.Linear(n_in // 2, n_in // 4),
        nn.ReLU(),  # 500
        nn.Linear(n_in // 4, n_in // 8),
        nn.ReLU(),  # 250
        nn.Linear(n_in // 8, n_in // 16),
        nn.ReLU(),  # 125
        nn.Linear(n_in // 16, n_in // 32),
        nn.ReLU(),  # 62
        nn.Linear(n_in // 32, n_in // 64),
        nn.ReLU(),  # 31
        nn.Linear(n_in // 64, n_out),
        nn.Sigmoid())

    def predict(self, x):
        return round(self.forward(x) * 127)

    model.predict = predict

    return model


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
    audio = esst.EasyLoader(filename=audio_path, sampleRate=SR)()
    score = midipath2mat(midi_score_path)
    new_score = transcribe(audio, score, data, velocity_model)

    # creating pretty_midi.PrettyMIDI object and inserting notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments = [pretty_midi.Instrument(0)]
    for row in new_score:
        midi.instruments[0].notes.append(
            pretty_midi.Note(100, int(row[0]), float(row[1]), float(row[2])))

    # writing to file
    midi.write(tofile)
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
            if spec is not None:
                mini_specs.append(spec)
                velocities.append(vels[i])
        mini_specs += specs
        velocities += vels

    pickle.dump((mini_specs, velocities), lzma.open(MINI_SPEC_PATH, 'wb'))
    print(
        f"number of (inputs, targets) in training set: {len(mini_specs)}, {len(velocities)}"
    )


def train(data):

    model = build_model((data[0].shape[0], BASIS)).to(DEVICE)
    inputs, targets = pickle.load(lzma.open(MINI_SPEC_PATH, 'rb'))

    # shuffle and split
    indices = list(range(len(inputs)))
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

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            super().__init__()
            self.inputs = inputs
            self.targets = targets

        def __getitem__(self, i):
            input = torch.tensor(self.inputs[i]).flatten()
            target = torch.tensor(self.targets[i])
            return input, target

    # creating loaders
    trainloader = torch.utils.data.DataLoader(Dataset(train_x, train_y),
                                              batch_size=BATCH_SIZE,
                                              num_workers=NJOBS,
                                              pin_memory=True)
    validloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_y),
                                              batch_size=BATCH_SIZE,
                                              num_workers=NJOBS,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(Dataset(test_x, test_y),
                                             batch_size=BATCH_SIZE,
                                             num_workers=NJOBS,
                                             pin_memory=True)

    optim = torch.optim.Adadelta(model.parameters())

    # training and validating
    best_epoch = 0
    best_params = None
    best_loss = 9999
    for epoch in range(EPOCHS):
        print(f"-- Epoch {epoch} --")
        trainloss, validloss = [], []
        print("-> Training")
        model.train()
        for inputs, targets in tqdm(trainloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optim.zero_grad()
            out = model.predict(inputs)
            loss = F.l1_loss(targets, out)
            loss.backward()
            optim.step()
            trainloss.append(loss.detach().cpu().numpy())

        print(f"training loss : {np.mean(trainloss)}")

        print("-> Validating")
        with torch.no_grad():
            model.eval()
            for inputs, targets in tqdm(validloader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                out = model.predict(inputs)
                loss = F.l1_loss(targets, out)
                validloss.append(loss.detach().cpu().numpy())

        validloss = np.mean(validloss)
        print(f"validation loss : {validloss}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            best_params = model.state_dict()
        elif epoch - best_epoch > 10:
            print("-- Early stop! --")
            break

    # saving params
    pickle.dump(open(VELOCITY_MODEL_PATH, 'wb'))
    model.load_state_dict(best_params)

    # testing
    print("-> Testing")
    testloss = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(testloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            out = model.predict(inputs)
            loss = F.l1_loss(targets, out)
            testloss.append(loss.detach().cpu().numpy())

        print(
            f"testing loss (mean, std): {np.mean(testloss)}, {np.std(testloss)}"
        )


def show_usage():
    print(
        f"Usage: {sys.argv[0]} [audio_path] [midi_score_path] [midi_output_path]"
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
        data = pickle.load(open(TEMPLATE_PATH, 'rb'))
        velocity_model = build_model((data[0].shape[0], BASIS)).to(DEVICE)
        velocity_model.load_state_dict(open(VELOCITY_MODEL_PATH, 'rb'))
        transcribe_from_paths(sys.argv[1], sys.argv[2], data, velocity_model,
                              sys.argv[3])
