from copy import copy
from .alignment.align_with_amt import audio_to_score_alignment
from .nmf import NMF
from .make_template import TEMPLATE_PATH, HOP_SIZE, SR, BASIS, FRAME_SIZE
from essentia.standard import FrameGenerator, PowerSpectrum, Windowing
import essentia as es
from .utils import make_pianoroll
import pickle
import torch
from torch import nn

DEVICE = 'cuda'
VELOCITY_MODEL_PATH = 'velocity_model.pkl'


def spectrogram(audio, frames=FRAME_SIZE, hop=HOP_SIZE):

    spectrogram = []
    w = Windowing(type='hann')
    spectrum_computer = PowerSpectrum(frames)
    for frame in FrameGenerator(frameSize=frames, hopSize=hop):
        spectrogram.append(spectrum_computer(w(frame)))

    return es.array(spectrogram).T


def transcribe(audio, score, audio_path, initW, velocity_model, res=0.01, sr=SR):
    """
    Takes an audio mono file and the non-aligned score mat format as in asmd.
    Align them and perform NMF with default templates.
    Returns new score with velocities and timings updated

    `res` is only used for alignment

    `velocity_model` is a callable wich takes a minispectrogram and returns the
    velocity (e.g. a PyTorch nn.Module)
    """
    score = copy(score)
    # align score
    new_ons, new_offs = audio_to_score_alignment(score, audio_path, res=res)
    score[:, 1] = new_ons
    score[:, 2] = new_offs

    # prepare initial matrices
    V = spectrogram(audio)
    res = (len(audio) / sr) / V.shape[1]
    initH = make_pianoroll(score, res=res, basis=BASIS, velocities=False,
                           attack=0.08)
    assert V.shape == (initW.shape[0], initH.shape[1]),\
        "V, W, H shapes are not comparable"
    assert initH.shape[0] == initW.shape[1],\
        "W, H have different ranks"

    # prepare constraints
    params = {'Mh': None, 'Mw': None}

    # perform nfm
    NMF(V, initW, initH, params, B=BASIS, num_iter=8)

    # another update unconstrained
    params['a2'] = 0
    params['a3'] = 0
    NMF(V, initW, initH, params, B=BASIS, num_iter=1)

    # use the updated H and W for computing mini-spectrograms
    # and predict velocities
    velocity_model = build_model((initW.shape[0], BASIS))
    initH = initH.reshape(BASIS, 128, -1)
    initW = initH.reshape(-1, 128, BASIS)
    for note in score:
        start = int((note[1] - 0.05) * res)
        end = int((note[2] + 0.05) * res) + 1
        mini_spec = initW[:, note[0], :] * initH[:, note[0], start:end]
        # numpy to torch and add batch dimension
        mini_spec = torch.tensor(mini_spec).to(DEVICE).unsqueeze(0)
        vel = velocity_model(mini_spec)
        note[3] = vel.cpu().value

    return score


def build_model(spec_size):
    n_in, n_h, n_out = spec_size[0], spec_size[1], 1
    model = nn.Sequential(nn.Linear(n_in, n_h), nn.SELU(), nn.Linear(n_h, n_h),
                          nn.SELU(), nn.Linear(n_h, n_h), nn.SELU(),
                          nn.Linear(n_h, n_h), nn.SELU(), nn.Linear(n_h, n_h),
                          nn.SELU(), nn.Linear(n_h, n_out)).to(DEVICE)
    model.load_state_dict(open(VELOCITY_MODEL_PATH, 'rb'))

    return model


def transcribe_from_paths(audio_path, midi_score_path, tofile=''):
    """
    Load a midi and an audio file and call `transcribe`. If `tofile` is not
    empty, it will also write a new MIDI file with the provided path.
    """


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} [audio_path], [midi_score_path] [midi_output_path]"
        )
    else:
        initW = pickle.load(open(TEMPLATE_PATH, 'rb'))
        velocity_model = build_model((initW.shape[0], BASIS))
        transcribe_from_paths(sys.argv[1], sys.argv[2], sys.argv[3], initW, velocity_model)
