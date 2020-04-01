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
from torch import nn

DEVICE = 'cuda'
VELOCITY_MODEL_PATH = 'velocity_model.pkl'
COST_FUNC = 'EucDist'


def spectrogram(audio, frames=FRAME_SIZE, hop=HOP_SIZE):

    spectrogram = []
    spec = esst.SpectrumCQ(numberBins=BINS, sampleRate=SR, windowType='hann')
    for frame in esst.FrameGenerator(audio, frameSize=frames, hopSize=hop):
        spectrogram.append(spec(frame))

    return es.array(spectrogram).T


def transcribe(audio,
               score,
               data,
               velocity_model,
               res=0.02,
               sr=SR,
               align=True):
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

    params = {'Mh': copy(initH), 'Mw': copy(initW)}

    # perform nfm
    NMF(V,
        initW,
        initH,
        params=params,
        B=BASIS,
        num_iter=20,
        cost_func=COST_FUNC)

    params['a3'] = 0
    NMF(V,
        initW,
        initH,
        params=params,
        B=BASIS,
        num_iter=2,
        cost_func=COST_FUNC,
        fixW=True)

    # use the updated H and W for computing mini-spectrograms
    # and predict velocities
    npitch = maxpitch - minpitch + 1
    initH = initH.reshape(npitch, BASIS, -1)
    initW = initH.reshape((-1, npitch, BASIS), order='C')
    for note in score:
        start = int((note[1] - 0.05) * res)
        end = int((note[2] + 0.05) * res) + 1
        mini_spec = initW[:, note[0] - minpitch, :] *\
            initH[note[0] - minpitch, :, start:end]

        # numpy to torch and add batch dimension
        # mini_spec = torch.tensor(mini_spec).to(DEVICE).unsqueeze(0)
        vel = velocity_model(mini_spec)
        note[3] = vel.cpu().value

    return score, V, initW, initH


def build_model(spec_size):
    n_in, n_h, n_out = spec_size[0], spec_size[1], 1
    model = nn.Sequential(nn.Linear(n_in, n_h), nn.SELU(), nn.Linear(n_h, n_h),
                          nn.SELU(), nn.Linear(n_h, n_h), nn.SELU(),
                          nn.Linear(n_h, n_h), nn.SELU(), nn.Linear(n_h, n_h),
                          nn.SELU(), nn.Linear(n_h, n_out)).to(DEVICE)
    # model.load_state_dict(open(VELOCITY_MODEL_PATH, 'rb'))

    return model


def transcribe_from_paths(audio_path,
                          midi_score_path,
                          data,
                          velocity_model,
                          tofile='out.mid'):
    """
    Load a midi and an audio file and call `transcribe`. If `tofile` is not
    empty, it will also write a new MIDI file with the provided path.
    """
    audio = esst.EasyLoader(filename=audio_path, sampleRate=SR)()
    score = midipath2mat(midi_score_path)
    new_score = transcribe(audio, score, data, velocity_model)

    # write midi


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} [audio_path], [midi_score_path] [midi_output_path]"
        )
    else:
        data = pickle.load(open(TEMPLATE_PATH, 'rb'))
        velocity_model = build_model((data[0].shape[0], BASIS))
        transcribe_from_paths(sys.argv[1], sys.argv[2], data, velocity_model,
                              sys.argv[3])
