import pretty_midi as pm
import numpy as np
import pickle
import essentia.standard as esst

SR = 44100
FRAME_SIZE = 8192
HOP_SIZE = 2048
BASIS = 10
TEMPLATE_PATH = 'nmf_template.pkl'

SCALE_PATH = ['to_be_synthesized/scale.mid', 'audio/pianoteq_scale.mp3']

notes = pm.PrettyMIDI(midi_file=SCALE_PATH[0]).instruments[0].notes
audio = esst.EasyLoader(filename=SCALE_PATH[1], sampleRate=SR)()
w = esst.Windowing(type='hann')
spectrum_computer = esst.PowerSpectrum(size=FRAME_SIZE)

template = np.zeros(FRAME_SIZE // 2 + 1, 128 * BASIS)
counter = np.zeros(128 * BASIS)

for note in notes:
    # start and end frame
    start = (note.start - 0.01) * SR
    end = (note.end + 0.01) * SR

    spd = np.zeros(4097)
    frames = esst.FrameGenerator(frameSize=FRAME_SIZE, hopSize=HOP_SIZE)
    counter[note.pitch] == len(frames)
    for frame in frames:
        spd += spectrum_computer(w(frame))
    template[:, note.pitch] += spd

template /= counter

pickle.dump(template, open(TEMPLATE_PATH, 'wb'))
