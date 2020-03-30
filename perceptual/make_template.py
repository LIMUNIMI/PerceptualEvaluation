import pretty_midi as pm
import numpy as np
import pickle
import sys
import essentia.standard as esst
from tqdm import tqdm

SR = 44100
FRAME_SIZE = 8192
HOP_SIZE = 2048
BASIS = 10
# the number of frames for the attack
ATTACK = 1
# the number of frames for the other basis
BASIS_L = 1
TEMPLATE_PATH = 'nmf_template.pkl'

SCALE_PATH = ['to_be_synthesized/scale.mid', 'audio/pianoteq_scale.mp3']

notes = pm.PrettyMIDI(midi_file=SCALE_PATH[0]).instruments[0].notes
audio = esst.EasyLoader(filename=SCALE_PATH[1], sampleRate=SR)()
w = esst.Windowing(type='hann')
spectrum_computer = esst.PowerSpectrum(size=FRAME_SIZE)

template = np.zeros(FRAME_SIZE // 2 + 1, BASIS, 128)
counter = np.zeros(BASIS, 128)

for note in tqdm(notes):
    # start and end frame
    start = (note.start - 0.01) * SR
    end = (note.end + 0.01) * SR
    ENDED = False

    spd = np.zeros(BASIS, FRAME_SIZE // 2 + 1)
    frames = esst.FrameGenerator(audio[start:end],
                                 frameSize=FRAME_SIZE,
                                 hopSize=HOP_SIZE)
    # attack
    for a in range(ATTACK):
        try:
            frame = next(frames)
        except StopIteration:
            print("Error: notes timing not correct")
            print(f"note: {start}, {end}, {len(audio)}")
            sys.exit(99)
        spd[0] += spectrum_computer(w(frame))
    counter[0, note.pitch] += ATTACK

    # other basis except the last one
    for b in range(1, BASIS-1):
        if not ENDED:
            for a in range(BASIS_L):
                try:
                    frame = next(frames)
                except StopIteration:
                    # note is shorter than the number of basis
                    ENDED = True
                    break
                spd[b] += spectrum_computer(w(frame))
                counter[b, note.pitch] += 1

    # last basis
    if not ENDED:
        for frame in frames:
            spd[BASIS-1] += spectrum_computer(w(frame))
            counter[BASIS-1, note.pitch] += 1
    template[:, :, note.pitch] += spd

template /= counter

pickle.dump(template, open(TEMPLATE_PATH, 'wb'))
