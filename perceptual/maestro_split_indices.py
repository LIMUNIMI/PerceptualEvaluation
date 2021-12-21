import json
from asmd.asmd import audioscoredataset

MAESTRO_JSON = "maestro-v2.0.0.json"


def search_audio_filename_in_original_maestro(filename, maestro):
    for song in maestro:
        if song["audio_filename"] == filename:
            return song["split"]
    return None


def maestro_splits():
    d = audioscoredataset.Dataset().filter(datasets=['Maestro'])

    maestro = json.load(open(MAESTRO_JSON))
    train, validation, test = [], [], []
    for i in range(len(d)):
        filename = d.paths[i][0][0][23:]
        split = search_audio_filename_in_original_maestro(filename, maestro)
        if split == "train":
            train.append(i)
        elif split == "validation":
            validation.append(i)
        elif split == "test":
            test.append(i)
        else:
            raise RuntimeError(filename +
                               "  not found in maestro original json")
    return train, validation, test
