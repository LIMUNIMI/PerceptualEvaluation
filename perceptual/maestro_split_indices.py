import pickle
import json
from asmd import audioscoredataset


def search_audio_filename_in_original_maestro(filename, maestro):
    for song in maestro:
        if song["audio_filename"] == filename:
            return song["split"]
    return None


if __name__ == "__main__":
    d = audioscoredataset.Dataset()
    d.filter(datasets=['Maestro'])

    maestro = json.load(open("maestro-v2.0.0.json"))
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

    pickle.dump((["Maestro"], [train, validation, test]),
                open("maestro_split_indices.pkl", "wb"))
