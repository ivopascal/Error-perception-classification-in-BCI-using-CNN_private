import numpy as np

from settings import SAMPLING_FREQUENCY


def milliseconds_to_samples(milliseconds: int, sampling_frequency: int = SAMPLING_FREQUENCY) -> int:
    return int(milliseconds * sampling_frequency / 1000)


def samples_to_milliseconds(samples: int, sampling_frequency: int = SAMPLING_FREQUENCY) -> int:
    return int(samples / sampling_frequency * 1000)


def numpy_entropy(probs, axis=-1, eps=1e-6):
    return -np.sum(probs * np.log(probs + eps), axis=axis)


def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)