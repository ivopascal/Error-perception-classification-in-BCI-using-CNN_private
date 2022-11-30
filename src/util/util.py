from settings import SAMPLING_FREQUENCY


def milliseconds_to_samples(milliseconds: int, sampling_frequency: int = SAMPLING_FREQUENCY) -> int:
    return int(milliseconds * sampling_frequency / 1000)


def samples_to_milliseconds(samples: int, sampling_frequency: int = SAMPLING_FREQUENCY) -> int:
    return int(samples / sampling_frequency * 1000)
