import numpy as np
from numpy.testing import assert_array_equal
import pytest

from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory


@pytest.fixture
def factory(fs, stim_calibration):
    tone = ToneFactory(fs=fs, level=94, frequency=1000,
                       calibration=stim_calibration)
    envelope = Cos2EnvelopeFactory(fs=fs, start_time=0, rise_time=0.5,
                                   duration=5, input_factory=tone)
    return envelope


def test_factory_reset(factory):
    n = factory.n_samples_remaining()
    full_waveform = factory.get_samples_remaining()
    full_waveform_post = factory.next(n)
    assert np.all(full_waveform_post == 0)
    assert n == full_waveform.shape[-1]

    factory.reset()
    full_waveform_reset = factory.get_samples_remaining()
    full_waveform_reset_post = factory.next(n)
    assert np.all(full_waveform_reset_post == 0)
    assert_array_equal(full_waveform, full_waveform_reset)


def test_factory_chunks(factory, chunksize):
    n = factory.n_samples_remaining()
    full_waveform = factory.get_samples_remaining()
    factory.reset()

    chunks = []
    while not factory.is_complete():
        chunks.append(factory.next(chunksize))
    chunks = np.concatenate(chunks, axis=-1)
    assert_array_equal(full_waveform, chunks[:n])
