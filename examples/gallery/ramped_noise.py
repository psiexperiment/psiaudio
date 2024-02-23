'''
Generate noise linearly ramped in dB
====================================

This demonstrates how to generate a bandlimited noise that is ramped linearly
at a fixed rate in dB per second.
'''
from matplotlib import pyplot as plt
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import EnvelopeFactory, BandlimitedFIRNoiseFactory

from psiaudio import util

################################################################################
# Define some basic parameters. The noise will be ramped from 0 dB SPL to
# `max_level` at the given `ramp_rate`. Sweep duration is defined by the max
# noise level and ramp rate.
#
# The calibration is set such that 1 Vrms = 1 Pa.
fs = 50e3
calibration = FlatCalibration.from_spl(94)

max_level = 80   # max level for testing
ramp_rate = 17.5 # dB/s

sweep_duration = max_level / ramp_rate * 2


################################################################################
# Assemble the factories that will generate the noise. The bartlette envelope
# is a triangular envelope. Since the envelope operates on the waveform, which
# is in units of Vrms, we need to apply an exponential transform to achieve the
# equivalent scaling for the dB domain.
noise = BandlimitedFIRNoiseFactory(
    fs=fs,
    fl=8e3,
    fh=16e3,
    level=94,
    calibration=calibration,
)


stim = EnvelopeFactory(
    envelope='bartlett',
    fs=fs,
    duration=sweep_duration,
    rise_time=None,
    transform=lambda x: x ** 10,
    input_factory=noise,
)


s = stim.get_samples_remaining()
t = np.arange(len(s)) / fs

plt.plot(t, s)
plt.show()

plt.specgram(s, Fs=fs, scale='dB')
plt.show()
