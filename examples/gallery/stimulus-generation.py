'''
Generating calibrated stimuli using factories
=============================================

This demonstrates how to combine the factories to create calibrated stimuli.
The calibrated stimuli are then generated in blocks. This block-based approach
allows us to create infinite-duration stimuli that are "fed" into the speaker
buffer incrementially.
'''


import matplotlib.pylab as plt

###############################################################################
# Psiaudio supports several types of calibration. The simplest calibration
# assumes that frequency response is "flat". In other words, if you send a 1V
# RMS tone to the speaker, it will always produce the same output level
# regardless of frequency. You can also have calibrations that compensate for
# variations in speaker output as a function of frequency.
#
# For simplicity, let's assume our speaker's response is flat. First, import
# the calibration class that supports flat calibrations.
from psiaudio.calibration import FlatCalibration


###############################################################################
# Now, let's assume that when we play a 1V RMS tone through the speaker, it
# produces an output of 114 dB SPL. Sensitivity of acoustic systems are often
# reported in millivolts per Pascal. 114 dB SPL is 10 Pascals. This means that
# the sensitivity of the speaker is 0.1 volt per Pascal or 100 millivolts per
# Pascal.
#
# We have a convenience method, `FlatCalibration.from_mv_pa` to create the
# calibration.
calibration = FlatCalibration.from_mv_pa(0.1e3)


###############################################################################
# One Pascal is 94 dB. Let's see if this works. The method,
# `Calibration.get_sf` gives us the RMS amplitude of the waveform needed to
# generate a tone at the given frequency and level. We would expect the RMS
# value to be 0.1.
calibration.get_sf(frequency=1000, level=94)

###############################################################################
# Remember that 6 dB translates to half on a linear scale. Let's confirm this
# works (we expect the RMS value to be 0.05).
calibration.get_sf(frequency=1000, level=94-6)

###############################################################################
# Now that we've defined our calibration, we can generate a stimulus waveform.
# Let's start with the simplest possible type of stimulus, a tone. First, we
# import the ToneFactory class and create an instance.
from psiaudio.stim import ToneFactory

fs = 100000
tone = ToneFactory(fs=fs, frequency=1000, level=80, calibration=calibration)

###############################################################################
# Note that we had to provide the sampling frequency (`fs`) the tone must be
# generated at along with other stimulus parameters.
#
# The instance supports several methods that are used by psiexperiment to
# properly handle the tone. For example, we need to know how long the stimulus
# is.
tone.get_duration()

###############################################################################
# This means the tone can run continuously for the full duration of the
# experiment. You may use a continuous waveform (e.g., bandlimited noise) for
# generating a background masker.
#
# Let's get the first 5000 samples of the tone.
waveform = tone.next(5000)
plt.plot(waveform)
plt.show()

###############################################################################
# Let's get the next 1000 samples.
waveform = tone.next(1000)
plt.plot(waveform)
plt.show()

###############################################################################
# As you can see, a factory supports *incremential* generation of waveforms.
# This enables us to generate infinitely long waveforms (such as maskers) that
# never repeat.
#
# Tones are boring. Let's look at a more interesting type of stimulus.
# Sinusoidally-amplitude modulated noise with a cosine-squared onset/offset
# ramp. The `Cos2EnvelopeFactory` is a modulator, which means that it takes, as
# it's input, another factory (e.g., a tone) and applies a transform to it.
from psiaudio.stim import Cos2EnvelopeFactory

tone = ToneFactory(fs=fs, frequency=16000, level=94, calibration=calibration)
envelope = Cos2EnvelopeFactory(fs=100000, start_time=0, rise_time=5e-3,
                               duration=10, input_factory=tone)

waveform = envelope.next(1000)
plt.figure()
plt.plot(waveform)

waveform = envelope.next(1000)
plt.figure()
plt.plot(waveform)

plt.figure()
plt.specgram(waveform, Fs=fs);
plt.show()

###############################################################################
# The Cos2EnvelopeFactory has a finite duration
envelope.get_duration()

###############################################################################
# Let's take a look at bandlimited noise, which is a more commonly used
# background masker.
from psiaudio.stim import BandlimitedNoiseFactory

noise = BandlimitedNoiseFactory(fs=fs, seed=0, level=94, fl=2000,
                                fh=8000,  filter_rolloff=1,
                                passband_attenuation=1,
                                stopband_attenuation=80,
                                equalize=False, calibration=calibration)
waveform = noise.next(5000)
plt.plot(waveform)
plt.show()

###############################################################################
# Like tone factories, the bandlimited noise factory can run forever if you
# want it to.
noise.get_duration()

###############################################################################
# Now, let's embed the noise in a sinusoidally amplitude-modulated (SAM)
# envelope. Note that when we create this factory, we provide the noise we
# created as an argument to the parameter `input_waveform`.
from psiaudio.stim import SAMEnvelopeFactory

sam_envelope = SAMEnvelopeFactory(fs=fs, depth=1, fm=5,
                                  delay=1, direction=1,
                                  calibration=calibration,
                                  input_factory=noise)
waveform = sam_envelope.next(fs*2)
plt.plot(waveform)
plt.show()

###############################################################################
# Unlike the Cos2EnvelopeFactory, the SAMEnvelopeFactory has a finite duration.
sam_envelope.get_duration()

###############################################################################
# Now, embed the SAM noise inside a cosine-squared envelope.
cos_envelope = Cos2EnvelopeFactory(fs=fs, start_time=0,
                                   rise_time=0.25, duration=4,
                                   input_factory=sam_envelope)

###############################################################################
# By definition, a cosine-squared envelope has a finite duration. Let's plot
# the first two seconds.
waveform = cos_envelope.next(fs*2)
plt.plot(waveform)
plt.show()

###############################################################################
# Now, the next two seconds.
waveform = cos_envelope.next(fs*2)
plt.plot(waveform)
plt.show()

###############################################################################
# What happens if we keep going? Remember the duration of the stimulus is only
# 4 seconds.
waveform = cos_envelope.next(fs*2)
plt.plot(waveform)
plt.show()

###############################################################################
# That's because the stimulus is over. We can check that this is the case.
cos_envelope.is_complete()

###############################################################################
# What if we want to start over at the beginning? Reset it.
cos_envelope.reset()
waveform = cos_envelope.next(100000*4)
plt.plot(waveform)
plt.show()

###############################################################################
# Not all stimuli have to be composed of individual building blocks (e.g.,
# envelopes, modulators and carriers). We can also define discrete waveform
# factories that can be used as-is. For example, chirps.
from psiaudio.stim import ChirpFactory

chirp = ChirpFactory(fs=100000, start_frequency=50, end_frequency=5000,
                     duration=1, level=94, calibration=calibration)

waveform = chirp.next(5000)
plt.plot(waveform)
chirp.get_duration()

###############################################################################
# To create your own, you would subclass `psiaudio.stim.Waveform` and implement
# the following methods:
#
# * ``__init_``: Where you perform potentially expensive computations (such as
#   the filter coefficients for bandlimited noise).
# * ``reset``: Where you reset any settings that are releavant to incremential
#   generation of the waveform (e.g., the initial state of the filter and the
#   random number generator for bandlimited noise).
# * ``next``: Where you actually generate the waveform.
# * ``get_duration``: The duration of the waveform. Return `np.inf` if
#   continuous.
#
# See the ``psiaudio.stim`` module for examples (e.g.,
# ``BandlimitedNoiseFactory`` and ``ChirpFactory``).
