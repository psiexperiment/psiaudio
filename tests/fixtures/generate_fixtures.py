"""
Regenerate golden fixture files for stim regression tests.

Run from the project root:
    python tests/fixtures/generate_fixtures.py [fixture name ...]

With no names, every fixture is regenerated.

Re-run this whenever stim.py is intentionally changed in a way that alters
the generated waveforms.  The regenerated .npy files should be committed
alongside the code change so the intent is visible in the diff.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from psiaudio import calibration, stim

FIXTURE_DIR = Path(__file__).parent
CAL = calibration.FlatCalibration.from_spl(94)


def load_golay_calibration():
    '''
    Speaker calibration measured with cftscal's Golay routine (coati,
    DS18-sealed, 2026-10-07), loaded the way cftscal's
    `CFTSSpeakerCalibration.load_golay` loads it. The file keeps only the rows
    that loader selects (n_bits=14, output_gain=-15), copied verbatim.

    Its sensitivity has deep, narrow reflection notches (to -34 dB re its
    1/3-octave average in 2.8-5.6 kHz), which is what made equalized noise
    play too loud. The tests load it through this function too.
    '''
    sens = pd.read_csv(FIXTURE_DIR / 'golay_speaker_sens.csv.gz',
                       index_col=['n_bits', 'output_gain', 'frequency'])
    s = sens.loc[14, -15]['sens']
    return calibration.InterpCalibration(s.index.values, s.values)


GOLAY_CAL = load_golay_calibration()

# Settings of the noise-exposure paradigm (noise-exp) with its FIR noise token,
# which builds the factory itself (`bandlimited_fir_noise` has no
# `max_correction`).
FIR_NOISE_KW = dict(fs=100e3, level=100, ntaps=1001, seed=1,
                    max_correction=20, calibration=GOLAY_CAL)
FIR_NOISE_SAMPLES = 10000

# Bandlimited FIR noise made with `legacy_level=True`, i.e. as psiaudio made it
# before its level was fixed. These keep exposures run the old way
# reproducible. They were first generated with psiaudio at eeb5b3d (before the
# short-lived `correct_level` option), so they are the old noise itself, not
# the current code's idea of it; regenerating them must not change them. Each
# entry is (fl, fh, equalize).
FIR_NOISE_CASES = {
    'fir_noise_golay_1p4k_2p8k_eq': (1.4e3, 2.8e3, True),
    'fir_noise_golay_1p4k_2p8k_noeq': (1.4e3, 2.8e3, False),
    'fir_noise_golay_2p8k_5p6k_eq': (2.8e3, 5.6e3, True),
    'fir_noise_golay_2p8k_5p6k_noeq': (2.8e3, 5.6e3, False),
    'fir_noise_golay_8k_16k_eq': (8e3, 16e3, True),
    'fir_noise_golay_8k_16k_noeq': (8e3, 16e3, False),
}


def fir_noise(fl, fh, equalize, **kwargs):
    factory = stim.BandlimitedFIRNoiseFactory(
        **FIR_NOISE_KW, fl=fl, fh=fh, equalize=equalize, **kwargs)
    return factory.next(FIR_NOISE_SAMPLES)


FIXTURES = {
    # ------------------------------------------------------------------
    # Tones
    # ------------------------------------------------------------------
    'tone_1kHz_100ms_fs100k': lambda: stim.tone(
        fs=100e3, frequency=1e3, level=1.0, duration=0.1),

    'tone_8kHz_50ms_polneg_fs100k': lambda: stim.tone(
        fs=100e3, frequency=8e3, level=1.0, duration=0.05, polarity=-1),

    # ------------------------------------------------------------------
    # Envelopes
    # ------------------------------------------------------------------
    'cos2envelope_200ms_rise10ms_fs100k': lambda: stim.cos2envelope(
        fs=100e3, duration=0.2, rise_time=10e-3),

    'gaussian_envelope_200ms_rise10ms_fs100k': lambda: stim.envelope(
        window='gaussian', fs=100e3, duration=0.2, rise_time=10e-3),

    # ------------------------------------------------------------------
    # Chirp
    # ------------------------------------------------------------------
    'chirp_1k_8k_10ms_fs100k': lambda: stim.chirp(
        fs=100e3, start_frequency=1e3, end_frequency=8e3,
        duration=0.01, level=1.0),

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    'broadband_noise_500ms_seed1_fs100k': lambda: stim.broadband_noise(
        fs=100e3, level=1.0, duration=0.5, seed=1),

    'bandlimited_noise_1k_8k_500ms_seed1_fs100k': lambda: stim.bandlimited_noise(
        fs=100e3, level=80, fl=1e3, fh=8e3, duration=0.5, seed=1,
        calibration=CAL),

    # ------------------------------------------------------------------
    # Gap stimuli (recently active in development)
    # ------------------------------------------------------------------
    # ``seed`` must be given explicitly for the noise-carrier cases: gap()
    # defaults to seed=None, which varies the noise token on every call, so an
    # unseeded fixture would never reproduce.
    'gap_noise_4k_1oct_5ms_2x50ms_fs100k': lambda: stim.gap(
        fs=100e3, fc=4e3, octaves=1, gap=5e-3, durations=[0.05, 0.05],
        rise_time=5e-3, level=80, calibration=CAL, seed=1),

    'gap_tone_4k_5ms_2x50ms_fs100k': lambda: stim.gap(
        fs=100e3, fc=4e3, octaves=0, gap=5e-3, durations=[0.05, 0.05],
        rise_time=5e-3, level=80, calibration=CAL),

    'gap_noise_4k_1oct_5ms_3x50ms_fs100k': lambda: stim.gap(
        fs=100e3, fc=4e3, octaves=1, gap=5e-3,
        durations=[0.05, 0.05, 0.05], rise_time=5e-3, level=80,
        calibration=CAL, seed=1),

    # ------------------------------------------------------------------
    # SAM envelope
    # ------------------------------------------------------------------
    'sam_envelope_depth1_fm100_eq_fs100k': lambda: stim.sam_envelope(
        offset=0, samples=10000, fs=100e3, depth=1.0, fm=100,
        delay=0, equalize=True),
}

# Bandlimited FIR noise through the Golay calibration, made the old way.
FIXTURES.update({
    name: (lambda case=case: fir_noise(*case, legacy_level=True))
    for name, case in FIR_NOISE_CASES.items()
})


if __name__ == '__main__':
    # Fixture names may be given to regenerate only those.
    names = sys.argv[1:] or list(FIXTURES)
    for name in names:
        waveform = FIXTURES[name]()
        path = FIXTURE_DIR / f'{name}.npy'
        np.save(path, waveform)
        print(f'Saved {path.name}  ({len(waveform)} samples)')
