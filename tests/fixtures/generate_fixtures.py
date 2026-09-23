"""
Regenerate golden fixture files for stim regression tests.

Run from the project root:
    python tests/fixtures/generate_fixtures.py

Re-run this whenever stim.py is intentionally changed in a way that alters
the generated waveforms.  The regenerated .npy files should be committed
alongside the code change so the intent is visible in the diff.
"""
from pathlib import Path

import numpy as np

from psiaudio import calibration, stim

FIXTURE_DIR = Path(__file__).parent
CAL = calibration.FlatCalibration.from_spl(94)


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


if __name__ == '__main__':
    for name, fn in FIXTURES.items():
        waveform = fn()
        path = FIXTURE_DIR / f'{name}.npy'
        np.save(path, waveform)
        print(f'Saved {path.name}  ({len(waveform)} samples)')
