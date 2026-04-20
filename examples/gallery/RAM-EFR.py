'''
====================
Generating a RAM EFR
====================
'''
import matplotlib.pyplot as plt
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import SquareWaveEnvelopeFactory, ToneFactory


calibration = FlatCalibration.unity()

fs = 100e3
fm = 110
fc = 8e3
depth = 1
duty_cycle = 0.25
tukey_window = 0.025

duration = 2/fm

factory = SquareWaveEnvelopeFactory(
    fs=fs,
    fm=fm,
    depth=depth,
    duty_cycle=duty_cycle,
    calibration=calibration,
    alpha=tukey_window,
    input_factory=ToneFactory(
        fs=fs,
        frequency=fc,
        level=1,
        calibration=calibration,
        phase=-np.pi/2,
    )
)

n_samples = round(duration * 0.025 * fs)


figure, axes = plt.subplots(3, 1, figsize=(6, 6), constrained_layout=True,
                            sharex=True, sharey=True)

to_merge = []
for i in range(100):
    s = factory.next(n_samples)
    t = (n_samples * i + np.arange(n_samples)) / fs
    axes[0].plot(t, s, '-')
    to_merge.append(s)
s_merged = np.concatenate(to_merge, axis=-1)

factory.reset()
s = factory.next(n_samples * 100)
t = np.arange(n_samples * 100) / fs
axes[1].plot(t, s, 'k-', zorder=-1)

axes[2].plot(t, s - s_merged, 'k-')
axes[0].set_title('RAM EFR incrementially generated')
axes[1].set_title('RAM EFR generated as a full section')
axes[2].set_title('Difference between incremential and full generation')

for ax in axes:
    ax.set_ylabel('Signal (V)')
axes[2].set_xlabel('Time (s)')

plt.show()
