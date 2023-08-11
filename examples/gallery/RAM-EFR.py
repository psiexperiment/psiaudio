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

n_samples = round(duration * 0.1 * fs)

for i in range(10):
    print(f'----- {i} -----')
    s = factory.next(n_samples)
    t = (n_samples * i + np.arange(n_samples)) * fs
    plt.plot(t, s, '.-')

factory.reset()
s = factory.next(n_samples * 10)
t = np.arange(n_samples * 10) * fs
plt.plot(t, s, 'k-', zorder=-1)

plt.show()
