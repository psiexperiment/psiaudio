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

#coerced_fc = fm * round(fc / fm)
#coerced_duty_cycle = round(duty_cycle / fm * coerced_fc) / coerced_fc * fm

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

samples = factory.next(round(duration * fs))
plt.plot(samples)
plt.show()
