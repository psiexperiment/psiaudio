'''
Generate bandlimited clicks
===========================

This demonstrates how to generate a bandlimited click of uniform SPL when you
have an acoustic system whose output varies with frequency (e.g., an EPL-style
starship).
'''
import matplotlib.pyplot as plt

from psiaudio import calibration
from psiaudio import stim
from psiaudio import util

figure, axes = plt.subplots(1, 2, figsize=(8, 4))

#flat_cal = calibration.FlatCalibration.from_spl(80)
cal = calibration.load_demo_starship()
fs = 100e3
window = 5e-3

for flb, fub in [(2e3, 32e3), (8e3, 32e3)]:
    for level in (60, 40):
        waveform = stim.bandlimited_click(
            fs=fs,
            flb=flb,
            fub=fub,
            window=window,
            level=level,
            calibration=cal,
            equalize=True,
        )

        psd = util.psd_df(waveform, fs)
        spl = cal.get_db(psd)
        psd_spl = util.rms_rfft_db(spl)
        pe_spl = cal.get_db(1e3, waveform.ptp())
        print(f'{level} {pe_spl} {psd_spl}')

        axes[0].plot(waveform)
        p, = axes[1].plot(spl)

        sl = util.band_to_spectrum_level(level, (fub-flb)*window)
        axes[1].axhline(sl, color=p.get_color(), ls=':')

axes[1].axis(xmin=0.5e3, xmax=50e3)
axes[1].set_xscale('octave')
plt.show()
