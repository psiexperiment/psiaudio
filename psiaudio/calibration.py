import logging
log = logging.getLogger(__name__)

from scipy.interpolate import interp1d
import numpy as np

from . import util


################################################################################
# Exceptions
################################################################################
mesg = '''
Unable to run the calibration. Please double-check that the microphone and
speaker amplifiers are powered on and that the devices are positioned properly.
If you keep receiving this message, the microphone and/or the speaker may have
gone bad and need to be replaced.

{}
'''
mesg = mesg.strip()


thd_err_mesg = 'Total harmonic distortion for {:.1f}Hz is {:.1f}%'
nf_err_mesg = 'Power at {:.1f}Hz has SNR of {:.2f}dB'


class CalibrationError(Exception):

    def __str__(self):
        return self.message


class CalibrationTHDError(CalibrationError):

    def __init__(self, frequency, thd):
        self.frequency = frequency
        self.thd = thd
        self.base_message = thd_err_mesg.format(frequency, thd)
        self.message = mesg.format(self.base_message)


class CalibrationNFError(CalibrationError):

    def __init__(self, frequency, snr):
        self.frequency = frequency
        self.snr = snr
        self.base_message = nf_err_mesg.format(frequency, snr)
        self.message = mesg.format(self.base_message)


################################################################################
# Calibration routines
################################################################################
class BaseCalibration:
    '''
    Assumes that the system is linear for a given frequency

    Parameters
    ----------
    frequency : 1D array
        Frequencies that system sensitivity was measured at.
    sensitivity : 1D array
        Sensitivity of system in dB(V/Pa).
    '''

    def get_spl(self, frequency, voltage):
        sensitivity = self.get_sens(frequency)
        return util.db(voltage) + sensitivity

    def get_sf(self, frequency, spl, attenuation=0):
        sensitivity = self.get_sens(frequency)
        vdb = spl - sensitivity + attenuation
        return 10**(vdb/20.0)

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        frequencies = np.arange(flb, fub)
        return self.get_sf(frequencies, spl).mean(axis=0)

    def get_attenuation(self, frequency, voltage, level):
        return self.get_spl(frequency, voltage)-level

    def get_gain(self, frequency, spl, attenuation=0):
        return util.db(self.get_sf(frequency, spl, attenuation))

    def set_fixed_gain(self, fixed_gain):
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        raise NotImplementedError


class FlatCalibration(BaseCalibration):

    @property
    def _get_mv_pa(self):
        return util.dbi(self.sensitivity) * 1e3

    @classmethod
    def unity(cls):
        '''
        Passthrough calibration allowing signal level to be specified in Vrms.
        '''
        return cls(sensitivity=0)

    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        '''
        Allows levels to be specified in dB attenuation
        '''
        return cls.from_spl(0, vrms, **kwargs)

    @classmethod
    def from_pascals(cls, magnitude, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded value (in Pascals)

        Parameters
        ----------
        frequency : array-like
            List of freuquencies (in Hz)
        magnitude : array-like
            List of magnitudes (e.g., speaker output in Pa) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms) - util.db(magnitude) - util.db(20e-6)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_spl(cls, spl, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded SPL

        Parameters
        ----------
        spl : array-like
            List of magnitudes (e.g., speaker output in SPL) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = spl - util.db(vrms)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_mv_pa(cls, mv_pa, **kwargs):
        sens = util.db(mv_pa * 1e-3)
        return cls(sensitivity=sens, **kwargs)

    @classmethod
    def as_attenuation(cls, vrms=1):
        return cls.from_spl(0, vrms, **kwargs)

    @classmethod
    def from_spl(cls, spl, vrms=1, **kwargs):
        sensitivity = spl - util.db(vrms)
        return cls(sensitivity, **kwargs)

    def __init__(self, sensitivity, fixed_gain=0):
        self.sensitivity = sensitivity
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        return self.sensitivity-self.fixed_gain

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        return self.get_sf(flb, spl)


class InterpCalibration(BaseCalibration):
    '''
    Use when calibration is not flat (i.e., uniform) across frequency.

    Parameters
    ----------
    frequency : array-like, Hz
        Calibrated frequencies (in Hz)
    sensitivity : array-like, dB(V/Pa)
        Sensitivity at calibrated frequency in dB(V/Pa) assuming 1 Vrms and 0 dB
        gain.  If you have sensitivity in V/Pa, just pass it in as
        20*np.log10(sens).
    fixed_gain : float
        Fixed gain of the input or output.  The sensitivity is calculated using
        a gain of 0 dB, so if the input (e.g. a microphone preamp) or output
        (e.g. a speaker amplifier) adds a fixed gain, this needs to be factored
        into the calculation.

        For input calibrations, the gain must be negative (e.g. if the
        microphone amplifier is set to 40 dB gain, then provide -40 as the
        value).
    '''
    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        '''
        Allows levels to be specified in dB attenuation
        '''
        return cls.from_spl([0, 100e3], [0, 0], vrms, **kwargs)

    @classmethod
    def from_pascals(cls, frequency, magnitude, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded value (in Pascals)

        Parameters
        ----------
        frequency : array-like
            List of frequencies (in Hz)
        magnitude : array-like
            List of magnitudes (e.g., speaker output in Pa) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms) - util.db(magnitude) - util.db(20e-6)
        return cls(frequency, sensitivity, **kwargs)

    @classmethod
    def from_spl(cls, frequency, spl, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded SPL

        Parameters
        ----------
        frequency : array-like
            List of freuquencies (in Hz)
        spl : array-like
            List of magnitudes (e.g., speaker output in SPL) for the specified
            RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = spl - util.db(vrms)
        return cls(frequency, sensitivity, **kwargs)

    def __init__(self, frequency, sensitivity, fixed_gain=0):
        self.frequency = np.asarray(frequency)
        self.sensitivity = np.asarray(sensitivity)
        self.fixed_gain = fixed_gain
        self._interp = interp1d(frequency, sensitivity, 'linear',
                                bounds_error=False)

    def get_sens(self, frequency):
        # Since sensitivity is in dB(V), subtracting fixed_gain from
        # sensitivity will *increase* the sensitivity of the system.
        return self._interp(frequency)-self.fixed_gain


class PointCalibration(BaseCalibration):

    def __init__(self, frequency, sensitivity, fixed_gain=0):
        if np.isscalar(frequency):
            frequency = [frequency]
        if np.isscalar(sensitivity):
            sensitivity = [sensitivity]
        self.frequency = np.array(frequency)
        self.sensitivity = np.array(sensitivity)
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        if np.iterable(frequency):
            return np.array([self._get_sens(f) for f in frequency])
        else:
            return self._get_sens(frequency)

    def _get_sens(self, frequency):
        try:
            i = np.flatnonzero(np.equal(self.frequency, frequency))[0]
        except IndexError:
            log.debug('Calibrated frequencies are %r', self.frequency)
            m = 'Frequency {} not calibrated'.format(frequency)
            raise CalibrationError(m)
        return self.sensitivity[i]-self.fixed_gain


if __name__ == '__main__':
    import doctest
    doctest.testmod()
