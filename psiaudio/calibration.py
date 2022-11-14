import logging
log = logging.getLogger(__name__)

from importlib import resources

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

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

    def __init__(self, message):
        self.message = message

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
        Sensitivity of system in dB
    attrs : {None, dict}
        Information regarding the calibration (e.g., the filename the
        calibration was loaded from).
    '''
    def __init__(self, reference, attrs=None):
        self.attrs = attrs
        self.reference = reference
        if reference is not None:
            setattr(self, f'get_{reference.lower()}', self.get_db)

    def _get_db(self, frequency, voltage):
        sensitivity = self.get_sens(frequency)
        return util.db(voltage) + sensitivity

    def get_db(self, *args):
        '''
        Convert voltage into dB re reference unit for calibration (e.g., dB
        SPL).

        This function accepts one or two arguments. If one argument is
        provided, it must be a series or dataframe. For the series, the index
        must be frequency (in Hz). For the dataframe, the columns must be
        frequency (in Hz). This plays well with the output of other functions
        in this module (e.g., `~util.psd_df`).

        If two arguments are provided, the first argument must be frequency and
        the second voltage.
        '''
        if len(args) == 1:
            if isinstance(args[0], pd.DataFrame):
                frequency = args[0].columns.values
                voltage = args[0].values
                db = self._get_db(frequency, voltage)
                return pd.DataFrame(db, index=args[0].index,
                                    columns=args[0].columns)
            elif isinstance(args[0], pd.Series):
                frequency = args[0].index.values
                voltage = args[0].values
                db = self._get_db(frequency, voltage)
                return pd.Series(db, index=args[0].index)
            else:
                raise ValueError('Unsupported input arguments')
        elif len(args) == 2:
            return self._get_db(*args)
        else:
            raise ValueError('Unsupported input arguments')

    def get_sf(self, frequency, level, attenuation=0):
        sensitivity = self.get_sens(frequency)
        vdb = level - sensitivity + attenuation
        return 10**(vdb/20.0)

    def get_mean_sf(self, flb, fub, level, attenuation=0):
        frequencies = np.arange(flb, fub)
        sf = self.get_sf(frequencies, level).mean(axis=0)
        if np.isnan(sf):
            raise ValueError('Requested range has some uncalibrated frequencies')
        return sf

    def get_attenuation(self, frequency, voltage, level):
        return self.get_db(frequency, voltage)-level

    def get_gain(self, frequency, level, attenuation=0):
        return util.db(self.get_sf(frequency, level, attenuation))

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
        return cls.from_db(0, vrms, **kwargs)

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
    def from_db(cls, level, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded level (in dB)

        Parameters
        ----------
        levels : array-like
            List of magnitudes in dB re desired unit (e.g., speaker output in
            SPL) for the specified RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = level - util.db(vrms)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_mv_pa(cls, mv_pa, **kwargs):
        sens = util.db(1 / (mv_pa * 1e-3)) - util.db(20e-6)
        return cls(sensitivity=sens, reference='SPL', **kwargs)

    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        return cls.from_db(0, vrms, reference='dB re 1Vrms', **kwargs)

    @classmethod
    def from_spl(cls, spl, vrms=1, **kwargs):
        sensitivity = spl - util.db(vrms)
        return cls(sensitivity, reference='SPL', **kwargs)

    def __init__(self, sensitivity, fixed_gain=0, reference=None, attrs=None):
        super().__init__(reference, attrs)
        self.sensitivity = sensitivity
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        sens = self.sensitivity-self.fixed_gain
        if np.iterable(frequency):
            return np.full_like(frequency, fill_value=sens, dtype=np.double)
        return sens

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        return self.get_sf(flb, spl)


class BaseFrequencyCalibration(BaseCalibration):

    @classmethod
    def from_pascals(cls, frequency, magnitude, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded value (in Pascals)

        Parameters
        ----------
        magnitude : array-like
            List of magnitudes (e.g., speaker output in Pa) for the specified
            RMS voltage.
        frequency : {None, array-like}
            List of frequencies (in Hz)
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = util.db(vrms) - util.db(magnitude) - util.db(20e-6)
        return cls(frequency, sensitivity, **kwargs)

    @classmethod
    def from_db(cls, frequency, level, vrms=1, **kwargs):
        '''
        Generates a calibration object based on the recorded level in dB re.
        reference.

        Parameters
        ----------
        frequency : array-like
            List of frequencies (in Hz)
        level : array-like
            List of magnitudes in dB re. reference (e.g., speaker output in
            SPL) for the specified RMS voltage.
        vrms : float
            RMS voltage (in Volts)

        Additional kwargs are passed to the class initialization.
        '''
        sensitivity = level - util.db(vrms)
        return cls(frequency=frequency, sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_spl(cls, frequency, spl, vrms=1, **kwargs):
        return cls.from_db(frequency, spl, vrms, reference='SPL', **kwargs)


class InterpCalibration(BaseFrequencyCalibration):
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
    def __init__(self, frequency, sensitivity, fixed_gain=0, phase=None,
                 fill_value=np.nan, reference=None, attrs=None):
        super().__init__(reference, attrs)
        self.frequency = np.asarray(frequency)
        self.sensitivity = np.asarray(sensitivity)
        self.fixed_gain = fixed_gain
        self._interp = interp1d(frequency, sensitivity, 'linear',
                                bounds_error=False, fill_value=fill_value)
        if phase is not None:
            self.phase = np.asarray(phase)
            self._interp_phase = interp1d(frequency, phase, 'linear',
                                          bounds_error=False,
                                          fill_value=fill_value)
        else:
            self.phase = None
            self._interp_phase = None

    def get_sens(self, frequency):
        # Since sensitivity is in dB(V), subtracting fixed_gain from
        # sensitivity will *increase* the sensitivity of the system.
        frequency = np.asarray(frequency)
        return self._interp(frequency)-self.fixed_gain

    def get_phase(self, frequency):
        if self._interp_phase is None:
            raise ValueError('No phase correction data available')
        return self._interp_phase(frequency)


class PointCalibration(BaseFrequencyCalibration):

    def __init__(self, frequency, sensitivity, fixed_gain=0, reference=None,
                 attrs=None):
        super().__init__(reference, attrs)
        if np.isscalar(frequency):
            frequency = [frequency]
        if np.isscalar(sensitivity):
            sensitivity = [sensitivity]
        self.frequency = np.array(frequency)
        self.sensitivity = np.array(sensitivity)
        self.fixed_gain = fixed_gain

        # Needed to enable vectorizing instance methods. The decorator approach
        # does not work.
        self.get_sens = np.vectorize(self._get_sens)

    def _get_sens(self, frequency):
        try:
            i = np.flatnonzero(np.equal(self.frequency, frequency))[0]
            return self.sensitivity[i]-self.fixed_gain
        except IndexError:
            log.debug('Calibrated frequencies are %r', self.frequency)
            raise CalibrationError(f'{frequency} Hz not calibrated')


if __name__ == '__main__':
    import doctest
    doctest.testmod()


################################################################################
# Example calibrations
################################################################################
def load_demo_starship():
    cal_file = resources.files('psiaudio.resources') / 'starship_cal.csv'
    with resources.as_file(cal_file) as fh:
        cal = pd.read_csv(fh)
        return InterpCalibration(cal['freq'], cal['SPL'], phase=cal['phase'])
