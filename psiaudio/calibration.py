"""
Audio Calibration Module for Psychoacoustic Experiments
======================================================

This module provides tools for calibrating audio systems used in psychoacoustic
and auditory research. It converts between electrical signals (voltages) and
acoustic quantities (sound pressure levels, Pascals) by accounting for the
frequency-dependent characteristics of microphones, speakers, and amplifiers.

Overview
--------
In auditory research, precise control of acoustic stimuli requires understanding
the relationship between digital signals and the actual sound pressure levels
they produce. This module handles that conversion through calibration objects
that encapsulate the system's transfer function.

The calibration process typically involves:
1. Playing known test signals through your audio system
2. Measuring the acoustic output with a calibrated microphone
3. Computing the system's sensitivity (dB SPL per Volt) at each frequency
4. Using this calibration to convert arbitrary signals to desired SPL levels

Calibration Classes
------------------
Choose the appropriate calibration class based on your system characteristics:

**FlatCalibration**
    Use when your audio system has approximately uniform frequency response
    or when frequency-specific calibration is not critical. Examples:
    - High-quality studio monitors in treated rooms
    - Systems where stimuli are pre-equalized
    - Quick prototyping or when measurement precision is not critical

    Advantages: Simple, fast, minimal calibration data required
    Disadvantages: May introduce frequency-dependent errors

**InterpCalibration**
    Use when your system has significant frequency-dependent characteristics
    that need correction. This is the most common choice for research. Examples:
    - Headphones with non-flat frequency response
    - In-ear monitors or insert earphones
    - Speaker systems without acoustic treatment
    - Any system requiring precise SPL control across frequency

    Advantages: High accuracy, supports equalization filtering, handles phase
    Disadvantages: Requires comprehensive calibration measurements

**PointCalibration**
    Use when calibration is only valid at specific discrete frequencies
    with no interpolation desired. Examples:
    - Systems calibrated only for pure tones at specific frequencies
    - When interpolation might introduce artifacts
    - Legacy calibration data with sparse frequency sampling

    Advantages: No interpolation artifacts, exact at calibrated frequencies
    Disadvantages: Limited to specific frequencies, no equalization support

Typical Workflow
---------------
1. **System Setup**: Position microphone and speakers, configure amplifiers
2. **Calibration Measurement**: Play calibrated test signals, record responses
3. **Create Calibration Object**: Use factory methods (from_spl, from_pascals, etc.)
4. **Apply Calibration**: Convert signals to desired SPL levels
5. **Optional Equalization**: Generate FIR filters to flatten response

Examples
--------
>>> # Create calibration from SPL measurements
>>> frequencies = np.array([100, 500, 1000, 2000, 8000])  # Hz
>>> spl_measured = np.array([92.1, 94.3, 94.0, 93.8, 89.2])  # dB SPL for 1 Vrms
>>> cal = InterpCalibration.from_spl(frequencies, spl_measured)

>>> # Convert 0.1 Vrms signal at 1 kHz to SPL
>>> spl = cal.get_db(1000, 0.1)  # Returns SPL in dB

>>> # Get scaling factor for 70 dB SPL at 1 kHz
>>> sf = cal.get_sf(1000, 70)  # Multiply your signal by this factor

>>> # Generate equalization filter
>>> eq_filt, zi = cal.make_eq_filter(fs=48000, fl=100, fh=8000)

Notes
-----
- All frequency values are in Hz
- Voltage values are RMS (not peak)
- SPL reference is 20 μPa (standard acoustic reference)
- Negative fixed_gain values account for microphone preamp gain
- Phase corrections are supported in InterpCalibration

See Also
--------
util : Utility functions for dB conversions and signal processing
stim : Stimulus generation and processing functions
"""

import logging
log = logging.getLogger(__name__)

from importlib import resources

from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
import pandas as pd

from . import util
from .stim import apply_max_correction


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
    """
    Base exception class for calibration-related errors.

    Parameters
    ----------
    message : str
        Error message describing the calibration issue.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class CalibrationTHDError(CalibrationError):
    """
    Exception raised when total harmonic distortion exceeds acceptable limits.

    Parameters
    ----------
    frequency : float
        Frequency at which THD error occurred, in Hz.
    thd : float
        Total harmonic distortion percentage that exceeded limits.
    """

    def __init__(self, frequency, thd):
        self.frequency = frequency
        self.thd = thd
        self.base_message = thd_err_mesg.format(frequency, thd)
        self.message = mesg.format(self.base_message)


class CalibrationNFError(CalibrationError):
    """
    Exception raised when signal-to-noise ratio is insufficient.

    Parameters
    ----------
    frequency : float
        Frequency at which SNR error occurred, in Hz.
    snr : float
        Signal-to-noise ratio in dB that was insufficient.
    """

    def __init__(self, frequency, snr):
        self.frequency = frequency
        self.snr = snr
        self.base_message = nf_err_mesg.format(frequency, snr)
        self.message = mesg.format(self.base_message)


################################################################################
# Calibration routines
################################################################################
class BaseCalibration:
    """
    Base class for audio system calibration assuming linearity for given frequencies.

    This class provides the foundation for calibrating audio systems by converting
    between voltage measurements and calibrated units (e.g., dB SPL). It assumes
    the system behaves linearly at each frequency.

    Parameters
    ----------
    reference : str or None
        Reference unit for calibration (e.g., 'SPL', 'Pa'). If provided,
        creates a dynamic method get_{reference.lower()} that calls get_db.
    attrs : dict, optional
        Additional information about the calibration such as filename or
        measurement conditions. Default is None.

    Attributes
    ----------
    attrs : dict or None
        Calibration metadata.
    reference : str or None
        Reference unit for the calibration.
    """

    def __init__(self, reference, attrs=None):
        self.attrs = attrs
        self.reference = reference
        if reference is not None:
            setattr(self, f'get_{reference.lower()}', self.get_db)

    def _get_db(self, frequency, voltage):
        """
        Internal method to convert voltage to dB re reference unit.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.
        voltage : array_like
            Voltage measurements in Vrms.

        Returns
        -------
        ndarray
            Values in dB re reference unit.
        """
        sensitivity = self.get_sens(frequency)
        return util.db(voltage) + sensitivity

    def get_db(self, *args):
        """
        Convert voltage into dB re reference unit for calibration.

        This function accepts one or two arguments:
        - One argument: pandas Series (index=frequency) or DataFrame (columns=frequency)
        - Two arguments: frequency array and voltage array

        Parameters
        ----------
        *args : tuple
            Either (data,) where data is Series/DataFrame, or (frequency, voltage).

            If one argument:
            - pandas.Series with frequency as index and voltage as values
            - pandas.DataFrame with frequency as columns and voltage as values

            If two arguments:
            - frequency : array_like
                Frequencies in Hz
            - voltage : array_like
                Voltage measurements in Vrms

        Returns
        -------
        pandas.Series, pandas.DataFrame, or ndarray
            Values in dB re reference unit. Return type matches input type.

        Raises
        ------
        ValueError
            If unsupported input arguments are provided.

        Examples
        --------
        >>> cal = SomeCalibration()
        >>> # Using Series
        >>> freq_series = pd.Series([0.1, 0.2], index=[1000, 2000])
        >>> db_series = cal.get_db(freq_series)
        >>> # Using arrays
        >>> db_values = cal.get_db([1000, 2000], [0.1, 0.2])
        """
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
        """
        Get scaling factor to achieve target level at specified frequency.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.
        level : array_like
            Target levels in dB re reference unit.
        attenuation : array_like, optional
            Additional attenuation in dB. Default is 0.

        Returns
        -------
        ndarray
            Scaling factors (dimensionless) to multiply signal by.

        Examples
        --------
        >>> cal = SomeCalibration()
        >>> sf = cal.get_sf(1000, 80)  # Get scaling for 80 dB SPL at 1 kHz
        """
        sensitivity = self.get_sens(frequency)
        vdb = level - sensitivity + attenuation
        return 10**(vdb/20.0)

    def get_attenuation(self, frequency, voltage, level):
        """
        Calculate attenuation needed to achieve target level.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.
        voltage : array_like
            Current voltage in Vrms.
        level : array_like
            Target level in dB re reference unit.

        Returns
        -------
        ndarray
            Required attenuation in dB.
        """
        return self.get_db(frequency, voltage)-level

    def get_gain(self, frequency, level, attenuation=0):
        """
        Get gain in dB needed to achieve target level.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.
        level : array_like
            Target level in dB re reference unit.
        attenuation : array_like, optional
            Additional attenuation in dB. Default is 0.

        Returns
        -------
        ndarray
            Required gain in dB.
        """
        return util.db(self.get_sf(frequency, level, attenuation))

    def set_fixed_gain(self, fixed_gain):
        """
        Set fixed gain applied by amplifiers in the signal chain.

        Parameters
        ----------
        fixed_gain : float
            Fixed gain in dB. Negative values indicate attenuation.
        """
        self.fixed_gain = fixed_gain

    def get_sens(self, frequency):
        """
        Get system sensitivity at specified frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.

        Returns
        -------
        ndarray
            Sensitivity values in dB.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError


class FlatCalibration(BaseCalibration):
    """
    Calibration with frequency-independent (flat) sensitivity.

    This calibration assumes the system has uniform sensitivity across
    all frequencies, making it suitable for systems with flat frequency
    response or when frequency-specific calibration is not needed.

    Parameters
    ----------
    sensitivity : float
        System sensitivity in dB (typically dB re 1V for the reference unit).
    fixed_gain : float, optional
        Fixed gain of amplifiers in signal chain, in dB. Default is 0.
    reference : str, optional
        Reference unit (e.g., 'SPL', 'Pa'). Default is None.
    attrs : dict, optional
        Additional calibration metadata. Default is None.

    Attributes
    ----------
    sensitivity : float
        System sensitivity in dB.
    fixed_gain : float
        Fixed gain applied by amplifiers.
    """

    @property
    def _get_mv_pa(self):
        """Get sensitivity in mV/Pa units."""
        return util.dbi(self.sensitivity) * 1e3

    @classmethod
    def unity(cls):
        """
        Create passthrough calibration for signal levels in Vrms.

        Returns
        -------
        FlatCalibration
            Unity calibration object with 0 dB sensitivity.

        Examples
        --------
        >>> cal = FlatCalibration.unity()
        >>> # Signal levels will be interpreted as Vrms
        """
        return cls(sensitivity=0)

    @classmethod
    def as_attenuation(cls, vrms=1, **kwargs):
        """
        Create calibration for specifying levels as dB attenuation.

        Parameters
        ----------
        vrms : float, optional
            Reference RMS voltage in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        FlatCalibration
            Calibration object for attenuation-based level specification.
        """
        return cls.from_db(0, vrms, **kwargs)

    @classmethod
    def from_pascals(cls, magnitude, vrms=1, **kwargs):
        """
        Create calibration from recorded pressure values.

        Parameters
        ----------
        magnitude : array_like
            Measured magnitudes in Pascals for the specified RMS voltage.
        vrms : float, optional
            RMS voltage used during measurement, in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        FlatCalibration
            Calibration object based on Pascal measurements.

        Examples
        --------
        >>> # Measured 0.1 Pa output for 1 Vrms input
        >>> cal = FlatCalibration.from_pascals(0.1, vrms=1)
        """
        sensitivity = util.db(vrms) - util.db(magnitude) - util.db(20e-6)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_db(cls, level, vrms=1, **kwargs):
        """
        Create calibration from recorded dB levels.

        Parameters
        ----------
        level : array_like
            Measured levels in dB re desired reference unit for specified RMS voltage.
        vrms : float, optional
            RMS voltage used during measurement, in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        FlatCalibration
            Calibration object based on dB level measurements.

        Examples
        --------
        >>> # System produces 94 dB SPL for 1 Vrms input
        >>> cal = FlatCalibration.from_db(94, vrms=1, reference='SPL')
        """
        sensitivity = level - util.db(vrms)
        return cls(sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_mv_pa(cls, mv_pa, **kwargs):
        """
        Create calibration from microphone sensitivity in mV/Pa.

        Parameters
        ----------
        mv_pa : float
            Microphone sensitivity in mV/Pa.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        FlatCalibration
            Calibration object with reference set to 'SPL'.
        """
        sens = util.db(1 / (mv_pa * 1e-3)) - util.db(20e-6)
        return cls(sensitivity=sens, reference='SPL', **kwargs)

    def to_mv_pa(self):
        """
        Convert sensitivity to mV/Pa units.

        Returns
        -------
        float
            Sensitivity in mV/Pa.
        """
        return 1e3 / util.dbi(self.sensitivity + util.db(20e-6))

    @classmethod
    def from_spl(cls, spl, vrms=1, **kwargs):
        """
        Create calibration from SPL measurements.

        Parameters
        ----------
        spl : float
            Measured sound pressure level in dB SPL.
        vrms : float, optional
            RMS voltage used during measurement. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        FlatCalibration
            Calibration object with SPL reference.
        """
        sensitivity = spl - util.db(vrms)
        return cls(sensitivity, reference='SPL', **kwargs)

    def __init__(self, sensitivity, fixed_gain=0, reference=None, attrs=None):
        super().__init__(reference, attrs)
        self.sensitivity = sensitivity
        self.fixed_gain = fixed_gain

    def get_level(self, voltage):
        """
        Convert voltage to calibrated physical units.

        Parameters
        ----------
        voltage : array_like
            Voltage measurements in Vrms.

        Returns
        -------
        ndarray
            Levels in physical units (e.g., Pascals if reference is SPL).
        """
        sensitivity = self.get_sens(1e3)
        return voltage * util.dbi(sensitivity)

    def get_sens(self, frequency):
        """
        Get system sensitivity (frequency-independent for flat calibration).

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz (ignored for flat calibration).

        Returns
        -------
        float or ndarray
            Sensitivity value(s) in dB, broadcast to match frequency shape.
        """
        sens = self.sensitivity-self.fixed_gain
        if np.iterable(frequency):
            return np.full_like(frequency, fill_value=sens, dtype=np.double)
        return sens

    def get_mean_sf(self, flb, fub, spl, attenuation=0):
        """
        Get mean scaling factor over frequency band (same as single frequency for flat calibration).

        Parameters
        ----------
        flb : float
            Lower frequency bound in Hz (ignored for flat calibration).
        fub : float
            Upper frequency bound in Hz (ignored for flat calibration).
        spl : float
            Target level in dB re reference unit.
        attenuation : float, optional
            Additional attenuation in dB. Default is 0.

        Returns
        -------
        float
            Scaling factor for the specified level.
        """
        return self.get_sf(flb, spl)


class BaseFrequencyCalibration(BaseCalibration):
    """
    Base class for frequency-dependent calibrations.

    This class provides common functionality for calibrations that vary
    with frequency, including factory methods for creating calibrations
    from different measurement types.
    """

    @classmethod
    def from_pascals(cls, frequency, magnitude, vrms=1, **kwargs):
        """
        Create calibration from pressure measurements at specific frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies where measurements were taken, in Hz.
        magnitude : array_like
            Measured magnitudes in Pascals for the specified RMS voltage.
        vrms : float, optional
            RMS voltage used during measurements, in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        BaseFrequencyCalibration subclass
            Calibration object based on Pascal measurements.
        """
        sensitivity = util.db(vrms) - util.db(magnitude) - util.db(20e-6)
        return cls(frequency, sensitivity, **kwargs)

    @classmethod
    def from_db(cls, frequency, level, vrms=1, **kwargs):
        """
        Create calibration from dB level measurements at specific frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies where measurements were taken, in Hz.
        level : array_like
            Measured levels in dB re reference unit for the specified RMS voltage.
        vrms : float, optional
            RMS voltage used during measurements, in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        BaseFrequencyCalibration subclass
            Calibration object based on dB level measurements.
        """
        sensitivity = level - util.db(vrms)
        return cls(frequency=frequency, sensitivity=sensitivity, **kwargs)

    @classmethod
    def from_spl(cls, frequency, spl, vrms=1, **kwargs):
        """
        Create calibration from SPL measurements at specific frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies where measurements were taken, in Hz.
        spl : array_like
            Measured sound pressure levels in dB SPL.
        vrms : float, optional
            RMS voltage used during measurements, in Volts. Default is 1.
        **kwargs
            Additional arguments passed to class constructor.

        Returns
        -------
        BaseFrequencyCalibration subclass
            Calibration object with SPL reference.
        """
        return cls.from_db(frequency, spl, vrms, reference='SPL', **kwargs)


class InterpCalibration(BaseFrequencyCalibration):
    """
    Calibration with frequency-dependent sensitivity using interpolation.

    This class handles calibrations where sensitivity varies with frequency
    by interpolating between measured calibration points. Supports both
    magnitude and phase corrections.

    Parameters
    ----------
    frequency : array_like
        Calibrated frequencies in Hz.
    sensitivity : array_like
        Sensitivity at calibrated frequencies in dB, assuming 1 Vrms and 0 dB gain.
    fixed_gain : float, optional
        Fixed gain of amplifiers in signal chain, in dB. For input calibrations
        (e.g., microphone preamps), use negative values. Default is 0.
    phase : array_like, optional
        Phase response at calibrated frequencies in radians. Default is None.
    reference : str, optional
        Reference unit (e.g., 'SPL', 'Pa'). Default is None.
    attrs : dict, optional
        Additional calibration metadata. Default is None.

    Attributes
    ----------
    frequency : ndarray
        Calibrated frequencies, rounded to 2 decimal places.
    sensitivity : ndarray
        Sensitivity values at calibrated frequencies.
    fixed_gain : float
        Fixed gain applied by amplifiers.
    phase : ndarray or None
        Phase response data if provided.
    """

    def __init__(self, frequency, sensitivity, fixed_gain=0, phase=None,
                 reference=None, attrs=None):
        super().__init__(reference, attrs)
        self.frequency = np.asarray(frequency).round(2)
        self.sensitivity = np.asarray(sensitivity)
        self.fixed_gain = fixed_gain
        self._interp = interp1d(frequency, sensitivity, 'linear',
                                bounds_error=False, fill_value='extrapolate')
        if phase is not None:
            self.phase = np.asarray(phase)
            self._interp_phase = interp1d(frequency, phase, 'linear',
                                          bounds_error=False,
                                          fill_value='extrapolate')
        else:
            self.phase = None
            self._interp_phase = None

    def get_sens(self, frequency):
        """
        Get interpolated sensitivity at specified frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.

        Returns
        -------
        ndarray
            Interpolated sensitivity values in dB.

        Raises
        ------
        ValueError
            If requested frequencies are outside the calibrated range.
        """
        # Since sensitivity is in dB(V), subtracting fixed_gain from
        # sensitivity will *increase* the sensitivity of the system.
        frequency = np.asarray(frequency, dtype=self.frequency.dtype)
        f_min = self.frequency.min()
        f_max = self.frequency.max()
        m = (frequency < f_min) | (frequency > f_max)
        if m.any():
            raise ValueError('Requested range has some uncalibrated frequencies. '
                             f'Requested {frequency[m]} Hz. Calibrated {f_min} to {f_max} Hz.')
        return self._interp(frequency)-self.fixed_gain

    def get_phase(self, frequency):
        """
        Get interpolated phase response at specified frequencies.

        Parameters
        ----------
        frequency : array_like
            Frequencies in Hz.

        Returns
        -------
        ndarray
            Interpolated phase values in radians.

        Raises
        ------
        ValueError
            If no phase correction data is available.
        """
        if self._interp_phase is None:
            raise ValueError('No phase correction data available')
        return self._interp_phase(frequency)

    def make_eq_filter(self, fs, fl=None, fh=None, window='hann', ntaps=1001,
                       max_correction=30, target_level=80, target_rms=1):
        """
        Generate FIR equalization filter to flatten frequency response.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        fl : float, optional
            Lower frequency limit in Hz. Defaults to minimum calibrated frequency.
        fh : float, optional
            Upper frequency limit in Hz. Defaults to maximum calibrated frequency.
        window : str, optional
            Window function for FIR filter design. Default is 'hann'.
        ntaps : int, optional
            Number of filter taps. Default is 1001.
        max_correction : float, optional
            Maximum correction in dB to prevent excessive amplification.
            Default is 30. Set to None to disable limiting.
        target_level : float, optional
            Target level in dB re reference unit. Default is 80.
        target_rms : float, optional
            Target RMS voltage. Default is 1.

        Returns
        -------
        tuple
            (filt, zi) where filt is the FIR filter coefficients and zi is
            the initial conditions for lfilter.

        Examples
        --------
        >>> cal = InterpCalibration(freq, sens)
        >>> filt, zi = cal.make_eq_filter(fs=48000, fl=100, fh=10000)
        >>> # Apply filter to equalize signal
        >>> eq_signal, _ = scipy.signal.lfilter(filt, [1], signal, zi=zi)
        """
        if fl is None:
            fl = self.frequency.min()
        if fh is None:
            fh = self.frequency.max()

        freq = np.arange(fl, fh+1)
        sf = self.get_sf(freq, target_level - util.db(target_rms))

        if max_correction is not None:
            sf = apply_max_correction(sf, max_correction)

        if fl == 1:
            freq = np.concatenate(([0], freq))
            sf = np.pad(sf, (1, 0))
        elif fl > 1:
            freq = np.concatenate(([0, fl], freq))
            sf = np.pad(sf, (2, 0))

        if fh == ((fs / 2) - 1):
            freq = np.concatenate((freq, [fs/2]))
            sf = np.pad(sf, (0, 1))
        if fh < ((fs / 2) - 1):
            freq = np.concatenate((freq, [fh, fs/2]))
            sf = np.pad(sf, (0, 2))

        filt = signal.firwin2(ntaps, freq=freq, gain=sf, window=window, fs=fs)
        zi = signal.lfilter_zi(filt, [1])
        return filt, zi

    def get_mean_sf(self, flb, fub, level, attenuation=0):
        """
        Get mean scaling factor over a frequency band.

        Parameters
        ----------
        flb : float
            Lower frequency bound in Hz (inclusive).
        fub : float
            Upper frequency bound in Hz (inclusive).
        level : float
            Target level in dB re reference unit.
        attenuation : float, optional
            Additional attenuation in dB. Default is 0.

        Returns
        -------
        float
            Mean scaling factor over the frequency band.

        Raises
        ------
        ValueError
            If the requested frequency range extends beyond calibrated range.
        """
        frequencies = np.arange(flb, fub)
        sf = self.get_sf(frequencies, level)
        if (flb < self.frequency.min()) or (fub > self.frequency.max()):
            f_min = self.frequency.min()
            f_max = self.frequency.max()
            raise ValueError('Requested range has some uncalibrated frequencies. '
                             f'Requested {flb} to {fub} Hz. Calibrated {f_min} to {f_max} Hz.')

        sf_mean = sf.mean(axis=0)
        return sf_mean


class PointCalibration(BaseFrequencyCalibration):
    """
    Calibration with discrete frequency points (no interpolation).

    This calibration only provides sensitivity values at specific measured
    frequencies without interpolation. Useful when calibration is only
    valid at discrete frequencies or when interpolation is not desired.

    Parameters
    ----------
    frequency : array_like
        Calibrated frequencies in Hz.
    sensitivity : array_like
        Sensitivity values at calibrated frequencies in dB.
    fixed_gain : float, optional
        Fixed gain of amplifiers in signal chain, in dB. Default is 0.
    reference : str, optional
        Reference unit (e.g., 'SPL', 'Pa'). Default is None.
    attrs : dict, optional
        Additional calibration metadata. Default is None.

    Attributes
    ----------
    frequency : ndarray
        Discrete calibrated frequencies.
    sensitivity : ndarray
        Sensitivity values at calibrated frequencies.
    fixed_gain : float
        Fixed gain applied by amplifiers.
    """

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
        """
        Get sensitivity at exact frequency match (internal method).

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        float
            Sensitivity value in dB.

        Raises
        ------
        CalibrationError
            If the requested frequency is not calibrated.
        """
        try:
            i = np.flatnonzero(np.equal(self.frequency, frequency))[0]
            return self.sensitivity[i]-self.fixed_gain
        except IndexError:
            log.debug('Calibrated frequencies are %r', self.frequency)
            raise CalibrationError(f'{frequency} Hz not calibrated')

    def get_mean_sf(self, flb, fub, level, attenuation=0):
        """
        Get mean scaling factor over frequency band.

        Parameters
        ----------
        flb : float
            Lower frequency bound in Hz.
        fub : float
            Upper frequency bound in Hz.
        level : float
            Target level in dB re reference unit.
        attenuation : float, optional
            Additional attenuation in dB. Default is 0.

        Raises
        ------
        ValueError
            Point calibrations do not support mean scaling factors over bands.
        """
        raise ValueError('Not implemented for PointCalibration')


def load_demo_starship():
    """
    Load demonstration calibration data for Starship system.

    Returns
    -------
    InterpCalibration
        Calibration object loaded from demo data including frequency response
        and phase information.

    Examples
    --------
    >>> cal = load_demo_starship()
    >>> # Use calibration for audio measurements
    >>> db_spl = cal.get_db(frequencies, voltages)
    """
    cal_file = resources.files('psiaudio.resources') / 'starship_cal.csv'
    with resources.as_file(cal_file) as fh:
        cal = pd.read_csv(fh)
        return InterpCalibration(cal['freq'], cal['SPL'], phase=cal['phase'])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
