'''
Defines custom scales that may be useful for plotting audio data

Example
-------

To set the x-axis to show ticks at octave intervals but ensure that the label
SI unit is in kHz even if the data SI unit is in Hz:

>>> ax.set_xscale('octave', octaves=0.5, data_si='', label_si='k')


'''
from psiaudio.util import octave_space
from matplotlib import ticker as mticker
from matplotlib import scale as mscale

# Factor to multiply tick location by to transform from data SI prefix to label
# SI prefix.
UNIT_CONVERSION = {
    ('', 'k'):  1e-3,
    ('', ''):   1,
    ('k', 'k'): 1,
    ('k', ''):  1e3,
}


class OctaveLocator(mticker.Locator):

    def __init__(self, octaves, data_si='', label_si='k', mode='nearest'):
        self.octaves = octaves
        self.data_si = data_si
        self.label_si = label_si
        self.conv = UNIT_CONVERSION[data_si, label_si]
        self.mode = mode

    def __call__(self):
        """Return the locations of the ticks"""
        bounds = self.axis.get_view_interval() * self.conv
        return self.tick_values(*bounds) / self.conv

    def tick_values(self, vmin, vmax):
        return octave_space(vmin, vmax, self.octaves, self.mode)


class OctaveFormatter(mticker.Formatter):

    def __init__(self, data_si='', label_si='k'):
        self.conv = UNIT_CONVERSION[data_si, label_si]

    def __call__(self, x, pos=None):
        label_x = x * self.conv
        if label_x >= 1:
            label_x = round(label_x)
        return f'{label_x}'


class OctaveScale(mscale.ScaleBase):

    name = 'octave'

    def __init__(self, axis, *, octaves=1, mode='nearest', data_si='', label_si='k'):
        super().__init__(axis)
        self.octaves = octaves
        self.mode = mode
        self.data_si = data_si
        self.label_si = label_si

    def get_transform(self):
        return mscale.LogTransform(2)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(OctaveLocator(self.octaves, self.data_si,
                                             self.label_si, self.mode))
        axis.set_major_formatter(OctaveFormatter(self.data_si, self.label_si))
        axis.set_minor_locator(mticker.NullLocator())
        axis.set_minor_formatter(mticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return (minpos if vmin <= 0 else vmin,
                minpos if vmax <= 0 else vmax)


mscale.register_scale(OctaveScale)
