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
from matplotlib import transforms as T
import numpy as np
import pandas as pd

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
        if vmin < 0:
            # This is usually the case when some padding has been added to the
            # axes even if the data is only positive values.
            vmin = self.axis.get_minpos()
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


def waterfall_plot(axes, waveforms, waterfall_level='level',
                   scale_method='mean', plotkw=None, x_transform=None):
    if x_transform is None:
        x_transform = lambda x: x
    levels = waveforms.index.get_level_values(waterfall_level)
    t = x_transform(waveforms.columns.values)
    waveforms = waveforms.values
    n = len(waveforms)
    offset_step = 1/(n+1)

    text_trans = T.blended_transform_factory(axes.figure.transFigure,
                                             axes.transAxes)

    limits = [(w.min(), w.max()) for w in waveforms]

    if scale_method == 'mean':
        base_scale = np.mean(np.abs(np.array(limits)))
    elif scale_method == 'max':
        base_scale = np.max(np.abs(np.array(limits)))
    else:
        raise ValueError(f'Unsupported scale_method "{scale_method}"')

    if plotkw is None:
        plotkw = {
            'color': 'k',
            'clip_on': False,
        }

    bscale_in_box = T.Bbox([[0, -base_scale], [1, base_scale]])
    bscale_out_box = T.Bbox([[0, -1], [1, 1]])
    bscale_in = T.BboxTransformFrom(bscale_in_box)
    bscale_out = T.BboxTransformTo(bscale_out_box)

    tscale_in_box = T.Bbox([[0, -1], [1, 1]])
    tscale_out_box = T.Bbox([[0, 0], [1, offset_step]])
    tscale_in = T.BboxTransformFrom(tscale_in_box)
    tscale_out = T.BboxTransformTo(tscale_out_box)

    for i, (l, w) in enumerate(zip(levels, waveforms)):
        y_min, y_max = w.min(), w.max()
        tnorm_in_box = T.Bbox([[0, -1], [1, 1]])
        tnorm_out_box = T.Bbox([[0, -1], [1, 1]])
        tnorm_in = T.BboxTransformFrom(tnorm_in_box)
        tnorm_out = T.BboxTransformTo(tnorm_out_box)

        offset = offset_step * i + offset_step * 0.5
        translate = T.Affine2D().translate(0, offset)

        y_trans = bscale_in + bscale_out + \
            tnorm_in + tnorm_out + \
            tscale_in + tscale_out + \
            translate + axes.transAxes
        trans = T.blended_transform_factory(axes.transData, y_trans)

        axes.plot(t, w, transform=trans, **plotkw)
        text_trans = T.blended_transform_factory(axes.transAxes, y_trans)
        axes.text(-0.05, 0, str(l), transform=text_trans)

    axes.set_yticks([])
    axes.grid()
    for spine in ('top', 'left', 'right'):
        axes.spines[spine].set_visible(False)
