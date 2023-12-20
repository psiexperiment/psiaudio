'''
Defines custom scales that may be useful for plotting audio data

Example
-------

To set the x-axis to show ticks at octave intervals but ensure that the label
SI unit is in kHz even if the data SI unit is in Hz:

>>> ax.set_xscale('octave', octaves=0.5, data_si='', label_si='k')


'''
import importlib
from matplotlib import ticker as mticker
from matplotlib import scale as mscale
from matplotlib import transforms as T
import numpy as np
import pandas as pd

from psiaudio.util import octave_space

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

    def __init__(self, data_si='', label_si='k', precision=None):
        self.conv = UNIT_CONVERSION[data_si, label_si]
        if precision is None:
            precision = 0 if label_si == '' else 1
        self.precision = precision

    def __call__(self, x, pos=None):
        label_x = x * self.conv
        if label_x >= 1:
            label_x = round(label_x, self.precision)
        return f'{label_x}'


class OctaveScale(mscale.ScaleBase):

    name = 'octave'

    def __init__(self, axis, *, octaves=1, mode='nearest', data_si='',
                 label_si='k', precision=None):
        super().__init__(axis)
        self.octaves = octaves
        self.mode = mode
        self.data_si = data_si
        self.label_si = label_si
        self.precision = precision

    def get_transform(self):
        return mscale.LogTransform(2)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(OctaveLocator(self.octaves, self.data_si,
                                             self.label_si, self.mode))
        axis.set_major_formatter(OctaveFormatter(self.data_si, self.label_si, self.precision))
        axis.set_minor_locator(mticker.NullLocator())
        axis.set_minor_formatter(mticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return (minpos if vmin <= 0 else vmin,
                minpos if vmax <= 0 else vmax)


mscale.register_scale(OctaveScale)


def waterfall_plot(axes, waveforms, waterfall_level='level',
                   scale_method='mean', base_scale_multiplier=1, plotkw=None,
                   x_transform=None, y_scale_bar_size=1):
    '''
    Parameters
    ----------
    axes : matplotlib Axes instance
        Axes to plot on
    '''
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
        base_scale = np.mean(np.abs(np.array(limits))) * base_scale_multiplier
    elif scale_method == 'max':
        base_scale = np.max(np.abs(np.array(limits))) * base_scale_multiplier
    else:
        raise ValueError(f'Unsupported scale_method "{scale_method}"')

    if plotkw is None:
        plotkw = {}
    plotkw.setdefault('color', 'k')
    plotkw.setdefault('clip_on', False)

    # Defines the "base scale" transform that is used to adjust the overall
    # scale of all plotted lines and widgets.
    bscale_in_box = T.Bbox([[0, -base_scale], [1, base_scale]])
    bscale_out_box = T.Bbox([[0, -1], [1, 1]])
    bscale_in = T.BboxTransformFrom(bscale_in_box)
    bscale_out = T.BboxTransformTo(bscale_out_box)

    # This compresses vertically so lines tend to be plotted within the range
    # allowed by the offset spacing. 
    tscale_in_box = T.Bbox([[0, -1], [1, 1]])
    tscale_out_box = T.Bbox([[0, 0], [1, offset_step]])
    tscale_in = T.BboxTransformFrom(tscale_in_box)
    tscale_out = T.BboxTransformTo(tscale_out_box)

    if y_scale_bar_size is not None:
        y_trans = bscale_in + bscale_out + \
            tscale_in + tscale_out + \
            T.Affine2D().translate(0, 1) + \
            axes.transAxes

        scale_bar_trans = T.blended_transform_factory(axes.transAxes, y_trans)
        axes.plot([1, 1], [0, y_scale_bar_size], transform=scale_bar_trans, color='r')

    for i, (l, w) in enumerate(zip(levels, waveforms)):
        y_min, y_max = w.min(), w.max()
        #tnorm_in_box = T.Bbox([[0, -1], [1, 1]])
        #tnorm_out_box = T.Bbox([[0, -1], [1, 1]])
        #tnorm_in = T.BboxTransformFrom(tnorm_in_box)
        #tnorm_out = T.BboxTransformTo(tnorm_out_box)

        offset = offset_step * i + offset_step * 0.5
        translate = T.Affine2D().translate(0, offset)

        #Add this after bscale tnorm_in + tnorm_out + \
        y_trans = bscale_in + bscale_out + \
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


def get_color_cycle(n, name='palettable.matplotlib.Viridis_20_r', fmt='matplotlib'):
    '''
    Return an iterator over the specified palettable color scheme that returns
    colors that can be use for plotting.

    Parameters
    ----------
    n : int
        Number of colors needed.
    name : string
        Name of fully qualified palettable color map (e.g.,
        palettable.matplotlib.Viridis_20_r). The number does not matter for
        continuous scales since we will be interpolating.
    fmt: {'matplotlib'}
        Format to return colors in. Select the version that is compatible with
        the plotting library you are using so you can pass the colors in
        directly to plot function.

    Returns
    -------
    iterator:
        Iterator that yields a color in the requested format.
    '''
    module_name, cmap_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)

    formatters = {
        'matplotlib': lambda x: x,
        'pyqtgraph': lambda x: tuple(int(v * 255) for v in x),
    }

    # This generates a LinearSegmentedColormap instance that interpolates to
    # the requested number of colors. We can then extract these colors by
    # calling the colormap with a mapping of 0 ... 1 where the number of values
    # in the array is the number of colors we need (spaced equally along 0 ...
    # 1).

    formatter = formatters[fmt]
    cmap = getattr(module, cmap_name)
    if cmap.type == 'qualitative':
        # For qualitative color maps, don't do any interpoloation because
        # they're not really designed for that.
        if len(cmap.mpl_colors) < n:
            raise ValueError('Not enough colors available')
        for i in range(n):
            yield cmap.mpl_colors[i]
    else:
        # Diverging and sequential can be interpolated.
        cmap = cmap.mpl_colormap.resampled(n)
        for i in np.linspace(0, 1, n):
            yield formatter(cmap(i))


def iter_colors(x, *args, **kw):
    '''
    Like enumerate, but yields colors instead of count.

    This only works if the length of the iterable can be determined in advance.
    Pass in as `list(generator)` if you are using generators.

    Parameters
    ----------
    x : iterable

    Additional arguments are passed to `get_color_cycle`.

    Returns
    -------
    iterator :
        An iterable object. The __next__ method of the iterator returns a tuple
        containing a color and the value obtained from iterating over iterable.

    Examples
    --------
    Quickly generate a dictionary mapping value to color. Useful for plotting.

    >>> values = [0, 10, 20, 40, 80]
    >>> value_colors = {value: color for color, value in iter_colors(values)
    '''
    n = len(x)
    c_iter = get_color_cycle(n, *args, **kw)
    for c, v in zip(c_iter, x):
        yield c, v
