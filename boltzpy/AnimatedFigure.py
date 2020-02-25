
import numpy as np
import matplotlib as mpl
import matplotlib.animation as mpl_ani
from time import time


# Todo merge plots und animationen (tmax=0 -> plot)
# Todo methode plot(t=None) und save(t=None) unterscheiden
# Todo set up legend
class AnimatedFigure:
    def __init__(self,
                 tmax=1,
                 backend="TkAgg",
                 figsize=(16, 9),
                 dpi=300,
                 writer='ffmpeg',
                 **kwargs):
        mpl.use(backend)
        import matplotlib.pyplot as plt
        self.figure = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        assert type(tmax) == int
        self.tmax = int(tmax)
        self.writer = mpl.animation.writers[writer](fps=15, bitrate=1800)
        self.animated_axes = []
        return

    def add_subplot(self, position, **kwargs):
        assert isinstance(position, tuple)
        assert 0 < position[2] <= position[0] * position[1]
        ax = self.figure.add_subplot(*position, **kwargs)
        animated_ax = AnimatedAxes(ax, tmax=self.tmax)
        self.animated_axes.append(animated_ax)
        return animated_ax

    def update(self, i):
        for ax in self.animated_axes:
            ax.update(i)
        return

    # Todo Edit update -> print time estimate
    def plot(self):
        self.figure.tight_layout()
        animation = mpl.animation.FuncAnimation(
            self.figure,
            self.update,
            frames=self.tmax,
            interval=1,
            blit=False
        )
        return animation

    # Todo check no file exists already
    # Todo check mp4 ending in file address
    def save(self, file_address):
        animation = self.plot()
        animation.save(file_address,
                       self.writer,
                       dpi=self.figure.dpi)
        return


# Todo add different modes:
#  Allow 0d plots
#   moving_dot: plot thin line, with thick dot at t=i
#   snake (plot t = [0,..,i])

class AnimatedAxes:
    def __init__(self, axes, tmax=1):
        assert isinstance(axes, mpl.axes.Axes)
        self.axes = axes
        assert type(tmax) == int
        self.tmax = int(tmax)
        self.animated_objects = []

    def plot(self, xdata, ydata, **kwargs):
        # assert constistency of time
        assert ydata.shape[0] == self.tmax
        # create animated line
        (mpl_line,) = self.axes.plot([], [], **kwargs)
        animated_line = AnimatedLine(mpl_line, xdata, ydata)
        self.animated_objects.append(animated_line)
        # update x limits
        self.axes.set_xlim(self.xmin, self.xmax)
        # update y limits, stretch range a bit, for nice look
        stretch = 1.25
        factors = np.array([(1 + stretch)/2, (1 - stretch)/2])
        ymin = np.sum(factors * [self.ymin, self.ymax])
        ymax = np.sum(factors * [self.ymax, self.ymin])
        self.axes.set_ylim(ymin, ymax)

    def update(self, i):
        for element in self.animated_objects:
            element.update(i)
        return

    @property
    def xmin(self):
        return np.min([e.xmin for e in self.animated_objects])

    @property
    def xmax(self):
        return np.max([e.xmax for e in self.animated_objects])

    @property
    def ymin(self):
        return np.min([e.ymin for e in self.animated_objects])

    @property
    def ymax(self):
        return np.max([e.ymax for e in self.animated_objects])


class AnimatedLine:
    def __init__(self, line, xdata, ydata):
        assert isinstance(line, mpl.lines.Line2D)
        self.line = line
        assert isinstance(xdata, np.ndarray)
        assert(xdata.ndim == 1)
        self.xdata = xdata
        assert isinstance(ydata, np.ndarray)
        assert (ydata.ndim == 2)
        assert (ydata.shape[1] == xdata.size)
        self.ydata = ydata
        self.update(0)

    def update(self, i):
        self.line.set_data(self.xdata, self.ydata[i, :])

    @property
    def tmax(self):
        return self.ydata.shape[0]

    @property
    def xmin(self):
        return np.min(self.xdata)

    @property
    def xmax(self):
        return np.max(self.xdata)

    @property
    def ymin(self):
        return np.min(self.ydata)

    @property
    def ymax(self):
        return np.max(self.ydata)
