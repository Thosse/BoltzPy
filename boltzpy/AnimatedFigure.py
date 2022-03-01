
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as mpl_ani
import boltzpy.Tools.TimeTracker as h_tt


# Todo check maximum tmax over all elements >= fig.tmax
# Todo set up legend
class AnimatedFigure:
    def __init__(self,
                 tmax=1,
                 backend="Qt5Agg",
                 figsize=(16, 9),
                 dpi=300,
                 writer='ffmpeg',
                 **kwargs):
        mpl.use(backend)
        import matplotlib.pyplot as plt
        self._plt = plt
        self.mpl_fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        assert type(tmax) == int
        assert tmax > 0
        self.tmax = int(tmax)
        self.writer = mpl.animation.writers[writer](fps=15, bitrate=1800)
        self.animated_axes = []
        self._time_tracker = h_tt.TimeTracker()
        return

    # Todo make position optional
    def add_subplot(self, position, dim=2, **kwargs):
        assert isinstance(position, tuple)
        assert 0 < position[2] <= position[0] * position[1]
        if dim == 2:
            mpl_axes = self.mpl_fig.add_subplot(*position, **kwargs)
        elif dim == 3:
            mpl_axes = self.mpl_fig.add_subplot(*position, projection='3d', **kwargs)
        else:
            raise ValueError
        animated_axes = AnimatedAxes(mpl_axes=mpl_axes, dim=dim, tmax=self.tmax)
        self.animated_axes.append(animated_axes)
        return animated_axes

    def update(self, i, track_time=True):
        for ax in self.animated_axes:
            ax.update(i)
        # print time estimate to console
        if track_time:
            self._time_tracker.print(i + 1, self.tmax)
        return

    def create_plot(self, track_time=True):
        self.mpl_fig.tight_layout()
        if track_time:
            self._time_tracker = h_tt.TimeTracker()
        if self.tmax == 1:
            print("Plotting...")
            self.update(0)
            return self.mpl_fig
        elif self.tmax > 1:
            print('Animating...')
            animation = mpl.animation.FuncAnimation(
                self.mpl_fig,
                self.update,
                frames=self.tmax,
                interval=1,
                fargs=(track_time,),
                blit=False
            )
            return animation
        else:
            raise ValueError

    def save(self, file_address):
        plot = self.create_plot(track_time=True)
        if self.tmax == 1:
            plot.savefig(file_address)
        elif self.tmax > 1:
            plot.save(file_address,
                      self.writer,
                      dpi=self.mpl_fig.dpi)
        else:
            return ValueError
        return

    def show(self):
        # must store plt, otherwise animations is not shown (figure instead)
        plt = self.create_plot(track_time=False)
        self._plt.show()


# Todo add different modes:
#  Allow 0d plots
#   moving_dot: plot thin line, with thick dot at t=i
#   snake (plot t = [0,..,i])
class AnimatedAxes:
    def __init__(self, mpl_axes, dim=2, tmax=1):
        assert dim in [2, 3]
        self.dim = dim
        if dim == 2:
            assert isinstance(mpl_axes, mpl.axes.Axes)
            assert not isinstance(mpl_axes, Axes3D)
        elif dim == 3:
            assert isinstance(mpl_axes, Axes3D)
        else:
            raise ValueError
        self.mpl_axes = mpl_axes
        assert type(tmax) == int
        assert tmax > 0
        self.tmax = int(tmax)
        self.animated_objects = []
        return

    def plot(self, xdata, ydata, zdata=None, **kwargs):
        # create animated line
        # Todo maybe this can be merged into a single class?
        if self.dim == 2:
            animated_obj = AnimatedLine(self.mpl_axes, xdata, ydata, **kwargs)
        elif self.dim == 3:
            animated_obj = AnimatedBar3d(self.mpl_axes, xdata, ydata, zdata, **kwargs)
        else:
            raise ValueError
        self.animated_objects.append(animated_obj)
        # update limits
        # update y limits, stretch range a bit, for nice look
        stretch = 1.25
        factors = np.array([(1 + stretch)/2, (1 - stretch)/2])
        self.mpl_axes.set_xlim(self.xmin, self.xmax)
        if self.dim == 2:
            ymin = np.sum(factors * [self.ymin, self.ymax])
            ymax = np.sum(factors * [self.ymax, self.ymin])
            self.mpl_axes.set_ylim(float(ymin), float(ymax))
        elif self.dim == 3:
            self.mpl_axes.set_ylim(self.ymin, self.ymax)
            zmin = np.sum(factors * [self.zmin, self.zmax])
            zmax = np.sum(factors * [self.zmax, self.zmin])
            self.mpl_axes.set_zlim(float(zmin), float(zmax))
        else:
            raise ValueError
        return

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

    @property
    def zmin(self):
        assert self.dim == 3
        return np.min([e.zmin for e in self.animated_objects])

    @property
    def zmax(self):
        assert self.dim == 3
        return np.max([e.zmax for e in self.animated_objects])


class AnimatedObject:
    def __init__(self,
                 mpl_axes,
                 xdata,
                 ydata,
                 zdata=None,
                 is_constant=False,
                 **kwargs):
        assert isinstance(xdata, np.ndarray)
        assert xdata.ndim == 1
        self.xdata = xdata
        assert isinstance(ydata, np.ndarray)
        assert ydata.ndim in [1, 2]
        assert ydata.shape[-1] == xdata.size
        self.ydata = ydata
        if zdata is not None:
            assert isinstance(zdata, np.ndarray)
            assert zdata.ndim in [1, 2]
            assert zdata.shape[-1] == xdata.size
        self.zdata = zdata
        assert isinstance(mpl_axes, mpl.axes.Axes)
        self.mpl_axes = mpl_axes
        assert isinstance(is_constant, bool)
        self.is_constant = is_constant
        self.kwargs = kwargs
        # self.dim == 2
        if self.zdata is None:
            if self.ydata.ndim == 1:
                self.ydata = self.ydata[np.newaxis, :]
            assert self.ydata.ndim == 2
            assert self.ydata.shape == (self.tmax, self.xdata.size)
        # self.dim == 3
        elif self.zdata is not None:
            # zdata
            if self.zdata.ndim == 1:
                self.zdata = self.zdata[np.newaxis, :]
            assert self.ydata.ndim == 1
            assert self.zdata.ndim == 2
            assert self.zdata.shape == (self.tmax, self.xdata.size)
            assert isinstance(self.mpl_axes, Axes3D)
        else:
            raise ValueError
        return

    def update(self, i):
        raise NotImplementedError

    @property
    def dim(self):
        # returns 2 if zdata is None, else 3
        return 3 - (self.zdata is None)

    @property
    def tmax(self):
        if self.dim == 2:
            return self.ydata.shape[0]
        elif self.dim == 3:
            return self.zdata.shape[0]
        else:
            raise ValueError

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

    @property
    def zmin(self):
        if self.dim == 3:
            return np.min(self.zdata)
        else:
            raise ValueError
            # return 0

    @property
    def zmax(self):
        if self.dim == 3:
            return np.max(self.zdata)
        else:
            raise ValueError
            # return 0


class AnimatedLine(AnimatedObject):
    def __init__(self, axes, xdata, ydata, **kwargs):
        super().__init__(axes, xdata, ydata, **kwargs)
        (self.line,) = self.mpl_axes.plot(self.xdata,
                                          self.ydata[0, :],
                                          **self.kwargs)
        self.update(0)

    def update(self, i):
        if not self.is_constant:
            self.line.set_data(self.xdata, self.ydata[i, :])
        return


class AnimatedBar3d(AnimatedObject):
    def __init__(self, axes, xdata, ydata, zdata, dx=None, dy=None, **kwargs):
        super().__init__(axes, xdata, ydata, zdata, **kwargs)
        # setup and store kwargs
        # use same color for each bar
        if "color" not in self.kwargs.keys():
            self.kwargs["color"] = "green"
        if "alpha" not in self.kwargs.keys():
            self.kwargs["alpha"] = 0.8
        # calculate bar-width
        if dx is not None:
            self.dx = dx
        else:
            self.dx = np.max(self.xdata[1:] - self.xdata[0:-1])
        if dy is not None:
            self.dy = dy
        else:
            self.dy = np.max(self.ydata[1:] - self.ydata[0:-1])
        # setup bars
        self.bars = [self.bar(0, v) for v in np.arange(self.xdata.size)]
        self.update(0)

    def update(self, i):
        if not self.is_constant:
            for bar in self.bars:
                bar.remove()
            self.bars = [self.bar(i, v) for v in np.arange(self.xdata.size)]
        return

    def bar(self, t, i):
        bar = self.mpl_axes.bar3d(self.xdata[i] - self.dx / 2,
                                  self.ydata[i] - self.dy / 2,
                                  0,
                                  self.dx,
                                  self.dy,
                                  self.zdata[t, i],
                                  **self.kwargs)
        return bar


# Todo add wireframe
#  -> must generate meshgrid from data / data must be changed into meshgrid form
# class AnimatedWireframe:
