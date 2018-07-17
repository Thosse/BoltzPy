
from boltzmann.configuration import specimen as b_spm
import boltzmann.constants as b_const

from time import time
import matplotlib.axes as mpl_axes
import matplotlib.lines as mpl_lines
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_ani
import h5py

import numpy as np


# Todo Check what happens if animation is not saved / snapshot is taken
# Todo may need specific DPI for snapshots. un-saved animation is not important
# if self._save_animation:
#     self.dpi = 300
# else:
#     self.dpi = 100
# Todo remove _save_animation -> always save animation
class Animation:
    """Handles the visualization of the results.

    Parameters
    ----------
    cnf : :class:`~boltzmann.configuration.Configuration`
    """
    def __init__(self, cnf):
        self._cnf = cnf
        self._writer = mpl_ani.writers['ffmpeg'](fps=15, bitrate=1800)
        return

    def snapshot(self,
                 time_step,
                 moment_array=None,
                 specimen_array=None,
                 file_name=None,
                 # Todo p_space -> cut of boundary effects,
                 # Todo color -> color some Specimen differently,
                 # Todo legend -> setup legend in the image?
                 ):
        """Creates a vector plot of the simulation
        at the desired time_step.
        Saves the file as *file_name*.eps in the Simulation folder.

        Parameters
        ----------
        time_step : :obj:`int`
        specimen_array : :obj:`~numpy.ndarray`  [:class:`~boltzmann.configuration.Specimen`], optional
        moment_array : :obj:`~numpy.ndarray` [:obj:`str`], optional
        file_name : :obj:`str`, optional
            File name of the vector image.
        """
        # Todo Reasonable asserts for time_step
        if specimen_array is None:
            specimen_array = self._cnf.s.specimen_array
        assert isinstance(specimen_array, np.ndarray)
        assert specimen_array.ndim == 1
        assert all([isinstance(specimen, b_spm.Specimen)
                    for specimen in specimen_array])
        if moment_array is None:
            moment_array = self._cnf.animated_moments
        else:
            assert isinstance(moment_array, np.ndarray)
            assert all([moment in self._cnf.animated_moments.flatten()
                        for moment in moment_array])
        if file_name is None:
            file_name = (self._cnf.file_address[0:-4]
                         + '_t={}.eps'.format(time_step))
        else:
            # Todo are the reasonable checks?
            pass
        # Set up the figure, subplots and lines
        figure = self.setup_figure()
        axes = self.setup_axes(figure, moment_array)
        lines = self.setup_lines(axes,
                                 specimen_array)
        self.setup_legend(figure,
                          lines,
                          specimen_array)

        # Get data from the hdf5-file and set up the lines
        species_names = [specimen.name for specimen in specimen_array]
        moments_flat = moment_array.flatten()
        hdf_group = h5py.File(self._cnf.file_address, mode='r')["Results"]
        for (s_idx, specimen) in enumerate(species_names):
            for (m_idx, moment) in enumerate(moments_flat):
                result_t = hdf_group[specimen][moment][time_step]
                lines[m_idx, s_idx].set_data(self._cnf.p.pG,
                                             result_t)
        # save the plot
        figure.savefig(file_name, format='eps')
        return

    def animate(self,
                moment_array=None,
                specimen_array=None):
        """Creates animated plot and saves it to disk.

        Sets up :obj:`matplotlib.animation.FuncAnimation` instance
        based on a figure with separate subplots
        for every moment in *moment_array*
        and separate lines for every specimen and moment.
        """
        if moment_array is None:
            moment_array = self._cnf.animated_moments
        else:
            assert isinstance(moment_array, np.ndarray)
            assert all([moment in self._cnf.animated_moments.flatten()
                        for moment in moment_array.flatten()])
        if specimen_array is None:
            specimen_array = self._cnf.s.specimen_array
        else:
            assert isinstance(specimen_array, np.ndarray)
            assert specimen_array.ndim == 1
            assert all([isinstance(specimen, b_spm.Specimen)
                        for specimen in specimen_array])
        ani_time = time()
        print('Animating....',
              end='\r')
        # Set up the figure, subplots and lines
        figure = self.setup_figure()
        axes = self.setup_axes(figure, moment_array)
        lines = self.setup_lines(axes,
                                 specimen_array)
        self.setup_legend(figure,
                          lines,
                          specimen_array)
        # Create Animation
        ani = mpl_ani.FuncAnimation(figure,
                                    self._update_data,
                                    fargs=(lines,
                                           specimen_array,
                                           moment_array),
                                    # Todo speed up!
                                    # init_func=init
                                    frames=self._cnf.t.size,
                                    interval=1,
                                    blit=False)
        ani.save(self._cnf.file_address[0:-4] + '.mp4',
                 self._writer,
                 dpi=figure.dpi)

        print('Animating....Done\n'
              'Time taken =  {} seconds'
              '\n'.format(round(time() - ani_time, 3)))
        return

    def _update_data(self,
                     time_step,
                     lines,
                     specimen_array,
                     moment_array):
        specimen_names = [specimen.name for specimen in specimen_array]
        moments = moment_array.flatten()
        file = h5py.File(self._cnf.file_address, mode='r')["Results"]
        for (i_s, specimen) in enumerate(specimen_names):
            for (i_m, moment) in enumerate(moments):
                result_t = file[specimen][moment][time_step]
                lines[i_m, i_s].set_data(self._cnf.p.pG,
                                         result_t)
        return lines

    def setup_figure(self,
                     figsize=None,
                     dpi=None):
        """Creates a :obj:`matplotlib.pyplot.figure` object and sets up its
        basic attributes (figsize, dpi).

        Parameters
        ----------
        figsize : :obj:`tuple` [:obj:`int`], optional
        dpi : :obj:`int`, optional

        Returns
        -------
        :obj:`matplotlib.pyplot.figure`
        """
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        if figsize is None:
            figsize = b_const.DEFAULT_FIGSIZE
        else:
            assert isinstance(figsize, tuple)
            assert all([isinstance(_, int) and _ > 0
                        for _ in figsize])
            assert len(figsize) == 2
        if dpi is None:
            dpi = b_const.DEFAULT_DPI
            assert isinstance(dpi, int)
        figure = plt.figure(figsize=figsize,
                            dpi=dpi)
        return figure

    def setup_axes(self,
                   figure,
                   moment_array):
        """
        Sets up the subplots, their position in the *figure* and basic
        preferences for all moments in *moment_array*.

        * Creates separate coordinate systems / axes for all moments.
          Each coordinate systems contains the lines of all :class:`Specimen`.
        * Specifies the position, the range of values
          and the title for each of each subplot / axes.

        Parameters
        ----------
        figure : :obj:`matplotlib.pyplot.figure`
        moment_array : :obj:`~numpy.ndarray` [:obj:`str`]

        Returns
        -------
        :obj:`~numpy.ndarray` [:obj:`matplotlib.axes.Axes`]
        """
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        assert isinstance(figure, plt.Figure)
        assert isinstance(moment_array, np.ndarray)
        assert all([moment in self._cnf.animated_moments.flatten()
                    for moment in moment_array.flatten()])
        # array of all subplots
        axes_array = np.empty(shape=moment_array.size,
                              dtype=object)
        moments_flat = moment_array.flatten()
        shape = moment_array.shape
        for (i_m, moment) in enumerate(moments_flat):
            # subplot are placed in a matrix grid
            (rows, columns) = shape
            # subplots begin counting at 1
            # flat index in the matrix grid, iterates over rows
            place = i_m + 1
            axes = figure.add_subplot(rows,
                                      columns,
                                      place)
            # Todo properly document / remove submethods
            # Todo Wait for this after implementing 2D Transport / P-Grids
            # set range of X-axis
            self._set_range(axes)
            # set range of Y-axis, based on occurring values in data
            self._set_val_limits(axes, moment)
            # add Titles to subplot
            axes.set_title(moment)
            self._set_tick_labels(axes, i_m)
            axes_array[i_m] = axes
        return axes_array

    def _set_range(self, axes):
        if self._cnf.p.dim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        x_boundaries = self._cnf.p.boundaries
        x_min = x_boundaries[0, 0]
        x_max = x_boundaries[1, 0]
        axes.set_xlim(x_min, x_max)
        return

    def _set_val_limits(self, axes, moment):
        """Sets the range of values (range of the density functions)
        for the subplot(ax) of the given moment.

        Calculates the minimum and maximum occurring value
        for this moment over all :obj:`~Configuration.Specimen`.
        These values are set as the limits of values

        Parameters
        ----------
        axes : :obj:`~numpy.ndarray` [:obj:`matplotlib.axes.Axes`]
            Subplot for moment
        moment : str
            Name of the moment
        """
        if self._cnf.p.dim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        min_val = 0
        max_val = 0
        # Todo This should depend only on Specimen_array, not on cnf.s
        # Todo Go to definition-> fix this
        species = self._cnf.s.names
        file = h5py.File(self._cnf.file_address, mode='r')["Results"]
        for specimen in species:
            file_m = file[specimen][moment]
            for t in self._cnf.t.iG[:, 0]:
                i_t = t // self._cnf.t.multi
                result_t = file_m[i_t]
                min_val = min(min_val, np.min(result_t))
                max_val = max(max_val, np.max(result_t))
        axes.set_ylim(1.25 * min_val, 1.25 * max_val)
        return

    def _set_tick_labels(self, axes, i_m):
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        # Todo This should depend only on moment_array, not on cnf.s
        # Todo Go to definition-> fix this
        shape = self._cnf.animated_moments.shape
        last_row = (shape[0]-1) * shape[1]
        if i_m < last_row:
            axes.set_xticklabels([])
        return

    def setup_lines(self,
                    axes_array,
                    specimen_array):
        """Sets up the plot lines.

        Creates a plot-line for each :class:`~boltzmann.configuration.Specimen`
        in each axes of *axes_array*.
        Each line is a separate plot containing data,
        which can be updated if necessary
        (:meth:`animated plot <run>`).

        Parameters
        ----------
        axes_array : :obj:`~numpy.ndarray` [:obj:`matplotlib.axes.Axes`]
        specimen_array : :obj:`~numpy.ndarray` [:class:`~boltzmann.configuration.Specimen`]

        Returns
        -------
        :obj:`~numpy.ndarray` [:obj:`matplotlib.lines.Line2D`]
        """
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        assert isinstance(axes_array, np.ndarray)
        assert all([isinstance(axes, mpl_axes.Axes)
                    for axes in axes_array])
        assert isinstance(specimen_array, np.ndarray)
        assert specimen_array.ndim == 1
        assert all([isinstance(specimen, b_spm.Specimen)
                    for specimen in specimen_array])
        lines = np.empty(shape=(axes_array.size, specimen_array.size),
                         dtype=object)
        for (a_idx, axes) in enumerate(axes_array):
            for (s_idx, specimen) in enumerate(specimen_array):
                # initialize line without any data
                line = axes.plot([],
                                 [],
                                 linestyle='-',
                                 color=specimen.color,
                                 linewidth=2)
                # plot returns a tuple of lines
                # in this case its a tuple with only one element
                lines[a_idx, s_idx] = line[0]
        return lines

    @staticmethod
    def setup_legend(figure,
                     line_array,
                     specimen_array,
                     loc='lower center'):
        """Configures the legend of *figure*.

        Parameters
        ----------
        figure : :obj:`matplotlib.pyplot.figure`
        line_array : :obj:`~numpy.ndarray` [:class:`matplotlib.lines.Line2D`]
        specimen_array : :obj:`~numpy.ndarray` [:class:`~boltzmann.configuration.Specimen`]
        loc : :obj:`str`, optional
            Location of the legend in *figure*
        """
        assert isinstance(figure, plt.Figure)
        assert isinstance(line_array, np.ndarray)
        assert line_array.ndim == 2
        assert all([isinstance(line, mpl_lines.Line2D)
                    for line in line_array.flatten()])
        assert isinstance(specimen_array, np.ndarray)
        assert specimen_array.ndim == 1
        assert all([isinstance(specimen, b_spm.Specimen)
                    for specimen in specimen_array])
        assert isinstance(loc, str)
        # get specimen names
        names = [specimen.name for specimen in specimen_array]
        figure.legend(line_array[0, :],
                      names,
                      loc=loc,
                      ncol=specimen_array.size,
                      # expand legend horizontally
                      mode='expand',
                      borderaxespad=0.5)
        return
