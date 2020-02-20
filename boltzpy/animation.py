
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.animation as mpl_ani
import matplotlib.axes as mpl_axes
import matplotlib.lines as mpl_lines
import matplotlib.pyplot as plt
from time import time

import boltzpy as bp
import boltzpy.constants as bp_c


# Todo add custom colors custom linewidth,... for animate and snapshot
# Todo option to not show moment name for snapshot and animate
# todo change order of params for snapshot / animate -> moment before species
class Animation:
    """Handles the visualization of the results.

    Parameters
    ----------
    simulation : :class:`~boltzpy.Simulation`
    """
    def __init__(self, simulation):
        assert isinstance(simulation, bp.Simulation)
        # Todo simulation.check_integrity(complete_check=False)
        self._sim = simulation
        self._writer = mpl_ani.writers['ffmpeg'](fps=15, bitrate=1800)
        return

    def snapshot(self,
                 time_step,
                 moment_array,
                 specimen_array,
                 file_name,
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
        specimen_array : :obj:`~numpy.ndarray`  [:class:`~boltzpy.Specimen`], optional
        moment_array : :obj:`~numpy.ndarray` [:obj:`str`], optional
        file_name : :obj:`str`, optional
            File name of the vector image.
        """
        # Todo Reasonable asserts for time_step
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
        file = h5py.File(self._sim.file_address + '.hdf5', mode='r')
        hdf5_group = file["Computation"]
        for (s_idx, specimen) in enumerate(species_names):
            for (m_idx, moment) in enumerate(moments_flat):
                result_t = hdf5_group[specimen][moment][time_step]
                lines[m_idx, s_idx].set_data(self._sim.p.pG,
                                             result_t)
        # save the plot
        figure.savefig(file_name, format='eps')
        return

    def animate(self,
                moment_array,
                specimen_array):
        """Creates animated plot and saves it to disk.

        Sets up :obj:`matplotlib.animation.FuncAnimation` instance
        based on a figure with separate subplots
        for every moment in *moment_array*
        and separate lines for every specimen and moment.
        """

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
                                    frames=self._sim.t.size,
                                    interval=1,
                                    blit=False)
        ani.save(self._sim.file_address + '.mp4',
                 self._writer,
                 dpi=figure.dpi)
        # Todo move format_time into helpers.py
        time_taken = bp.Data._format_time(int(time() - ani_time))

        print('Animating....Done\n'
              'Time taken =  {} seconds'
              '\n'.format(time_taken))
        return

    def _update_data(self,
                     time_step,
                     lines,
                     specimen_array,
                     moment_array):
        specimen_names = [specimen.name for specimen in specimen_array]
        moments = moment_array.flatten()
        file = h5py.File(self._sim.file_address + '.hdf5', mode='r')
        hdf5_group = file["Computation"]
        for (specimen_idx, specimen) in enumerate(specimen_names):
            for (moment_idx, moment) in enumerate(moments):
                result_t = hdf5_group[moment][time_step, ...,  specimen_idx]
                lines[moment_idx, specimen_idx].set_data(self._sim.p.pG,
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
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        if figsize is None:
            figsize = bp_c.DEFAULT_FIGSIZE
        else:
            assert isinstance(figsize, tuple)
            assert all([isinstance(_, int) and _ > 0
                        for _ in figsize])
            assert len(figsize) == 2
        if dpi is None:
            dpi = bp_c.DEFAULT_DPI
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
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        assert isinstance(figure, plt.Figure)
        assert isinstance(moment_array, np.ndarray)
        assert all([moment
                    in self._sim.output_parameters.flatten()
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
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        x_min = np.min(self._sim.p.iG, axis=0) * self._sim.p.delta
        x_max = np.max(self._sim.p.iG, axis=0) * self._sim.p.delta
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
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        file = h5py.File(self._sim.file_address + '.hdf5', mode='r')
        hdf5_dataset = file["Computation"][moment]
        min_val = min(0, np.min(hdf5_dataset))
        max_val = max(0, np.max(hdf5_dataset))
        axes.set_ylim(1.25 * min_val, 1.25 * max_val)
        return

    def _set_tick_labels(self, axes, i_m):
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        shape = self._sim.output_parameters.shape
        last_row = (shape[0]-1) * shape[1]
        if i_m < last_row:
            axes.set_xticklabels([])
        return

    def setup_lines(self,
                    axes_array,
                    specimen_array):
        """Sets up the plot lines.

        Creates a plot-line for each :class:`~boltzpy.Specimen`
        in each axes of *axes_array*.
        Each line is a separate plot containing data,
        which can be updated if necessary
        (:meth:`animated plot <run>`).

        Parameters
        ----------
        axes_array : :obj:`~numpy.ndarray` [:obj:`matplotlib.axes.Axes`]
        specimen_array : :obj:`~numpy.ndarray` [:class:`~boltzpy.Specimen`]

        Returns
        -------
        :obj:`~numpy.ndarray` [:obj:`matplotlib.lines.Line2D`]
        """
        if self._sim.p.ndim != 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        assert isinstance(axes_array, np.ndarray)
        assert all([isinstance(axes, mpl_axes.Axes)
                    for axes in axes_array])
        assert isinstance(specimen_array, np.ndarray)
        assert specimen_array.ndim == 1
        assert all([isinstance(specimen, bp.Specimen)
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
        specimen_array : :obj:`~numpy.ndarray` [:class:`~boltzpy.Specimen`]
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
        assert all([isinstance(specimen, bp.Specimen)
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
