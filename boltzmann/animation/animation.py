
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


class Animation:
    """Handles Animation of the Results

    Parameters
    ----------
    cnf : :class:`~boltzmann.configuration.Configuration`
    """
    def __init__(self,
                 cnf,
                 save_animation=True):
        self._cnf = cnf
        self._save_animation = save_animation
        self.fig_size = (16, 9)
        if self._save_animation:
            self.dpi = 300
        else:
            self.dpi = 100
        self._writer = animation.writers['ffmpeg'](fps=15, bitrate=1800)
        self._figure = plt.figure()
        self._axes = np.zeros((0,), dtype=object)
        self._lines = np.zeros((0,), dtype=object)
        return

    @property
    def file_address(self):
        """:obj:`str` :
        Complete address (path + filename + file type)
        of output animation file.
        """
        file_address = self._cnf.get_file_address(self._cnf.name, 'mp4')
        return file_address

    def run(self):
        """Sets up Figure, creates animated plot
        and either shows it or saves it to disk."""
        ani_time = time()
        print('Animating....',
              end='\r')
        self._setup_figure()
        self._setup_axes()
        self._setup_lines()
        self._setup_legend()
        self._animate()
        print('Animating....Done\n'
              'Time taken =  {} seconds'
              '\n'.format(round(time() - ani_time, 3)))
        return

    def _setup_figure(self):
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        self._figure = plt.figure(figsize=self.fig_size,
                                  dpi=self.dpi)
        return

    def _setup_axes(self):
        """Sets up Subplots (Position and Preferences)

        For each animated moment there is a separate coordinate system (axes)
        in which that moment is animated for all specimen.
        This Method specifies the position, the range of values
        and the title for each of these axes.
        """
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        # list of all subplots
        axes = []
        moments = self._cnf.animated_moments.flatten()
        shape = self._cnf.animated_moments.shape
        for (i_m, moment) in enumerate(moments):
            # subplots begin counting at 1
            ax = self._figure.add_subplot(shape[0], shape[1], i_m + 1)
            # set range of X-axis
            self._set_range(ax)
            # set range of Y-axis, based on occurring values in data
            self._set_val_limits(ax, moment)
            # give subplots titles
            ax.set_title(moment)
            self._set_tick_labels(ax, i_m)
            axes.append(ax)
        self._axes = np.array(axes)
        return

    def _set_range(self, ax):
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems'
            raise NotImplementedError(message)
        x_boundaries = self._cnf.p.boundaries
        x_min = x_boundaries[0, 0]
        x_max = x_boundaries[1, 0]
        ax.set_xlim(x_min, x_max)
        return

    def _set_val_limits(self, ax, moment):
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        min_val = 0
        max_val = 0
        for t in self._cnf.t.G:
            file_address = self._cnf.get_file_address(moment, 'npy', t)
            result_t = np.load(file_address)
            min_val = min(min_val, np.min(result_t))
            max_val = max(max_val, np.max(result_t))
        ax.set_ylim(1.25 * min_val, 1.25 * max_val)
        return

    def _set_tick_labels(self, ax, i_m):
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        shape = self._cnf.animated_moments.shape
        last_row = (shape[0]-1) * shape[1]
        if i_m < last_row:
            ax.set_xticklabels([])
        return

    def _setup_lines(self):
        """ Sets up Plot lines

        For each pair of (animated moment, specimen)
        there is a separate line to be plotted.
        Since ach one of those lines is a separate plot,
        it contains data which is updated for each new frame.
        This date is located in :attr:`_lines`.
        """
        if self._cnf.p.dim is not 1:
            message = 'Animation is currently only implemented ' \
                      'for 1D Problems' \
                      'This needs to be done for 3D plots'
            raise NotImplementedError(message)
        lines = []
        moments = self._cnf.animated_moments.flatten()
        for (m, moment) in enumerate(moments):
            moment_lines = []
            for s in range(self._cnf.s.n):
                # initialize line without any data
                line = self._axes[m].plot([],
                                          [],
                                          linestyle='-',
                                          color=self._cnf.s.colors[s],
                                          linewidth=2)
                # plot returns a tuple, we only need the first element
                line = line[0]
                moment_lines.append(line)
            lines.append(moment_lines)
        self._lines = np.array(lines)
        return

    def _setup_legend(self):
        # List of Species-Names
        names = self._cnf.s.names
        # Position of the Legend in the figure
        location = 'lower center'
        # number of columns in Legend
        n_columns = self._cnf.s.n
        self._figure.legend(self._lines[0, :],
                            names,
                            loc=location,
                            ncol=n_columns,
                            # expand legend horizontally
                            mode='expand',
                            borderaxespad=0.5)
        return

    def _animate(self, ):
        ani = animation.FuncAnimation(self._figure,
                                      self._update_data,
                                      fargs=(self._lines,),
                                      # Todo speed up!
                                      # init_func=init
                                      frames=self._cnf.t.size,
                                      interval=1,
                                      blit=False)
        if self._save_animation:
            ani.save(self.file_address,
                     self._writer,
                     dpi=self.dpi)
        else:
            plt.show()
        return

    def _update_data(self, t, lines):
        n_specimen = self._cnf.s.n
        moments = self._cnf.animated_moments.flatten()
        for (m, moment) in enumerate(moments):
            file_address = self._cnf.get_file_address(moment,
                                                      'npy',
                                                      self._cnf.t.G[t])
            result_t = np.load(file_address)
            for s in range(n_specimen):
                lines[m, s].set_data(self._cnf.p.G*self._cnf.p.d,
                                     result_t[s, :])
        return lines
