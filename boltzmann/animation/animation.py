
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


class Animation:
    r"""Determines Calculation Output and handles animation or results

    Attributes
    ----------
    save_animation : bool
        If True, render and save the animation into a video file, but don't
        play the video. If False, play the video, but don't save it
    writer : :class:`matplotlib.animation.FFMpegWriter`
    figure : :class:`matplotlib.figure.Figure`
        """
    def __init__(self,
                 cnf,
                 data):
        self.cnf = cnf
        # self.moments = self.cnf.animated_moments
        self.save_animation = False
        self.writer = animation.writers['ffmpeg'](fps=15, bitrate=1800)
        # Todo replace this by a filename property of Configuration class
        # Todo Add configuration attribute: path!
        # self.files = [self.cnf.file_name + '_' + mom
        #               for mom in self.cnf.animated_moments]
        self.figure = self.setup_figure()
        self.axes = self.setup_axes(data)
        self.lines = self.setup_plot_lines()
        self.setup_legend()
        return

    # Todo remove data - read from file
    def run(self, data):
        ani_time = time()
        self.animate(data)
        print("Animation - Done\n"
              "Time taken =  {} seconds"
              "".format(round(time() - ani_time, 3)))
        return

    def animate(self, data):
        print(data.shape)
        ani = animation.FuncAnimation(self.figure,
                                      self.update_data,
                                      fargs=(data,
                                             self.lines),
                                      # init_func=init   # Todo speeds up!
                                      frames=self.cnf.t.size,
                                      interval=1,
                                      blit=False)
        input()
        if self.save_animation:
            # Todo Do this properly
            print('add Saving Method!')
            # sav_t = time()
            # ani.save(cf_path + '_' + cf_name + '_animation.mp4',
            #          WRITER,
            #          dpi=300)
            # print('Saving Animation         -- %.3e s\n' % (time() - sav_t))
        else:
            plt.show("Press any Key: ")
        return

    def update_data(self, t, data, lines):
        for m in range(data.shape[1]):
            for s in range(data.shape[2]):
                lines[m, s].set_data(self.cnf.p.G*self.cnf.p.d,
                                     data[t, m, s, :])
        return lines

    def setup_figure(self):
        if self.cnf.p.dim is not 1:
            print("This is only defined for 1D Problems!")
            assert False
        # Todo put this into configuration
        # Todo add auto configuration
        fig_size = (16, 9)
        # Todo put this into configuration
        # Todo add auto configuration
        if self.save_animation:
            dpi = 300
        else:
            dpi = 100
        return plt.figure(figsize=fig_size, dpi=dpi)

    def setup_axes(self, data):
        """Sets up Subplots (Position and Preferences)

        For each animated moment there is a separate coordinate system (axes)
        in which that moment is animated for all specimen.
        This Method specifies the position, the range of values
        and the title for each of these axes.
        """
        if self.cnf.p.dim is not 1:
            print("This is only defined for 1-D Problems!")
            assert False
        # list of all subplots
        axes = []
        moments = self.cnf.animated_moments.flatten()
        for (i_m, name) in enumerate(moments):
            # Todo This grid shape should be customizable
            # add subplot in with shape as in cnf.animated_moments
            shape = self.cnf.animated_moments.shape
            ax = self.figure.add_subplot(shape[0], shape[1], i_m + 1)
            # set range of X-axis
            x_boundary = self.cnf.p.boundaries
            x_min = x_boundary[0]
            x_max = x_boundary[1]
            ax.set_xlim(x_min, x_max)
            # set range of Y-axis, based on occurring values in data
            # TODO check if there is something wrong here
            y_min = min(0, 1.25 * np.min(data[:, i_m, :, :]))
            y_max = 1.25 * np.max(data[:, i_m, :, :])
            ax.set_ylim(y_min, y_max)
            # give subplots titles
            ax.set_title(name)
            # Todo This depends on grid shape (soon customizable)
            # show X-axis values only on lowest row
            if i_m < 4:
                ax.set_xticklabels([])

            axes.append(ax)
        return np.array(axes)

    def setup_plot_lines(self):
        """ Sets up Plot lines

        For each animated moment there is a separate line for each specimen.
        Each one of those lines is a separate plot,
        containing data which is updated for each new frame.
        These plots are located in
        :attr:`boltzmann.animation.Animation.lines`,
        a structured array for easy addressing.
        """
        lines = []
        moments = self.cnf.animated_moments.flatten()
        for (m, name) in enumerate(moments):
            moment_lines = []
            for s in range(self.cnf.s.n):
                # initialize line without any data
                line = self.axes[m].plot([],
                                         [],
                                         linestyle='-',
                                         color=self.cnf.s.color[s],
                                         linewidth=2)
                # plot returns a tuple, we only need the first element
                line = line[0]
                moment_lines.append(line)
            lines.append(moment_lines)
        return np.array(lines)

    def setup_legend(self):
        self.figure.legend(self.lines[0, 0:self.cnf.s.n],
                           self.cnf.s.name,    # List of Species-Names
                           loc='lower center',  # Position Legend at the bottom
                           ncol=min(self.cnf.s.n, 8),   # use at most 8 columns
                           mode='expand',       # expand legend horizontally
                           borderaxespad=0.5)
        return
