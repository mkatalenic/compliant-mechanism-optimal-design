#!/usr/bin/env python3

'''
   >==> >=>     >=>
 >=>     >=>   >=>
>=>       >=> >=>
 >=>       >=>=>
   >==>     >=>

TODO vrste prikaza
TODO prikaz čiste mreže
TODO prikaz deformirane
TODO prikaz naprezanja
TODO prikaz funkcije cilja
'''

import numpy as np
import matplotlib.pyplot as plt

import geometry_creation as gc


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([], [], **kwargs)
        if "label" in kwargs:
            kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda: self.fig.canvas.draw_idle())
        self.timer.start()


class draw_mesh():

    # plt.style.use('dark_background')
    plt.style.use('fast')
    my_figure = plt.figure()
    my_ax = my_figure.add_subplot(1, 1, 1)
    my_ax.set_aspect('equal')

    @classmethod
    def from_object(self,
                    mesh: gc.Mesh):
        '''Mesh drawing setup from the given mesh object'''

        self._is_from_object = True

        self.used_mesh = mesh

    @classmethod
    def from_file(self):
        '''Mesh drawing setup from the given file structure'''

        self._is_from_object = False
        pass

    def make_drawing(self):
        x_lim_10_perc = (np.max(self.used_mesh.node_array[:, 0]) -
                         np.min(self.used_mesh.node_array[:, 0])) * 0.1
        y_lim_10_perc = (np.max(self.used_mesh.node_array[:, 1]) -
                         np.min(self.used_mesh.node_array[:, 1])) * 0.1
        self.my_ax.set_xlim([
            np.min(self.used_mesh.node_array[:, 0]) - x_lim_10_perc,
            np.max(self.used_mesh.node_array[:, 0]) + x_lim_10_perc
        ])
        self.my_ax.set_ylim([
            np.min(self.used_mesh.node_array[:, 1]) - y_lim_10_perc,
            np.max(self.used_mesh.node_array[:, 1]) + y_lim_10_perc
        ])
        used_beams = np.array(
            [i for i, _ in enumerate(
                self.used_mesh.beam_width_array
            ) if _ != 0]
        )

        used_nodes = np.empty((0),
                              dtype=np.int32)

        for beam_idx in used_beams:
            used_nodes = np.append(
                used_nodes,
                self.used_mesh._fetch_beam_nodes(beam_idx)
            )

        used_nodes = np.unique(used_nodes)

        all_nodes_coordinates = self.used_mesh.node_array

        # Plot beams
        for beam, width in zip(used_beams,
                               self.used_mesh.beam_width_array[used_beams]):

            nodes_per_beam = self.used_mesh._fetch_beam_nodes(beam)

            data_linewidth_plot(all_nodes_coordinates[nodes_per_beam][:, 0],
                                all_nodes_coordinates[nodes_per_beam][:, 1],
                                ax=self.my_ax,
                                linewidth=width,
                                color='purple',
                                zorder=1)

            self.my_ax.text(
                (
                    all_nodes_coordinates[nodes_per_beam[0]][0]
                    +
                    all_nodes_coordinates[nodes_per_beam[-1]][0]
                ) / 2,
                (
                    all_nodes_coordinates[nodes_per_beam[0]][1]
                    +
                    all_nodes_coordinates[nodes_per_beam[-1]][1]
                ) / 2,
                f'b_{beam}',
                fontweight='black',
                fontsize='small',
                horizontalalignment='center'
            )

        # Plot nodes and boundaries
        for node in np.intersect1d(self.used_mesh.main_nodes, used_nodes):
            if node in self.used_mesh.boundary_array[:, 0]:
                DOF_to_print = str(
                    self.used_mesh.boundary_array[
                        self.used_mesh.boundary_array[:, 0] == node
                    ][:, 1:]
                )[2:-2]
                self.my_ax.scatter(all_nodes_coordinates[node][0],
                                   all_nodes_coordinates[node][1],
                                   s=300,
                                   color='green',
                                   marker=f'${DOF_to_print}$',
                                   edgecolor='red',
                                   zorder=1)
            else:
                self.my_ax.scatter(all_nodes_coordinates[node][0],
                                   all_nodes_coordinates[node][1],
                                   s=30,
                                   color='green',
                                   marker='o',
                                   edgecolor='red',
                                   zorder=1)

        # Plot forces
        for force in self.used_mesh.force_array:
            resultant_force = np.sqrt(np.sum(force[1:]**2))
            direction = force[1:] / resultant_force
            dx, dy = -1 * (
                (np.max(self.used_mesh.node_array) -
                 np.min(self.used_mesh.node_array)) * direction / 10
            )
            self.my_ax.arrow(all_nodes_coordinates[int(force[0])][0] + dx,
                             all_nodes_coordinates[int(force[0])][1] + dy,
                             - dx,
                             - dy,
                             width=0.05,
                             head_width=0.1,
                             length_includes_head=True,
                             color='red')
            self.my_ax.text(all_nodes_coordinates[int(force[0])][0] + dx,
                            all_nodes_coordinates[int(force[0])][1] + dy,
                            f'{resultant_force:.1E}N',
                            fontweight='black',
                            fontsize='small',
                            horizontalalignment='center')

        # Plot initial displacement
        for init_disp in self.used_mesh.initial_displacement_array:
            self.my_ax.arrow(all_nodes_coordinates[int(init_disp[0])][0],
                             all_nodes_coordinates[int(init_disp[0])][1],
                             init_disp[1],
                             init_disp[2],
                             width=0.05,
                             head_width=0.3,
                             color='red',
                             zorder=0)
