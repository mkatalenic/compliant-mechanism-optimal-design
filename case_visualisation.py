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


class draw_mesh():

    plt.style.use('dark_background')
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

            self.my_ax.plot(all_nodes_coordinates[nodes_per_beam][:, 0],
                            all_nodes_coordinates[nodes_per_beam][:, 1],
                            '-',
                            color='white',
                            lw=width / self.used_mesh.minimal_beam_width / 10,
                            zorder=0)

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
                             head_width=0.3,
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
