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

        used_nodes_coordinates = self.used_mesh.node_array[used_nodes][:, :-1]

        all_nodes_coordinates = self.used_mesh.node_array[:, :-1]
