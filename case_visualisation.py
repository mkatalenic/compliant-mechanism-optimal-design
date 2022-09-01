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
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

import geometry_creation as gc
import calculix_manipulation as cm


plt.style.use('dark_background')
class mesh_drawer():

    my_figure = plt.figure(dpi=400, figsize=(10, 10))
    subfigs = my_figure.subfigures(2, 1, wspace=0.02, height_ratios=[2, 1])

    upper_subfig = subfigs[0].subfigures(1,2,wspace=0.02,width_ratios=[2, 1])
    my_ax = upper_subfig[0].add_subplot(1, 1, 1)
    my_info_ax = upper_subfig[1].add_subplot(1, 1, 1)
    my_ax.set_aspect('equal')

    lower_subfig = subfigs[1]
    my_res_ax = lower_subfig.add_subplot(1, 1, 1)
    my_fitness_ax = my_res_ax.twinx()

    make_drawing_counter = 0

    @classmethod
    def from_object(self,
                    mesh: gc.Mesh):
        '''Mesh drawing setup from the given mesh object'''

        self._is_from_object = True

        self.used_mesh = mesh

    def plot_obj_constr_fitness(self,
                                iteration,
                                objectives,
                                constraints,
                                fitness):

        if iteration == 0:

            self.iteration_tracker = np.array([iteration])
            self.objectives_tracker = np.array([objectives])
            self.constraints_tracker = np.array([constraints[1:]])
            self.fitness_tracker = np.array([fitness])

        if iteration != 0:

            self.iteration_tracker = np.append(self.iteration_tracker, [iteration], axis=0)
            self.objectives_tracker = np.append(self.objectives_tracker, [objectives], axis=0)
            self.constraints_tracker = np.append(self.constraints_tracker, [constraints[1:]], axis=0)
            self.fitness_tracker = np.append(self.fitness_tracker, [fitness], axis=0)

        it_arr = self.iteration_tracker
        obj_arr = self.objectives_tracker
        const_arr = self.constraints_tracker
        fit_arr = self.fitness_tracker

        self.my_res_ax.set_xlabel('iteration')
        self.my_res_ax.set_ylabel('Value')

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        const_colors = colors[:const_arr.shape[1]]
        obj_colors = colors[const_arr.shape[1]:const_arr.shape[1] + obj_arr.shape[1]]

        # for const_id in range(const_arr.shape[1]):
        #     self.my_res_ax.plot(
        #         it_arr,
        #         const_arr[:,const_id],
        #         color = const_colors[const_id],
        #         label=f'C{const_id}'
        #     )
        self.my_res_ax.tick_params(axis='y')
        for obj_id in range(obj_arr.shape[1]):
            self.my_res_ax.plot(
                it_arr,
                obj_arr[:,obj_id],
                color = obj_colors[obj_id],
                label=f'O{obj_id}'
            )
        self.my_res_ax.legend(bbox_to_anchor=(1., 1.05), loc='lower right')

        self.my_fitness_ax.set_ylabel('Fitness', color='red')  # we already handled the x-label with ax1
        self.my_fitness_ax.plot(it_arr,
                                fit_arr,
                                color='red')
        self.my_fitness_ax.tick_params(axis='y')
        self.my_fitness_ax.set_yscale('log')
        self.my_res_ax.set_yscale('log')


    def make_drawing(self,
                     info_dict: dict,
                     used_mesh=None,
                     displacement=None,
                     stress=None,
                     used_nodes_idx=None,
                     stress_max_min=(None, None),
                     displacement_scale=1,
                     beam_names=False):

        self.make_drawing_counter += 1

        text_array = np.empty((0, 2),
                              dtype=object)

        for i, (key, value) in enumerate(info_dict.items()):
            text_array = np.append(
                text_array,
                [[f'{key}', f'{value}']],
                axis=0
            )

        cellcolours = [['purple', 'black'] for _ in range(np.shape(text_array)[0])]

        info_table = self.my_info_ax.table(cellText=text_array,
                                           loc='center',
                                           cellColours=cellcolours,
                                           edges='closed')

        info_table.auto_set_font_size(False)
        info_table.set_fontsize(10)

        self.my_ax.grid(False)

        x_lim = (np.max(self.used_mesh.node_array[:, 0]) -
                         np.min(self.used_mesh.node_array[:, 0])) * 0.2
        y_lim = (np.max(self.used_mesh.node_array[:, 1]) -
                         np.min(self.used_mesh.node_array[:, 1])) * 0.2

        if x_lim > y_lim:
            y_lim = x_lim
        else:
            x_lim = y_lim

        self.my_ax.set_xlim([
            np.min(self.used_mesh.node_array[:, 0]) - x_lim,
            np.max(self.used_mesh.node_array[:, 0]) + x_lim
        ])
        self.my_ax.set_ylim([
            np.min(self.used_mesh.node_array[:, 1]) - y_lim,
            np.max(self.used_mesh.node_array[:, 1]) + y_lim
        ])

        used_beams = np.array(
            [i for i, _ in enumerate(
                self.used_mesh.beam_width_array
            ) if float(_) != 0.]
        )

        used_nodes = np.unique(used_nodes_idx)


        if displacement is None:
            displacement = np.zeros((np.size(used_nodes), 2))

        undeformed_node_coordinates = self.used_mesh.node_array
        for beam, _ in enumerate(self.used_mesh.beam_array):
            nodes_per_beam = self.used_mesh._fetch_beam_nodes(beam)
            self.my_ax.plot(undeformed_node_coordinates[nodes_per_beam][:, 0],
                            undeformed_node_coordinates[nodes_per_beam][:, 1],
                            color='w',
                            linestyle='dashdot',
                            linewidth=1,
                            alpha=0.7,
                            zorder=-1)

        if stress is not None:
            jet = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=stress_max_min[1],
                                     vmax=stress_max_min[0])

            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            if self.make_drawing_counter == 1:
                self.my_figure.colorbar(scalarMap,
                                        ax=self.my_ax,
                                        location='bottom',
                                        aspect=30)

            all_nodes_stress = np.zeros(self.used_mesh.node_array.shape[0])
            # print(f'{all_nodes_stress.shape=}')
            # print(f'{stress.shape=}')
            # print(f'{displacement.shape=}')
            # print(f'{used_nodes.shape=}')
            all_nodes_stress[used_nodes] += cm.calculate_von_mises_stress(stress)

        all_nodes_coordinates = np.zeros_like(self.used_mesh.node_array)
        all_nodes_coordinates[used_nodes] = displacement * displacement_scale
        all_nodes_coordinates += self.used_mesh.node_array

        # Plot beams
        for beam, width in zip(used_beams,
                               self.used_mesh.beam_width_array[used_beams]):

            nodes_per_beam = self.used_mesh._fetch_beam_nodes(beam)

            ppd = 20./self.my_figure.dpi
            trans = self.my_ax.transData.transform
            lw = ((trans((1, width))-trans((0, 0)))*ppd)[1]

            if stress is not None:
                lc = mcoll.LineCollection(
                    [all_nodes_coordinates[nodes_per_beam]],
                    linewidths=lw,
                    colors=scalarMap.to_rgba(all_nodes_stress[nodes_per_beam]),
                    zorder=1
                )

            else:
                lc = mcoll.LineCollection(
                    [all_nodes_coordinates[nodes_per_beam]],
                    linewidths=lw,
                    colors=['black' for _ in nodes_per_beam],
                    zorder=1
                )

            self.my_ax.add_collection(lc)


            # plots beam names on beams
            if beam_names:
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
                    color='red',
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
            # else:
            #     self.my_ax.scatter(all_nodes_coordinates[node][0],
            #                        all_nodes_coordinates[node][1],
            #                        s=30,
            #                        color='green',
            #                        marker='o',
            #                        edgecolor='red',
            #                        zorder=1)

        # Plot forces
        for force in self.used_mesh.force_array:
            resultant_force = np.sqrt(np.sum(force[1:]**2))
            direction = force[1:] / resultant_force
            dx, dy = -1 * (
                (np.max(self.used_mesh.node_array) -
                 np.min(self.used_mesh.node_array)) * direction / 10
            )

            arrow_width = np.sum([
                - np.min(self.used_mesh.node_array[:, 0]) - x_lim,
                np.max(self.used_mesh.node_array[:, 0]) + x_lim
            ]) * 0.008

            self.my_ax.arrow(all_nodes_coordinates[int(force[0])][0] + dx,
                             all_nodes_coordinates[int(force[0])][1] + dy,
                             - dx,
                             - dy,
                             width=arrow_width,
                             head_width=arrow_width * 1.5,
                             length_includes_head=True,
                             color='red')

#             self.my_ax.text(f'{resultant_force:.2f}',
#                                 xy=(all_nodes_coordinates[int(force[0])][0],
#                                     all_nodes_coordinates[int(force[0])][1]),
#                                 xytext=(dx/2,
#                                         dy/2),
#                                 textcoords='offset points')

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

        self.my_info_ax.set_axis_off()

    def save_drawing(self,
                     name: str):

        if not os.path.exists(
                os.path.join(
                    os.getcwd(),
                    'img'
                )
        ):
            os.mkdir(
                os.path.join(
                    os.getcwd(),
                    'img'
                )
            )

        self.my_figure.savefig(
            os.path.join(
                os.getcwd(),
                'img',
                f'{name}.jpg'
            ),
            dpi=400)
