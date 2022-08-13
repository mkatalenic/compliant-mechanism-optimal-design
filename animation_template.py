#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import re

import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv


example_mesh = cm.calculix_manipulator(
    gc.read_mesh_state()
)

debele = [112, 113, 115, 114, 116, 111, 148, 118, 121, 119, 122, 117,
          75, 76, 78, 77, 74, 79, 81, 82, 83, 84, 85, 87, 80, 41, 40,
          42, 44, 47, 46, 45, 48, 43, 50, 51, 53, 54, 49, 11, 13, 14, 12,
          17, 16, 15, 19, 20, 21, 63, 64, 26, 27, 58, 100, 106, 107,
          145, 146, 151, 133, 134, 132, 136, 139, 135, 137]

tanke = [153, 144, 147, 138, 101, 57, 9, 18, 23, 28, 66, 142, 60, 68]

example_mesh.used_mesh.set_width_array(
    np.zeros(
        np.shape(
            example_mesh.used_mesh.beam_array
        )[0]
    )
)

# potrebno postaviti za potrebe dohvaćanja podataka
example_mesh.used_mesh.beam_width_array[debele] = 5
example_mesh.used_mesh.beam_width_array[tanke] = 5

example_mesh.load_from_info(len(tanke))

example_mesh.used_mesh.beam_height = 5.8

example_mesh.load_best_ccx_solutions()

animation_iteratons = int(np.size(example_mesh.iteration_list))

drawer = cv.mesh_drawer()

drawer.from_object(example_mesh.used_mesh)
drawer.my_figure.suptitle('Optimizacija primjera iz literature')

von_mis_stress_per_it = np.array(
    [cm.calculate_von_mises_stress(stress_per_it) for stress_per_it in example_mesh.calculated_stress]
)

max_von_misses_stress = np.max(von_mis_stress_per_it)
min_von_misses_stress = np.min(von_mis_stress_per_it)
print(max_von_misses_stress)
print(min_von_misses_stress)

def drawing_function(iterator):

    drawer.my_ax.clear()
    drawer.my_info_ax.clear()

    example_mesh.used_mesh.beam_width_array[tanke] = \
        example_mesh.calculated_widths[iterator]

    optimization_iteration = example_mesh.iteration_list[iterator]

    displacement = example_mesh.calculated_displacement[iterator]
    stress = example_mesh.calculated_stress[iterator]

    y_error = np.sum(
        (example_mesh.used_mesh.final_displacement_array[:, 1:][:, 1] -
         displacement[example_mesh.final_displacement_node_positions][:, 1])**2,
        axis=0
    )

    io_ratio = example_mesh.used_mesh.final_displacement_array[:, 1:][0, 1] / displacement[example_mesh.force_node_positions[0]][1]

    # print(f'displ_final: {example_mesh.used_mesh.final_displacement_array[:, 1:][0, 1]}')
    # print(f'displ: {example_mesh.used_mesh.final_displacement_array[:, 1:][0, 1]}')
    # print(f'ratio: {io_ratio}')

    info_dict = {
        'Iteracija': int(optimization_iteration),
        'h' + '[' + r'$\mu m$' + ']': f'{example_mesh.used_mesh.beam_height:.5E}',  # visina mreže
        r'$dt_{j}$' + '[' + r'$\mu m$' + ']': f'{example_mesh.used_mesh.final_displacement_array[:, 1:][:, 1]}',  # Traženi pomaci
        r'$dd_{j}$' + '[' + r'$\mu m$' + ']': f'{displacement[:, 1][example_mesh.final_displacement_node_positions]}',  # Dobiveni pomaci
        r'$y_{err}=\left(\sum{dt_{j}-dd_{j}}\right)^2$': f'{y_error:.5E}',  # error
        'output/input ratio r': f'{io_ratio:.2f}'  # output to input ratio
    }

    drawer.make_drawing(info_dict,
                        displacement,
                        stress,
                        (max_von_misses_stress, min_von_misses_stress),
                        displacement_scale=20)
    drawer.my_ax.set_title('Rezultati optimizacije')
    drawer.save_drawing(f'best_{int(optimization_iteration)}')
    print(f'Made up to: {optimization_iteration} iteration' +
          f'\nMax y displacement: {displacement[:, 1][example_mesh.final_displacement_node_positions]}')

animation = anim.FuncAnimation(drawer.my_figure,
                               drawing_function,
                               animation_iteratons,
                               interval=300)

animation.save('probni.mp4')
