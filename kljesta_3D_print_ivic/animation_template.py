#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import re

import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv


mesh = cm.calculix_manipulator(
    gc.read_mesh_state()
)

# Remove unwanted beams (removes beam inside of polygon)
removed_beams = mesh.beam_laso(
    [
        (70e-3, -1e-3),
        (70e-3, 12e-3),
        (110e-3, 12e-3),
        (110e-3, -1e-3)
    ]
)

# Set thickness to zero for all beams
mesh.set_width_array(np.zeros(mesh.beam_array.shape[0]))

# Beam to be optimized (varying thickness)
used_beams = [
    x for x in range(mesh.beam_array.shape[0]) if x not in removed_beams
]


# potrebno postaviti za potrebe dohvaćanja podataka
mesh.used_mesh.beam_width_array[used_beams] = 5e-3

mesh.load_from_info(used_beams.size)

mesh.used_mesh.beam_height = 8e-3

mesh.load_best_ccx_solutions()

animation_iteratons = int(np.size(mesh.iteration_list))

drawer = cv.mesh_drawer()

drawer.from_object(mesh.used_mesh)
drawer.my_figure.suptitle('Optimizacija kljesta')

von_mis_stress_per_it = np.array(
    [cm.calculate_von_mises_stress(stress_per_it) for stress_per_it in mesh.calculated_stress]
)

max_von_misses_stress = np.max(von_mis_stress_per_it)
min_von_misses_stress = np.min(von_mis_stress_per_it)
print(max_von_misses_stress)
print(min_von_misses_stress)

def drawing_function(iterator):

    drawer.my_ax.clear()
    drawer.my_info_ax.clear()

    mesh.used_mesh.beam_width_array[used_beams] = \
        mesh.calculated_widths[iterator]

    optimization_iteration = mesh.iteration_list[iterator]

    displacement = mesh.calculated_displacement[iterator]
    stress = mesh.calculated_stress[iterator]

    u_goal = mesh.used_mesh.final_displacement_array[:, 1:]
    u_calc = displacement[mesh.final_displacement_node_positions]

    errors = u_calc - u_goal

    x_error = np.sum(np.abs(errors[:, 0]))
    y_error = np.sum(np.abs(errors[:, 1]))

    info_dict = {
        'Iteracija': int(optimization_iteration),
        'h' + '[' + r'$\mu m$' + ']': f'{mesh.used_mesh.beam_height:.5E}',  # visina mreže
        r'$dt_{j}$' + '[' + r'$\mu m$' + ']': f'{mesh.used_mesh.final_displacement_array[:, 1:][:, 1]}',  # Traženi pomaci
        r'$dd_{j}$' + '[' + r'$\mu m$' + ']': f'{displacement[:, 1][mesh.final_displacement_node_positions]}',  # Dobiveni pomaci
        r'$y_{err}=\left(\sum{dt_{j}-dd_{j}}\right)^2$': f'{y_error:.5E}',  # error
        r'$x_{err}=\left(\sum{dt_{j}-dd_{j}}\right)^2$': f'{x_error:.5E}',  # error
    }

    drawer.make_drawing(info_dict,
                        displacement,
                        stress,
                        (max_von_misses_stress, min_von_misses_stress),
                        displacement_scale=20)
    drawer.my_ax.set_title('Rezultati optimizacije')
    drawer.save_drawing(f'best_{int(optimization_iteration)}')
    print(f'Made up to: {optimization_iteration} iteration' +
          f'\nMax y displacement: {displacement[:, 1][mesh.final_displacement_node_positions]}')

animation = anim.FuncAnimation(drawer.my_figure,
                               drawing_function,
                               animation_iteratons,
                               interval=300)

animation.save('probni.mp4')
