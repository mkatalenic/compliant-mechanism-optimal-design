#!/usr/bin/env python3

import sys
import os
import shutil

import numpy as np

sys.path.append("..")

from scipy.optimize import minimize
from scipy.optimize import Bounds

import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

mesh = gc.SimpleMeshCreator((2980e6, 0.2),
                            10e-3,
                            100e-3,
                            25e-3,
                            (6, 2),
                            'x')

mesh.minimal_beam_width = 2e-3
mesh.beam_height = 8e-3


mesh.set_width_array(
    np.ones(
        np.shape(
            mesh.beam_array
        )[0]
    ) * 5e-3
)

max_volume = mesh.calculate_mechanism_volume()

mesh.set_width_array(
    np.zeros(
        np.shape(
            mesh.beam_array
        )[0]
    )
)

# remove unwanted beams
removed_beams = mesh.beam_laso(
    [
        (70e-3, -1e-3),
        (70e-3, 12e-3),
        (110e-3, 12e-3),
        (110e-3, -1e-3)
    ]
)

used_beams = [
    x for x in range(np.shape(mesh.beam_array)[0]) if x not in removed_beams
]

for node in mesh.node_laso(
        [
            (-1e-3, 1e-3),
            (67e-3, 1e-3),
            (67e-3, -1e-3),
            (-1e-3, -1e-3)
        ],
        only_main_nodes=False):
    mesh.create_boundary(node,
                         (0, 1, 1),
                         set_unremovable=True)

mesh.create_boundary(
    (0, 25e-3),
    (1, 1, 0),
    set_unremovable=True
)

mesh.create_initial_displacement(
    (0, 0),
    (5e-3, 0)
)

# mesh.create_force(
#     (0, 0),
#     (10, 0)
# )

mesh.set_final_displacement(
    (3/2*100e-3, 0),
    (6e-3, 0)
)

mesh.set_final_displacement(
    (6/5*100e-3, 12.5e-3),
    (0, -6e-3)
)

mesh.set_final_displacement(
    (100e-3, 12.5e-3),
    (-2e-3, -8e-3)
)

mesh.write_beginning_state()
ccx_manipulator = cm.calculix_manipulator(mesh)

import random
import string


def generateRandomAlphaNumericString(length):
    # Generate alphanumeric string
    letters = string.ascii_lowercase + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str

# drawer = cv.mesh_drawer()
# drawer.from_object(ccx_manipulator.used_mesh)
# drawer.my_figure.suptitle('Optimizacija klješta')


def min_fun(beam_widths):
    penalty = 0
    out_weighted_sum = 0

    ccx_manipulator.used_mesh.beam_width_array[used_beams] = beam_widths
    ccx_results = ccx_manipulator.run_ccx(
        generateRandomAlphaNumericString(20),
        delete_after_completion=True,
        von_mises_instead_of_principal=True
    )

    if ccx_results is False:
        penalty += 5e2

    else:
        (displacement, vm_stress) = ccx_results

        volume_weight = 0.05

        error_weights = (1 - volume_weight)/np.size(
            ccx_manipulator.final_displacement_node_positions
        )

        errors = ccx_manipulator.used_mesh.final_displacement_array[:, 1:] -\
            displacement[ccx_manipulator.final_displacement_node_positions]

        mechanism_volume = ccx_manipulator.used_mesh.calculate_mechanism_volume()

        dimensionless_vol_w_weights = mechanism_volume/max_volume*volume_weight

        x_errors_w_weights = np.sum(errors[:, 0]**2) * error_weights / 100e-3 * 6
        y_errors_w_weights = np.sum(errors[:, 1]**2) * error_weights / 25e-3 * 2

        out_weighted_sum += x_errors_w_weights + y_errors_w_weights
        out_weighted_sum += dimensionless_vol_w_weights

        print(50*'-')
        print(f'mechanism_volume: {mechanism_volume}')
        print(f'weighted_sum: {out_weighted_sum}')
        print('widths:')
        print(beam_widths)
        print(50*'-')

    return out_weighted_sum + penalty


x0 = np.ones(
    np.size(used_beams)
)

x0 = np.random.random(np.size(x0)) * 4e-3 + 1.8e-3

res = minimize(min_fun,
               x0,
               method='nelder-mead',
               options={'disp': True,
                        'return_all': True})

print(res.x)
