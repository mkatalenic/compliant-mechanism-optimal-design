#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import sys

sys.path.append("..")

import numpy as np

from calculix_manipulation import (
    run_ccx,
)

import case_visualisation as cv
from helper_functions import (
    used_nodes_in_current_mesh_state,
    final_displacement_node_positions_in_ccx_results,
)

from geometry_creation import read_mesh_state, Mesh
from mesh_setup import MAXIMAL_MESH_VOLUME, BEAM_INTERFACE, MAXIMAL_BEAM_WIDTH

# Read initial mesh setup
used_mesh: Mesh = read_mesh_state('mesh_setup.pkl')

OPTIMIZATION_DIMENSIONS = len(
    [state for state in used_mesh.beam_width_beginning_map if state == True]
)

bm_counter = -1
beam_interface_positions = np.empty((0), dtype=int)
for beam_idx, beginning_map in enumerate(used_mesh.beam_width_beginning_map):
    if beginning_map == True:
        bm_counter +=1
    if beam_idx in BEAM_INTERFACE:
        beam_interface_positions = np.append(beam_interface_positions, beam_idx)


OPTIMIZATION_UPPER_BOUND = np.full((OPTIMIZATION_DIMENSIONS), MAXIMAL_BEAM_WIDTH)  # mm
OPTIMIZATION_UPPER_BOUND[beam_interface_positions] = MAXIMAL_BEAM_WIDTH
OPTIMIZATION_LOWER_BOUND = np.full((OPTIMIZATION_DIMENSIONS), 0)
OPTIMIZATION_LOWER_BOUND[beam_interface_positions] = MAXIMAL_BEAM_WIDTH
OPTIMIZATION_OBJECTIVES_AND_WEIGHTS = {
    "volume": 1e-3,
    "x_error": 0.2,
    "y_error": 0.8,
}

weights_sum = 0

for name, weight in OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.items():
    weights_sum += weight
for name, weight in OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.items():
    OPTIMIZATION_OBJECTIVES_AND_WEIGHTS[name] = weight/weights_sum

no_beginning_cases = 200
np.random.seed(140792)
randomised_beginning = np.random.random((no_beginning_cases, OPTIMIZATION_DIMENSIONS)) * OPTIMIZATION_UPPER_BOUND
randomised_beginning[:, beam_interface_positions] = MAXIMAL_BEAM_WIDTH

fitness_res = np.empty((0), dtype=float)

for case_id, case_widths in enumerate(randomised_beginning):
    used_mesh.set_width_array(case_widths)

    results = run_ccx(
        mesh=used_mesh,
        ccx_case_name=str(case_id),
        delete_after_completion=False,
        von_mises_instead_of_principal=True,
        ccx_name='ccx'
    )

    displacement, stress = results
    np.savez_compressed(
        f"ccx_files/{case_id}/data.npz",
        displacement=displacement,
        stress=stress,
        used_nodes_read=used_nodes_in_current_mesh_state(used_mesh),
    )
    current_volume = used_mesh.calculate_mechanism_volume()
    non_dim_volume = current_volume / MAXIMAL_MESH_VOLUME

    u_goal = used_mesh.final_displacement_array[:, 1:]
    u_calc = displacement[
        final_displacement_node_positions_in_ccx_results(used_mesh)
    ]
    error = (u_calc - u_goal) * 1e2

    # mask = np.abs(u_goal) > 0
    # error[mask] = error[mask] / u_goal[mask]

    # error = np.abs(error)
    error = error**2

    # x_error = np.average(error[:, 0])
    x_error = error[0, 0]
    y_error = np.average(error[:, 1])

    stress_constraint = stress.max() - 20e6
    if stress_constraint > 0:
        punish = 1
    else:
        punish = 0

    fitness = x_error * OPTIMIZATION_OBJECTIVES_AND_WEIGHTS['x_error'] +\
        y_error * OPTIMIZATION_OBJECTIVES_AND_WEIGHTS['y_error'] +\
        punish
    fitness_res = np.append(fitness_res, fitness)

    print(f'{case_id=}, {[non_dim_volume, x_error, y_error]}, {fitness=}')

np.savez_compressed(
    "random_res.npz",
    fitness=fitness_res,
    rand_widths=randomised_beginning,
)
