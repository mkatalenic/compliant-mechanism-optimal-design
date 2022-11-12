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

# Optimization parameters
OPTIMIZATOR_NAME = "GGS"
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
OPTIMIZATION_LOWER_BOUND = np.full((OPTIMIZATION_DIMENSIONS), 0.)
OPTIMIZATION_LOWER_BOUND[beam_interface_positions] = MAXIMAL_BEAM_WIDTH
OPTIMIZATION_ITERATIONS = 100
OPTIMIZATION_MAXIMUM_EVALUATIONS = 200000
OPTIMIZATION_OBJECTIVES_AND_WEIGHTS = {
    "volume": 1e-3,
    "x_error": 0.2,
    "y_error": 0.8,
}

OPTIMIZATION_CONSTRAINTS_LABELS = ["invalid_simulation", "stress"]
OPTIMIZER_SPECIFIC_PARAMETERS = {"n": 21, "k_max": 2}

rand_res = np.load("random_res.npz")
all_rand_fitness = rand_res['fitness']
all_rand_fitness_sorted = np.sort(all_rand_fitness)

m = np.empty_like(all_rand_fitness, dtype=int)
for idx_in_non_sort, fit in enumerate(all_rand_fitness):
    for idx_in_sort, fit_sort in enumerate(all_rand_fitness_sorted):
        if fit == fit_sort:
            m[idx_in_non_sort] = idx_in_sort

all_rand_widths = rand_res['rand_widths']
all_rand_widths_sorted = all_rand_widths[m]

BEGINNING_DESIGN_VECTOR = all_rand_widths_sorted[3]

BEGINNING_DESIGN_VECTOR[beam_interface_positions] = MAXIMAL_BEAM_WIDTH
OPTIMIZATION_STARTING_DESIGN_VECTOR = BEGINNING_DESIGN_VECTOR
FAILED_OPTIMIZATION_RETURN = (np.nan, np.nan, np.nan, 1, np.nan)


def minimization_function(beam_widths, unique_str="", debug=False):
    if os.path.exists(os.path.join(os.getcwd(), "ccx_files", unique_str)):
        shutil.rmtree(os.path.join(os.getcwd(), "ccx_files", unique_str))

    ccx_results: tuple[np.ndarray, np.ndarray] | bool

    if used_mesh.set_width_array(beam_widths) is False:
        return FAILED_OPTIMIZATION_RETURN
    else:
        results = run_ccx(
            mesh=used_mesh,
            ccx_case_name=unique_str,
            delete_after_completion=False,
            von_mises_instead_of_principal=True,
            ccx_name='ccx'
        )

        if results is None:
            return FAILED_OPTIMIZATION_RETURN

        else:
            displacement, stress = results
            np.savez_compressed(
                f"ccx_files/{unique_str}/data.npz",
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

            stress_constraint = stress.max() - 100e6

            if debug:
                print(f"{u_goal=}")
                print(f"{u_calc=}")
                print(f"{displacement=}")
                print(
                    f"{final_displacement_node_positions_in_ccx_results(used_mesh)=}"
                )

        return (
            non_dim_volume,
            x_error,
            y_error,
            0,
            stress_constraint,
        )


# postavke animacije
drawer = cv.mesh_drawer(used_for_printig=True)
drawer.from_object(used_mesh)

TEST_DIR = "./ccx_files"


def post_iteration_processing(it, candidates, best):
    if candidates[0] <= best:
        if os.path.exists(f"{TEST_DIR}/best_it{it}"):
            shutil.rmtree(f"{TEST_DIR}/best_it{it}")

        os.rename(f"{TEST_DIR}/{candidates[0].unique_str}", f"{TEST_DIR}/best_it{it}")

        # Log keeps track of new best solutions in each iteration
        with open(f"{TEST_DIR}/log.txt", "a") as log:
            X = ", ".join(f"{x:13.6e}" for x in candidates[0].X)
            O = ", ".join(f"{o:13.6e}" for o in candidates[0].O)
            C = ", ".join(f"{c:13.6e}" for c in candidates[0].C)
            log.write(
                f"{it:6d} X:[{X}], O:[{O}], C:[{C}]"
                + f" fitness:{candidates[0].f:13.6e}\n"
            )

        drawer.my_ax.clear()
        drawer.my_info_ax.clear()
        drawer.my_res_ax.clear()
        drawer.my_fitness_ax.clear()

        npz = np.load(f"{TEST_DIR}/best_it{it}/data.npz")

        kljesta_info = {
            "Iteracija": int(it),
            "h " + "[m]": f"{used_mesh.beam_height:.5E}",
            "Volumen": f"{candidates[0].O[0]*100:.2f}%",
            "x error": f"{candidates[0].O[1]*100:.2f}%",
            "y error": f"{candidates[0].O[2]*100:.2f}%",
        }

        # w = np.full(used_mesh.beam_array.shape[0], 0, dtype=float)
        # w[used_nodes_in_current_mesh_state(used_mesh)] = candidates[0].X
        used_mesh.set_width_array(candidates[0].X)

        drawer.make_drawing(
            kljesta_info,
            npz["displacement"],
            npz["stress"],
            npz["used_nodes_read"],
            (10e6, 0),
            displacement_scale=1,
            beam_names=False,
        )

        drawer.plot_obj_constr_fitness(
            it, candidates[0].O, candidates[0].C, candidates[0].f, ['volumen', 'x error', 'y error']
        )

        drawer.save_drawing(f"best_{it}")

        drawer.check_and_make_copies_best()

    # Remove the best from candidates
    # (since its directory is already renamed)
    candidates = np.delete(candidates, 0)

    # Remove candidates' directories
    for c in candidates:
        if os.path.exists(f"{TEST_DIR}/{c.unique_str}"):
            try:
                shutil.rmtree(f"{TEST_DIR}/{c.unique_str}")
            except:
                pass

    return


if __name__ == "__main__":
    print(type(used_mesh.beam_width_array))
    print([used_mesh.beam_width_array > 0.0])
    print(f'{used_mesh.beam_width_beginning_map=}')
