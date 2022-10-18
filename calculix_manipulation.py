#!/usr/bin/env python3

"""
   >==>    >===>>=>>==>
 >=>       >=>  >>  >=>
>=>        >=>  >>  >=>
 >=>       >=>  >>  >=>
   >==>    >==>  >>  >=>
"""
from __future__ import annotations

import os
import re
import subprocess
from shutil import rmtree

import numpy as np

from geometry_creation import Mesh

from my_typing import Stress

# Import translation functions
from helper_functions import (
    translate_node_coordinates,
    translate_beam_definitions,
    translate_material,
    translate_beam_cross_section,
    set_2D_calculix_case,
    translate_mesh_boundaries,
    translate_initial_displacement,
    use_nonlinear_solver,
    translate_mesh_forces,
    calculix_input_file_tail,
)

# Import other used functions
from helper_functions import (
    used_nodes_in_current_mesh_state,
    ccx_results_to_disp_stress_converter,
    calculate_von_mises_stress,
)


def translate_mesh(mesh: Mesh, nonlinear_calculation: bool = True) -> str:
    """Translates mesh object to ccx_input format"""
    return "".join(
        [
            translate_node_coordinates(mesh),
            translate_beam_definitions(mesh),
            translate_material(mesh),
            translate_beam_cross_section(mesh),
            set_2D_calculix_case(mesh),
            translate_mesh_boundaries(mesh),
            translate_initial_displacement(mesh),
            use_nonlinear_solver(nonlinear_calculation),
            translate_mesh_forces(mesh),
            calculix_input_file_tail(),
        ]
    )


def write_to_ccx_input_file(mesh: Mesh, ccx_case_name: str) -> None:
    """Writes translated mesh to ccx input file"""

    if not os.path.exists(os.path.join(os.getcwd(), "ccx_files")):
        try:
            os.mkdir(os.path.join(os.getcwd(), "ccx_files"))
        except:
            pass
    os.mkdir(os.path.join(os.getcwd(), "ccx_files", ccx_case_name))

    with open(
        os.path.join(os.getcwd(), "ccx_files", ccx_case_name, f"{ccx_case_name}.inp"),
        "w",
    ) as input_file:
        input_file.writelines(translate_mesh(mesh))


def read_ccx_results(
    mesh: Mesh, ccx_case_path: str, von_mises: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reads ccx results from a .frd file"""

    no_of_used_nodes = used_nodes_in_current_mesh_state(mesh).size
    displacement_array = np.empty(shape=(0, 3), dtype=np.float64)
    stress_array = np.empty(shape=(0, 6), dtype=np.float64)

    if os.path.split(ccx_case_path)[-1].endswith(".frd"):
        case_file = ccx_case_path

    else:
        case_file = os.path.join(
            ccx_case_path, f"{os.path.split(ccx_case_path)[-1]}.frd"
        )

    with open(case_file, "r") as results_file:
        displacement_list, stress_list = ccx_results_to_disp_stress_converter(
            results_file
        )

    if len(displacement_list) == 0 or len(stress_list) == 0:
        return None

    else:
        for displacement_per_node, stress_per_node in zip(
            displacement_list, stress_list
        ):
            displacement_array = np.append(
                displacement_array,
                np.reshape(np.array(displacement_per_node, dtype=np.float64), (1, 3)),
                axis=0,
            )
            stress_array = np.append(
                stress_array,
                np.reshape(np.array(stress_per_node, dtype=np.float64), (1, 6)),
                axis=0,
            )

        if von_mises:
            von_mises_eq_stress_array = np.empty((0), dtype=float)
            for stress_per_node in stress_array[-no_of_used_nodes:]:
                sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz = [
                    stress_per_node[i] for i in range(6)
                ]

                von_mises_eq_stress = calculate_von_mises_stress(
                    Stress(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz)
                )

                von_mises_eq_stress_array = np.append(
                    von_mises_eq_stress_array, von_mises_eq_stress
                )

            return (
                displacement_array[:, :-1][-no_of_used_nodes:],
                von_mises_eq_stress_array,
            )

        else:
            return (
                displacement_array[:, :-1][-no_of_used_nodes:],
                stress_array[-no_of_used_nodes:],
            )


def run_ccx(
    mesh: Mesh,
    ccx_case_name: str,
    delete_after_completion: bool = False,
    von_mises_instead_of_principal: bool = True,
    ccx_name: str = "ccx",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs ccx and fetches results using read_results()"""
    ccx_file_path = os.path.join(os.getcwd(), "ccx_files", ccx_case_name)

    write_to_ccx_input_file(mesh, ccx_case_name)

    with subprocess.Popen(
        [ccx_name, ccx_case_name],
        cwd=ccx_file_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        out, err = process.communicate()

    for line in str(out).replace("\\n", "\n").split("\n"):
        if line.startswith(" *ERROR") or len(err) != 0:
            return None  # U slučaju propale analize

    results = read_ccx_results(
        mesh, ccx_file_path, von_mises=von_mises_instead_of_principal
    )

    if delete_after_completion:
        rmtree(ccx_file_path)

    if results is False:
        return None
    else:
        return results  # U slučaju uspješne analize


def load_from_info(
    mesh: Mesh,
    widths_size: int | None = None,
    include_height: bool = False,
    log_txt_location: str = "ccx_files",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Loads results from the log"""

    calculated_heights = np.empty(0, dtype=float)
    iteration_list = np.empty(0, dtype=float)

    if widths_size is None:
        calculated_widths = np.empty((0, mesh.beam_width_array.shape[1]))
    else:
        calculated_widths = np.empty((0, widths_size), dtype=float)

    with open(
        os.path.join(os.getcwd(), log_txt_location, "log.txt"),
    ) as log_file:
        for case in log_file:
            iteration_list = np.append(iteration_list, int(case.split()[0]))
            res = re.findall(r"\[([^\]]*)\]", case)

            if include_height:
                calculated_heights = np.append(
                    calculated_heights, float(res[0].split(", ")[0])
                )

                widths_in_iteration = np.array(res[0].split(", ")[1:], dtype=float)

            else:
                widths_in_iteration = np.array(res[0].split(", "), dtype=float)

            no_widths = widths_in_iteration.size
            widths_in_iteration = np.reshape(widths_in_iteration, (1, no_widths))
            calculated_widths = np.append(
                calculated_widths, widths_in_iteration, axis=0
            )

    if include_height:
        return iteration_list, calculated_widths, calculated_heights
    else:
        return iteration_list, calculated_widths


def load_best_ccx_solutions(
    mesh: Mesh, best_it_location: str = "ccx_files"
) -> tuple[np.ndarray, np.ndarray]:

    """Loads best solutions from Indago optimization"""

    displacement = np.empty((mesh.node_array.shape[0], 2), dtype=float)
    stress = np.empty((mesh.node_array.shape[0], 6), dtype=float)

    calculated_displacement = np.empty((0, mesh.node_array.shape[0], 2), dtype=float)

    calculated_stress = np.empty((0, mesh.node_array.shape[0], 6), dtype=float)

    directory_iteration_array = np.empty(0, dtype=int)

    for directory in os.listdir(os.path.join(os.getcwd(), best_it_location)):
        if not directory.startswith("best_it"):
            continue

        directory_iteration_array = np.append(
            directory_iteration_array, int(directory.strip("best_it"))
        )

    directory_iteration_array = np.sort(directory_iteration_array)

    for iteration in directory_iteration_array:

        path_to_ccx_files = os.path.join(
            os.getcwd(), "ccx_files", f"best_it{iteration}"
        )

        ccx_files = os.listdir(path_to_ccx_files)

        for ccx_file in ccx_files:
            if ccx_file.endswith(".frd"):
                displacement, stress = read_ccx_results(
                    mesh,
                    os.path.join(
                        os.getcwd(), "ccx_files", f"best_it{iteration}", ccx_file
                    ),
                )

        calculated_displacement = np.append(
            calculated_displacement, np.reshape(displacement, (1, -1, 2)), axis=0
        )

        calculated_stress = np.append(
            calculated_stress, np.reshape(stress, (1, -1, 6)), axis=0
        )

    return calculated_displacement, calculated_stress
