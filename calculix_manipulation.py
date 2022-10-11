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
<<<<<<< HEAD
            displacement_array = np.append(
                displacement_array,
                np.reshape(np.array(displacement_per_node, dtype=np.float64), (1, 3)),
                axis=0,
            )
            stress_array = np.append(
                stress_array,
                np.reshape(np.array(stress_per_node, dtype=np.float64), (1, 6)),
                axis=0,
=======
            os.mkdir(os.path.join(os.getcwd(), 'ccx_files'))

    final_displacement_node_positions = np.empty(shape=(0),
                                                 dtype=int)

    force_node_positions = np.empty(shape=(0),
                                    dtype=int)

    boundary_node_positions = np.empty(shape=(0),
                                       dtype=int)

    def translate_mesh(self):

        output_string = ''

        beams_to_write = self.used_mesh.beam_array[
            [width > 0 for width in self.used_mesh.beam_width_array]
        ]

        nodes_to_write = np.empty((0), dtype=np.int32)
        for idx, beam in enumerate(self.used_mesh.beam_array):
            for beam_to_write in beams_to_write:
                if beam[0] == beam_to_write[0] and beam[1] == beam_to_write[1]:
                    nodes_to_write = np.append(
                        nodes_to_write,
                        self.used_mesh._fetch_beam_nodes(idx)
                    )

        nodes_to_write = np.unique(nodes_to_write)
        nodes_idx_to_write = nodes_to_write + 1
        self._no_of_used_nodes = np.size(nodes_to_write)

        # Node translator
        output_string += '*node, nset=nall\n'
        node_counter = 0
        for string in [
            f'{i}, {np.array2string(row, separator=",")[1:-1]}\n'
            for i, row in zip(nodes_idx_to_write,
                              self.used_mesh.node_array[nodes_to_write]
                              )
        ]:
            node_counter += 1
            for node, _, _ in self.used_mesh.final_displacement_array:
                node = int(node)
                if node == int(string.split(',')[0]) and \
                   node_counter not in self.final_displacement_node_positions and \
                   np.shape(self.used_mesh.final_displacement_array)[0] != self.final_displacement_node_positions.size:
                    self.final_displacement_node_positions = np.append(
                        self.final_displacement_node_positions,
                        node_counter
                    )

            for node in self.used_mesh.force_array[:, 0]:
                if node + 1 == int(string.split(',')[0]) and \
                   node_counter not in self.force_node_positions:
                    self.force_node_positions = np.append(
                        self.force_node_positions,
                        node_counter
                    )

            for node in self.used_mesh.boundary_array[:, 0]:
                if node + 1 == int(string.split(',')[0]) and \
                   node_counter not in self.boundary_node_positions:
                    self.boundary_node_positions = np.append(
                        self.boundary_node_positions,
                        node_counter
                    )

            output_string += string

        # Beam translator
        elset_name_list: list[str] = []
        written_beam_index = 0
        element_index = 0
        for idx, beam in enumerate(self.used_mesh.beam_array):
            for beam_to_write in beams_to_write:
                if beam[0] == beam_to_write[0] and beam[1] == beam_to_write[1]:
                    written_beam_index += 1
                    elset_name = f'b_{written_beam_index}'
                    elset_name_list.append(elset_name)

                    output_string += f'*element,type=b32,elset={elset_name}\n'

                    beam_elements = np.reshape(
                        self.used_mesh._fetch_beam_nodes(idx)[:-1],
                        (-1, 2)
                    )
                    beam_elements = np.append(
                        beam_elements,
                        np.reshape(
                            np.append(
                                beam_elements[1:, 0],
                                self.used_mesh._fetch_beam_nodes(idx)[-1]
                            ),
                            (-1, 1)
                        ),
                        axis=1
                    )
                    beam_elements += 1

                    for element in beam_elements:
                        element_index += 1
                        nodes_per_el_str = np.array2string(
                            element,
                            separator=','
                        )[1:-1]
                        output_string += f'{element_index},' +\
                            f' {nodes_per_el_str}\n'

        output_string += '*elset, elset=elall\n'
        for name in elset_name_list:
            output_string += f'{name},\n'

        # Materials writer
        output_string += '*material, name=mesh_material\n'
        output_string += '*elastic, type=iso\n'
        output_string += f'{self.used_mesh.material}'[1:-1] + '\n'

        # Beam width writer
        widths_to_write = self.used_mesh.beam_width_array[
            [width > 0 for width in self.used_mesh.beam_width_array]
        ]
        for element_set_name, width in zip(elset_name_list, widths_to_write):
            output_string += f'*beam section, elset={element_set_name},'
            output_string += 'material=mesh_material, section=rect\n'
            output_string += f'{self.used_mesh.beam_height}, {width:.20f}\n'
            output_string += '0.d0,0.d0,1.d0\n'

        # 2D case definition
        output_string += '*boundary\n'
        for node in nodes_idx_to_write:
            output_string += f'{node}, 3,5\n'

        # Boundary writer
        output_string += '*boundary\n'
        for boundary_definition in self.used_mesh.boundary_array:
            idx, x_ax, y_ax, z_rot = boundary_definition
            idx += 1
            if x_ax != 0:
                output_string += f'{idx}, {x_ax}\n'
            if y_ax != 0:
                output_string += f'{idx}, {y_ax*2}\n'
            if z_rot != 0:
                output_string += f'{idx}, {z_rot*6}\n'

        # Initial displacement writer
        if np.size(self.used_mesh.initial_displacement_array) != 0:
            output_string += '*initial conditions,type=displacement\n'
        for (idx, disp_x, disp_y) in self.used_mesh.initial_displacement_array:
            disp_x_sting = f'{int(idx + 1)}, 1, {disp_x}\n'
            disp_y_sting = f'{int(idx + 1)}, 2, {disp_y}\n'
            if disp_x != 0:
                output_string += disp_x_sting
            if disp_y != 0:
                output_string += disp_y_sting

        # Linear/nonlinear solver
        if self.nonlinear_calculation:
            output_string += '*step, nlgeom\n*static\n'
        else:
            output_string += '*step\n*static\n'

        # Force writer
        if np.size(self.used_mesh.force_array) != 0:
            output_string += '*cload\n'
        for (idx, force_x, force_y) in self.used_mesh.force_array:
            force_x_sting = f'{int(idx + 1)}, 1, {force_x}\n'
            force_y_sting = f'{int(idx + 1)}, 2, {force_y}\n'
            if force_x != 0:
                output_string += force_x_sting
            if force_y != 0:
                output_string += force_y_sting

        # File tail
        output_string += '*el print, elset=elall\ns\n'
        output_string += '*node file, output=2d, nset=nall\nu\n'
        output_string += '*el file, elset=elall\ns,noe\n'
        output_string += '*end step'

        return output_string

    def write_to_input_file(self,
                            ccx_case_name: str) -> str:

        if self.ccx_directory:
            solver_path = os.path.join(os.getcwd(),
                                       'ccx_files',
                                       ccx_case_name)
            os.mkdir(solver_path)

        else:
            solver_path = os.path.join(os.getcwd(),
                                       ccx_case_name)

        with open(os.path.join(solver_path, f'{ccx_case_name}.inp'),
                  'w') as input_file:
            input_file.writelines(self.translate_mesh())

        return solver_path

    def read_results(self,
                     ccx_case_path: str,
                     von_mises: bool = False):

        if os.path.split(ccx_case_path)[-1].endswith('.frd'):
            case_file = ccx_case_path

        else:
            case_file = os.path.join(
                ccx_case_path,
                f'{os.path.split(ccx_case_path)[-1]}.frd'
>>>>>>> de028e5286366b5f95380719159af10ebdfd1db4
            )

        if von_mises:
            sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz = [
                stress_array[-no_of_used_nodes:][i] for i in range(6)
            ]

            von_mises_eq_stress = calculate_von_mises_stress(
                Stress(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz)
            )

            return (displacement_array[:, :-1][-no_of_used_nodes:], von_mises_eq_stress)

        else:
            return (
                displacement_array[:, :-1][-no_of_used_nodes:],
                stress_array[-no_of_used_nodes:],
            )


<<<<<<< HEAD
def run_ccx(
    mesh: Mesh,
    ccx_case_name: str,
    delete_after_completion: bool = False,
    von_mises_instead_of_principal: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs ccx and fetches results using read_results()"""
    ccx_file_path = os.path.join(os.getcwd(), "ccx_files", ccx_case_name)

    # Bura ccx
    ccx_name = "ccx_2.19"
=======
        # Bura ccx
        # ccx_name = 'ccx_2.19'

        # Home ccx
        ccx_name = 'ccx'
>>>>>>> de028e5286366b5f95380719159af10ebdfd1db4

    # Home ccx
    # ccx_name = "ccx"

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
