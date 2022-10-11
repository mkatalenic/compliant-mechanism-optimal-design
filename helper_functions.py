#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil

from io import TextIOWrapper
import re
import numpy as np

from geometry_creation import Mesh
from my_typing import Stress, Coordinates


def calculate_von_mises_stress(stress: Stress) -> np.ndarray:
    """Calculates VonMises stress from the given stress matrix"""

    von_mises_eq_stress = np.sqrt(
        (stress.sigma_x - stress.sigma_y) ** 2
        + (stress.sigma_y - stress.sigma_z) ** 2
        + (stress.sigma_z - stress.sigma_x) ** 2
        + 6 * (stress.tau_xy**2 + stress.tau_yz**2 + stress.tau_xz**2)
    ) / np.sqrt(2)

    return von_mises_eq_stress


def ccx_output_formatter(ccx_output: str) -> list[float]:

    """
    Formats native .frd data to useful data
    """

    exponentials = re.split("E", ccx_output)[1:]
    output_numbers = re.split("E...", ccx_output)[:-1]

    output_numbers = [
        float(num) * 10 ** float(exp[:3])
        for num, exp in zip(output_numbers, exponentials)
    ]

    return output_numbers


def used_nodes_in_current_mesh_state(mesh: Mesh) -> np.ndarray:
    """Returns an array of used nodes in the currnet mesh state"""
    used_beams_idx = np.where(mesh.beam_width_array > 0)[0]
    used_nodes_idx = np.empty((0), dtype=np.int32)
    for beam_idx in used_beams_idx:
        used_nodes_idx = np.append(used_nodes_idx, mesh._fetch_beam_nodes(beam_idx))
    used_nodes_idx = np.unique(used_nodes_idx)
    return used_nodes_idx


def translate_node_coordinates(mesh: Mesh) -> str:
    """Translates mesh node coordinates to ccx input string"""

    used_nodes = used_nodes_in_current_mesh_state(mesh)
    used_nodes_idx_to_write = used_nodes + 1
    output_ccx_nodes_string = "*node, nset=nall\n" + "".join(
        [
            f'{i}, {np.array2string(row, separator=",")[1:-1]}\n'
            for i, row in zip(used_nodes_idx_to_write, mesh.node_array[used_nodes])
        ]
    )

    return output_ccx_nodes_string


def initial_displacement_node_positions_in_ccx_results(mesh: Mesh) -> np.ndarray:
    """Fetches defined initial nodes position from ccx input"""

    initial_displacement_node_positions_in_ccx_results = np.empty((0), dtype=int)

    for node_counter, used_node in enumerate(used_nodes_in_current_mesh_state(mesh)):

        for initial_displacement_node in mesh.final_displacement_array[:, 0]:
            initial_displacement_node = int(initial_displacement_node)
            if (
                initial_displacement_node == used_node
                and node_counter
                not in initial_displacement_node_positions_in_ccx_results
                and mesh.initial_displacement_array.shape[0]
                != initial_displacement_node_positions_in_ccx_results.size
            ):
                initial_displacement_node_positions_in_ccx_results = np.append(
                    initial_displacement_node_positions_in_ccx_results, node_counter
                )
    return initial_displacement_node_positions_in_ccx_results


def final_displacement_node_positions_in_ccx_results(mesh: Mesh) -> np.ndarray:
    """Fetches defined final displacement node positions from ccx input"""

    final_displacement_node_positions_in_ccx_results = np.empty((0), dtype=int)

    for node_counter, used_node in enumerate(used_nodes_in_current_mesh_state(mesh)):

        for final_displacement_node in mesh.final_displacement_array[:, 0]:
            final_displacement_node = int(final_displacement_node)
            if (
                final_displacement_node == used_node
                and node_counter not in final_displacement_node_positions_in_ccx_results
                and mesh.final_displacement_array.shape[0]
                != final_displacement_node_positions_in_ccx_results.size
            ):
                final_displacement_node_positions_in_ccx_results = np.append(
                    final_displacement_node_positions_in_ccx_results, node_counter
                )
    return final_displacement_node_positions_in_ccx_results


def force_node_positions_in_ccx_results(mesh: Mesh) -> np.ndarray:
    """Fetches defined force node positions from ccx input"""

    force_node_positions_in_ccx_results = np.empty((0), dtype=int)

    for node_counter, used_node in enumerate(used_nodes_in_current_mesh_state(mesh)):

        for force_node in mesh.force_array[:, 0]:
            force_node = int(force_node)
            if (
                force_node == used_node
                and node_counter not in force_node_positions_in_ccx_results
                and mesh.force_array.shape[0]
                != force_node_positions_in_ccx_results.size
            ):
                force_node_positions_in_ccx_results = np.append(
                    force_node_positions_in_ccx_results, node_counter
                )
    return force_node_positions_in_ccx_results


def boundary_nodes_positions_in_ccx_results(mesh: Mesh) -> np.ndarray:
    """Fetches defined boundary node positions from ccx input"""

    boundary_node_positions_in_ccx_results = np.empty((0), dtype=int)

    for node_counter, used_node in enumerate(used_nodes_in_current_mesh_state(mesh)):

        for boundary_node in mesh.boundary_array[:, 0]:
            boundary_node = int(boundary_node)
            if (
                boundary_node == used_node
                and node_counter not in boundary_node_positions_in_ccx_results
            ):
                boundary_node_positions_in_ccx_results = np.append(
                    boundary_node_positions_in_ccx_results, node_counter
                )

    return boundary_node_positions_in_ccx_results


def convert_beam_nodes_for_ccx_b32_elements(mesh: Mesh, beam_index: int) -> np.ndarray:
    """Adds and formats beams to b32 beam ccx format"""

    beam_elements = np.reshape(mesh._fetch_beam_nodes(beam_index)[:-1], (-1, 2))

    b32_beam = np.append(
        beam_elements,
        np.reshape(
            np.append(beam_elements[1:, 0], mesh._fetch_beam_nodes(beam_index)[-1]),
            (-1, 1),
        ),
        axis=1,
    )

    return b32_beam


def translate_beam_definitions(mesh: Mesh) -> str:
    """Translates defined beams to ccx input string"""

    used_beams_idx = np.where(mesh.beam_width_array > 0)[0]
    elset_names = [f"b_{beam_index}" for beam_index in used_beams_idx]
    elset_headers = [
        f"*element,type=b32,elset={elset_name}\n" for elset_name in elset_names
    ]
    ccx_beam_element_representation_list = []
    element_idx = 0
    full_element_string = ""

    for beam_idx in used_beams_idx:
        full_element_string = ""
        for element in convert_beam_nodes_for_ccx_b32_elements(mesh, int(beam_idx)):
            element_idx += 1
            nodes_per_el_str = np.array2string(element + 1, separator=",")[1:-1]
            full_element_string += f"{element_idx}, {nodes_per_el_str}\n"

        ccx_beam_element_representation_list.append(full_element_string)
    beam_definitions_string = ""
    for header, elements_per_beam in zip(
        elset_headers, ccx_beam_element_representation_list
    ):
        beam_definitions_string += header + elements_per_beam

    beam_definitions_string += "*elset, elset=elall\n"
    beam_definitions_string += "".join(
        [f"{elset_name},\n" for elset_name in elset_names]
    )
    return beam_definitions_string


def translate_material(mesh: Mesh) -> str:
    """Translates material propperties to ccx input string"""
    return "".join(
        [
            "*material, name=mesh_material\n",
            "*elastic, type=iso\n",
            f"{mesh.material.E}, {mesh.material.Poisson}\n",
        ]
    )


def translate_beam_cross_section(mesh: Mesh) -> str:
    """Treanslate beam cross sections to ccx input string"""
    return "".join(
        [
            f"*beam section, elset=b_{beam_index}, "
            + "material=mesh_material, section=rect\n"
            + f"{mesh.beam_height}, "
            + f"{mesh.beam_width_array[beam_index]:.20f}\n"
            + "0.d0,0.d0,1.d0\n"
            for beam_index in np.where(mesh.beam_width_array > 0)[0]
        ]
    )


def set_2D_calculix_case(mesh: Mesh) -> str:
    """Sets 2D boundary conditions for mesh nodes"""
    header = "*boundary\n"
    return header + "".join(
        [f"{node}, 3,5\n" for node in used_nodes_in_current_mesh_state(mesh) + 1]
    )


def translate_mesh_boundaries(mesh: Mesh) -> str:
    """Translates boundaries to ccx input string"""

    header = "*boundary\n"
    boundary_string = ""
    for boundary_definition in mesh.boundary_array:
        idx, x_ax, y_ax, z_rot = boundary_definition
        idx += 1
        if x_ax != 0:
            boundary_string += f"{idx}, {x_ax}\n"
        if y_ax != 0:
            boundary_string += f"{idx}, {y_ax*2}\n"
        if z_rot != 0:
            boundary_string += f"{idx}, {z_rot*6}\n"
    return header + boundary_string


def translate_initial_displacement(mesh: Mesh) -> str:
    """Translates initial displacements to ccx input string"""

    if mesh.initial_displacement_array.size == 0:
        return ""
    header = "*initial conditions, type=displacement\n"
    initial_displacement_string = ""
    for (idx, disp_x, disp_y) in mesh.initial_displacement_array:
        disp_x_sting = f"{int(idx + 1)}, 1, {disp_x}\n"
        disp_y_sting = f"{int(idx + 1)}, 2, {disp_y}\n"
        if disp_x != 0:
            initial_displacement_string += disp_x_sting
        if disp_y != 0:
            initial_displacement_string += disp_y_sting
    return header + initial_displacement_string


def use_nonlinear_solver(use_nonlinear: bool) -> str:
    """Sets up linear/nonlinear calculation"""

    if use_nonlinear:
        return "*step, nlgeom\n*static\n"
    else:
        return "*step\n*static\n"


def translate_mesh_forces(mesh: Mesh) -> str:
    """Translates applied forces to ccx input string"""

    if mesh.force_array.size == 0:
        return ""
    header = "*cload\n"
    force_string = ""
    for (idx, force_x, force_y) in mesh.force_array:
        force_x_sting = f"{int(idx + 1)}, 1, {force_x}\n"
        force_y_sting = f"{int(idx + 1)}, 2, {force_y}\n"
        if force_x != 0:
            force_string += force_x_sting
        if force_y != 0:
            force_string += force_y_sting
    return header + force_string


def calculix_input_file_tail() -> str:
    """Just sets the needed tail of the ccx input"""

    return "".join(
        [
            "*el print, elset=elall\ns\n",
            "*node file, output=2d, nset=nall\nu\n",
            "*el file, elset=elall\ns,noe\n",
            "*end step",
        ]
    )


def ccx_results_to_disp_stress_converter(
    results_file: TextIOWrapper,
) -> tuple:
    """Translate results from ccx .frd to displacement and stress arrays"""

    displacement_list = []
    stress_list = []

    used_nodes_id_list = []

    displacement_section = False
    stress_section = False

    for line in results_file:
        if line.startswith("    1C"):
            continue

        if line[5:].startswith("DISP"):
            displacement_section = True

        if line[5:].startswith("STRESS"):
            stress_section = True

        if line.startswith(" -3"):
            displacement_section = False
            stress_section = False

        if displacement_section:
            if (
                not "DISP" in line
                and not "D1" in line
                and not "D2" in line
                and not "D3" in line
                and not "ALL" in line
            ):
                node_id = int(line.strip()[2:12].strip()) - 1
                used_nodes_id_list.append(node_id)
            if len(ccx_output_formatter(line.strip()[12:])) != 0:
                displacement_list.append(ccx_output_formatter(line.strip()[12:]))

        if stress_section:
            if len(ccx_output_formatter(line.strip()[12:])) != 0:
                stress_list.append(ccx_output_formatter(line.strip()[12:]))

    return displacement_list, stress_list
