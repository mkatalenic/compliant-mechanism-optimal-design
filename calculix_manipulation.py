#!/usr/bin/env python3

'''
   >==>    >===>>=>>==>
 >=>       >=>  >>  >=>
>=>        >=>  >>  >=>
 >=>       >=>  >>  >=>
   >==>    >==>  >>  >=>
'''
import os
import re
import subprocess
from shutil import rmtree

import numpy as np

import geometry_creation as gc


def ccx_output_string_formatter(output_string: str):

    '''
    Formats native .frd data to useful data
    '''

    exponentials = re.split('E', output_string)[1:]
    output_numbers = re.split('E...', output_string)[:-1]

    output_numbers = [float(num) * 10**float(exp[:3])
                      for num, exp in zip(output_numbers, exponentials)]

    return output_numbers


class calculix_manipulator():

    def __init__(cm,
                 used_mesh: gc.Mesh,
                 nonlin: bool = True,
                 create_calculix_directory: bool = True):

        cm.used_mesh = used_mesh

        cm.nonlinear_calculation = nonlin

        cm.ccx_directory = create_calculix_directory

        if create_calculix_directory and not os.path.exists(
                os.path.join(
                    os.getcwd(),
                    'ccx_files'
                )
        ):
            os.mkdir(os.path.join(os.getcwd(), 'ccx_files'))

    def translate_mesh(cm):

        output_string = ''

        beams_to_write = cm.used_mesh.beam_array[
            [width > 0 for width in cm.used_mesh.beam_width_array]
        ]

        nodes_to_write = np.empty((0), dtype=np.int32)
        for idx, beam in enumerate(cm.used_mesh.beam_array):
            for beam_to_write in beams_to_write:
                if beam[0] == beam_to_write[0] and beam[1] == beam_to_write[1]:
                    nodes_to_write = np.append(
                        nodes_to_write,
                        cm.used_mesh._fetch_beam_nodes(idx)
                    )

        nodes_to_write = np.unique(nodes_to_write)
        nodes_idx_to_write = nodes_to_write + 1

        # Node translator
        output_string += '*node, nset=nall\n'
        for string in [
            f'{i}, {np.array2string(row, separator=",")[1:-1]}\n'
            for i, row in zip(nodes_idx_to_write,
                              cm.used_mesh.node_array[nodes_to_write]
                              )
        ]:
            output_string += string

        # Beam translator
        elset_name_list: list[str] = []
        written_beam_index = 0
        element_index = 0
        for idx, beam in enumerate(cm.used_mesh.beam_array):
            for beam_to_write in beams_to_write:
                if beam[0] == beam_to_write[0] and beam[1] == beam_to_write[1]:
                    written_beam_index += 1
                    elset_name = f'b_{written_beam_index}'
                    elset_name_list.append(elset_name)

                    output_string += f'*element,type=b32,elset={elset_name}\n'

                    beam_elements = np.reshape(
                        cm.used_mesh._fetch_beam_nodes(idx)[:-1],
                        (-1, 2)
                    )
                    beam_elements = np.append(
                        beam_elements,
                        np.reshape(
                            np.append(
                                beam_elements[1:, 0],
                                cm.used_mesh._fetch_beam_nodes(idx)[-1]
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
                        output_string += f'{element_index}, {nodes_per_el_str}\n'

        output_string += '*elset, elset=elall\n'
        for name in elset_name_list:
            output_string += f'{name},\n'

        # Materials writer
        output_string += '*material, name=mesh_material\n'
        output_string += '*elastic, type=iso\n'
        output_string += f'{cm.used_mesh.material}'[1:-1] + '\n'

        # Beam width writer
        widths_to_write = cm.used_mesh.beam_width_array[
            [width > 0 for width in cm.used_mesh.beam_width_array]
        ]
        for element_set_name, width in zip(elset_name_list, widths_to_write):
            output_string += f'*beam section, elset={element_set_name},'
            output_string += 'material=mesh_material, section=rect\n'
            output_string += f'{cm.used_mesh.beam_height}, {width}\n'
            output_string += '0.d0,0.d0,1.d0\n'

        # 2D case definition
        output_string += '*boundary\n'
        for node in nodes_idx_to_write:
            output_string += f'{node}, 3,5\n'

        # Boundary writer
        output_string += '*boundary\n'
        for boundary_definition in cm.used_mesh.boundary_array:
            idx, x_ax, y_ax, z_rot = boundary_definition
            idx += 1
            if x_ax != 0:
                output_string += f'{idx}, {x_ax}\n'
            if y_ax != 0:
                output_string += f'{idx}, {y_ax*2}\n'
            if z_rot != 0:
                output_string += f'{idx}, {z_rot*6}\n'

        # Linear/nonlinear solver
        if cm.nonlinear_calculation:
            output_string += '*step, nlgeom\n*static\n'
        else:
            output_string += '*step\n*static\n'

        # Initial displacement writer
        if np.size(cm.used_mesh.initial_displacement_array) != 0:
            output_string += '*boundary\n'
        for (idx, disp_x, disp_y) in cm.used_mesh.initial_displacement_array:
            disp_x_sting = f'{int(idx + 1)}, 1, {disp_x}\n'
            disp_y_sting = f'{int(idx + 1)}, 2, {disp_y}\n'
            if disp_x != 0:
                output_string += disp_x_sting
            if disp_y != 0:
                output_string += disp_y_sting

        # Force writer
        if np.size(cm.used_mesh.force_array) != 0:
            output_string += '*cload\n'
        for (idx, force_x, force_y) in cm.used_mesh.force_array:
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

    def write_to_input_file(cm,
                            ccx_case_name: str) -> str:

        if cm.ccx_directory:
            solver_path = os.path.join(os.getcwd(),
                                       'ccx_files',
                                       ccx_case_name)
            os.mkdir(solver_path)

        else:
            solver_path = os.path.join(os.getcwd(),
                                       ccx_case_name)

        with open(os.path.join(solver_path, f'{ccx_case_name}.inp'),
                  'w') as input_file:
            input_file.writelines(cm.translate_mesh())

        return solver_path

    def read_results(cm,
                     ccx_case_path: str):

        with open(
                os.path.join(
                    ccx_case_path,
                    f'{os.path.split(ccx_case_path)[-1]}.frd'
                ),
                'r'
        ) as results_file:

            displacement_list = []
            stress_list = []

            displacement_array = np.empty(shape=(0, 3),
                                          dtype=np.float64)
            stress_array = np.empty(shape=(0, 6),
                                    dtype=np.float64)

            displacement_section = False
            stress_section = False

            for line in results_file:

                if line[5:].startswith('DISP'):
                    displacement_section = True

                if line[5:].startswith('STRESS'):
                    stress_section = True

                if line.startswith(' -3'):
                    displacement_section = False
                    stress_section = False

                if displacement_section:
                    displacement_list.append(
                        ccx_output_string_formatter(
                            line.strip()[12:]
                        )
                    )

            if stress_section:
                stress_list.append(
                    ccx_output_string_formatter(
                        line.strip()[12:]
                    )
                )

        for node in displacement_list:
            if len(node) > 0:
                displacement_array = np.append(
                    displacement_array,
                    np.reshape(np.array(node), (1, 3)),
                    axis=0
                )

        for node in stress_list:
            if len(node) > 0:
                stress_array = np.append(
                    stress_array,
                    np.reshape(np.array(node), (1, 6)),
                    axis=0
                )

        return displacement_array[:, :-1], stress_array

    def run_ccx(cm,
                ccx_case_name: str,
                delete_after_completion: bool = False):

        # Bura ccx
        # ccx_name = 'ccx_2.19'

        # Home ccx
        ccx_name = 'ccx'

        ccx_file_path = cm.write_to_input_file(ccx_case_name)

        with subprocess.Popen([ccx_name, ccx_case_name],
                              cwd=ccx_file_path,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            out, err = process.communicate()

        for line in str(out).replace('\\n', '\n').split('\n'):
            if line.startswith(' *ERROR') or len(err) != 0:
                return False  # U slučaju propale analize

        results = cm.read_results(ccx_file_path)

        if delete_after_completion:
            rmtree(ccx_file_path)

        return results  # U slučaju uspješne analize
