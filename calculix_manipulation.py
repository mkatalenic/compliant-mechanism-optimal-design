#!/usr/bin/env python3

''' >==>    >===>>=>>==>
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

    def __init__(self,
                 used_mesh: gc.Mesh,
                 nonlin: bool = True,
                 create_calculix_directory: bool = True):

        self.used_mesh = used_mesh

        self.nonlinear_calculation = nonlin

        self.ccx_directory = create_calculix_directory

        if create_calculix_directory and not os.path.exists(
                os.path.join(
                    os.getcwd(),
                    'ccx_files'
                )
        ):
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
            output_string += f'{self.used_mesh.beam_height}, {width}\n'
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
            )

        with open(
                case_file,
                'r'
        ) as results_file:

            displacement_list = []
            stress_list = []

            used_nodes_id_list = []

            displacement_array = np.empty(shape=(0, 3),
                                          dtype=np.float64)
            stress_array = np.empty(shape=(0, 6),
                                    dtype=np.float64)

            displacement_section = False
            stress_section = False

            for line in results_file:
                if line.startswith('    1C'):
                    next

                if line[5:].startswith('DISP'):
                    displacement_section = True

                if line[5:].startswith('STRESS'):
                    stress_section = True

                if line.startswith(' -3'):
                    displacement_section = False
                    stress_section = False

                if displacement_section:
                    if not 'DISP' in line and not 'D1' in line and not 'D2' in line and not 'D3' in line and not 'ALL' in line:
                        node_id = int(line.strip()[2:12].strip()) - 1
                        used_nodes_id_list.append(node_id)
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

        if len(displacement_list) == 0 or len(stress_list) == 0:
            return False

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

        if von_mises:
            stress = stress_array[-self._no_of_used_nodes:]
            sig_x = stress[:, 0]
            sig_y = stress[:, 1]
            sig_z = stress[:, 2]
            tau_xy = stress[:, 3]
            tau_yz = stress[:, 4]
            tau_zx = stress[:, 5]
            von_mises_eq_stress = np.sqrt(
                (sig_x - sig_y)**2 +
                (sig_y - sig_z)**2 +
                (sig_z - sig_x)**2 +
                6 * (
                    tau_xy**2 +
                    tau_yz**2 +
                    tau_zx**2
                    )
            ) / np.sqrt(2)
            return (displacement_array[:, :-1][-self._no_of_used_nodes:],
                    von_mises_eq_stress), used_nodes_id_list

        else:
            return (displacement_array[:, :-1][-self._no_of_used_nodes:],
                    stress_array[-self._no_of_used_nodes:]), used_nodes_id_list

    def run_ccx(self,
                ccx_case_name: str,
                delete_after_completion: bool = False,
                von_mises_instead_of_principal: bool = True):

        # Bura ccx
        # ccx_name = 'ccx_2.19'

        # Home ccx
        ccx_name = 'ccx'

        ccx_file_path = self.write_to_input_file(ccx_case_name)

        with subprocess.Popen([ccx_name, ccx_case_name],
                              cwd=ccx_file_path,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            out, err = process.communicate()

        for line in str(out).replace('\\n', '\n').split('\n'):
            if line.startswith(' *ERROR') or len(err) != 0:
                return False, False  # U slučaju propale analize

        results, used_nodes_read = self.read_results(
            ccx_file_path,
            von_mises=von_mises_instead_of_principal
        )
        if results is False:
            return False, False

        if delete_after_completion:
            rmtree(ccx_file_path)

        return results, used_nodes_read  # U slučaju uspješne analize

    def load_from_info(self,
                       widths_size=None,
                       include_height=False):

        if widths_size is None:
            self.calculated_widths = np.empty(
                (0,
                 np.shape(
                     self.used_mesh.beam_width_array)[1]
                 )
                )
        else:
            self.calculated_widths = np.empty((0, widths_size),
                                              dtype=float)
        self.iteration_list = np.empty(0, dtype=float)

        if include_height:
            self.calculated_heights = np.empty(0, dtype=float)

        with open(
                os.path.join(
                    os.getcwd(),
                    'ccx_files',
                    'log.txt'
                ),
        ) as log_file:
            for case in log_file:
                self.iteration_list = np.append(
                    self.iteration_list,
                    int(case.split()[0]))
                res = re.findall(r"\[([^\]]*)\]", case)

                if include_height:
                    self.calculated_heights = np.append(
                        self.calculated_heights,
                        float(res[0].split(', ')[0])
                    )

                if include_height:
                    widths_in_iteration = np.array(res[0].split(', ')[1:],
                                                   dtype=float)
                else:
                    widths_in_iteration = np.array(res[0].split(', '),
                                                   dtype=float)

                no_widths = np.size(widths_in_iteration)
                widths_in_iteration = np.reshape(
                    widths_in_iteration, (1, no_widths)
                )
                self.calculated_widths = np.append(
                    self.calculated_widths,
                    widths_in_iteration,
                    axis=0
                )

    def load_best_ccx_solutions(
            self
    ):
        self.translate_mesh()
        self.calculated_displacement = np.empty((0, self.used_mesh.node_array.shape[0], 2),
                                                dtype=float)
        self.calculated_stress = np.empty((0, self.used_mesh.node_array.shape[0], 6),
                                          dtype=float)

        directory_iteration_array = np.empty(0, dtype=int)

        for directory in os.listdir(
                os.path.join(
                    os.getcwd(),
                    'ccx_files'
                )
        ):
            if not directory.startswith('best_it'):
                continue

            directory_iteration_array = np.append(
                directory_iteration_array,
                int(directory.strip('best_it'))
            )

        directory_iteration_array = np.sort(directory_iteration_array)

        for iteration in directory_iteration_array:

            path_to_ccx_files = os.path.join(
                os.getcwd(),
                'ccx_files',
                f'best_it{iteration}'
            )

            ccx_files = os.listdir(
                path_to_ccx_files
            )

            for ccx_file in ccx_files:
                if ccx_file.endswith('.frd'):
                    disp, stress = self.read_results(
                        os.path.join(
                            os.getcwd(),
                            'ccx_files',
                            f'best_it{iteration}',
                            ccx_file
                        )
                    )

            self.calculated_displacement = np.append(
                self.calculated_displacement,
                np.reshape(
                    disp,
                    (1, -1, 2)
                ),
                axis=0
            )

            self.calculated_stress = np.append(
                self.calculated_stress,
                np.reshape(
                    stress,
                    (1, -1, 6)
                ),
                axis=0
            )


def calculate_von_mises_stress(stress):

        sig_x = stress[:, 0]
        sig_y = stress[:, 1]
        sig_z = stress[:, 2]
        tau_xy = stress[:, 3]
        tau_yz = stress[:, 4]
        tau_zx = stress[:, 5]
        von_mises_eq_stress = np.sqrt(
            (sig_x - sig_y)**2 +
            (sig_y - sig_z)**2 +
            (sig_z - sig_x)**2 +
            6 * (
                tau_xy**2 +
                tau_yz**2 +
                tau_zx**2
                )
        ) / np.sqrt(2)
        return von_mises_eq_stress
