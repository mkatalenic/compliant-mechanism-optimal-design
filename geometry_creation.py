#!/usr/bin/env python3

'''
MESH DESIGN FRAMEWORK

   >=>       >==>
 >=>  >=>  >=>
>=>   >=> >=>
 >=>  >=>  >=>
     >=>     >==>
  >=>
'''

import numpy as np
import os
import pickle


class Mesh():
    '''
    Main class containing:
        -> node definition
        -> beam definition
        -> width definition
        -> boundary definitions
        -> beginning state definitions
        -> end state definitions
        => simple automated mesh creator
    '''

    def __init__(self,
                 material: tuple,
                 max_finite_element_size: float):
        '''
        Mesh initialization
        '''

        self.material = material  # (E, Poasson)

        self.max_element_size = max_finite_element_size

    '''
    NODE DEFINITIONS
    '''

    node_array = np.empty((0, 2), dtype=np.float64)
    non_removable_nodes = np.empty(0, dtype=np.int32)
    main_nodes = np.empty(0, dtype=np.int32)

    def create_node(self,
                    coordinates: tuple,
                    main_node: bool = False,
                    removable: bool = True):

        current_node_index = int(np.shape(self.node_array)[0])

        self.node_array = np.append(
            self.node_array,
            np.reshape(coordinates, (1, 2)),
            axis=0
        )

        if not removable:
            self.non_removable_nodes = np.append(
                self.non_removable_nodes,
                current_node_index
            )

        if main_node:
            self.main_nodes = np.append(
                self.main_nodes,
                current_node_index
            )

    def _set_unremovable(self,
                         node):
        '''Sets node as non removable'''

        self.non_removable_nodes = np.append(
            self.non_removable_nodes,
            node
        )

    '''
    NODE FETCHING
    '''

    def _near_node_index(self,
                         node_def):
        '''Near node coordinates to node index'''

        if type(node_def) == tuple:
            used_array = self.node_array

            closest_node_index = np.argmin(
                np.sqrt(
                    np.sum(
                        np.square(
                            used_array
                            - np.repeat(np.array(node_def).reshape((1, 2)),
                                        np.shape(used_array)[0],
                                        axis=0)
                        ), axis=1
                    )
                ), axis=0
            )

            return closest_node_index

        else:
            return node_def

    def node_laso(self,
                  poly_points: list[tuple],
                  only_main_nodes: bool = True):
        '''
        Collects points that are inside of a given polygon

        The polygon is constructed of consecutive points
        with the last given point connecting to the first given point.

        This function uses the "ray-casting-method"
        to determine if a node is inside of the given polygon.
        '''

        polygon_array = np.append(np.array(poly_points),
                                  np.array(poly_points)[0].reshape(1, 2),
                                  axis=0)
        if only_main_nodes:
            used_array = self.node_array[self.main_nodes]
            contained_node_id = self.main_nodes
        else:
            used_array = self.node_array
            contained_node_id = np.arange(np.shape(used_array)[0])

        def boundary_intersection(point_of_interest: tuple,
                                  first_boundary_point: tuple,
                                  second_boundary_point: tuple) -> bool:
            '''
            Checks if the hor line starting at the given point
            intersects the boundary
            '''

            P = point_of_interest

            if first_boundary_point[1] == second_boundary_point[1]:
                A = first_boundary_point
                B = second_boundary_point
                if P[1] == A[1] and min(A[0], B[0]) < P[0] < max(A[0], B[0]):
                    return True
                else:
                    return False

            elif first_boundary_point[1] > second_boundary_point[1]:
                B = first_boundary_point
                A = second_boundary_point
            else:
                A = first_boundary_point
                B = second_boundary_point

            if P[1] == A[1] or P[1] == B[1]:
                P = (P[0], P[1] + 1e-8)

            if P[1] < A[1] or P[1] > B[1]:
                return False
            elif P[0] >= max(A[0], B[0]):
                return False
            else:
                if P[0] < min(A[0], B[0]):
                    return True
                else:
                    if A[0] != B[0]:
                        angle_XAB = (B[1] - A[1])/(B[0] - A[0])
                    else:
                        angle_XAB = 1e10
                    if A[0] != P[0]:
                        angle_XAP = (P[1] - A[1])/(P[0] - A[0])
                    else:
                        angle_XAP = 1e10

                    if angle_XAP >= angle_XAB:
                        return True
                    else:
                        return False

        counter_array = []
        for point in used_array:
            counter = 0
            for first, second in zip(polygon_array[:-1], polygon_array[1:]):
                counter += boundary_intersection(tuple(point),
                                                 tuple(first),
                                                 tuple(second))
            counter_array.append(counter)

        return contained_node_id[[result % 2 != 0 for result in counter_array]]

    beam_array = np.empty((0, 3),
                          dtype=np.int32)

    beam_mid_nodes_array = np.empty((0),
                                    dtype=np.int32)

    def create_beam(self,
                    first_node,
                    last_node):
        '''Creates a beam '''

        f_node = self._near_node_index(first_node)
        l_node = self._near_node_index(last_node)

        beam_size = np.sqrt(
            np.sum(
                (self.node_array[f_node] - self.node_array[l_node])**2
            )
        )

        no_nodes_per_beam = int(beam_size / self.max_element_size)

        self.beam_array = np.append(
            self.beam_array,
            [[f_node, l_node, no_nodes_per_beam*2 - 1]],
            axis=0
        )

        for new_node in np.linspace(
                self.node_array[first_node],
                self.node_array[last_node],
                no_nodes_per_beam*2,
                False
        )[1:]:

            self.create_node(new_node)

            current_node_index = int(np.shape(self.node_array)[0] - 1)

            self.beam_mid_nodes_array = np.append(
                self.beam_mid_nodes_array,
                current_node_index
            )

    def _fetch_beam_nodes(self,
                          beam_to_fetch) -> np.ndarray:
        '''
        Fetches beam midnodes and outputs them
        together with the first and last node
        '''

        out_nodes = self.beam_array[beam_to_fetch][:-1]

        for i, beam in enumerate(self.beam_array):
            if i == beam_to_fetch:

                mid_nodes = self.beam_mid_nodes_array[
                    np.sum(self.beam_array[:i, -1]):
                    np.sum(self.beam_array[:i, -1]) + beam[-1]
                ]

                out_nodes = np.insert(
                    out_nodes,
                    1,
                    mid_nodes
                )

        return out_nodes

    def beam_laso(self,
                  polygon_points: list[tuple]):
        '''
        Ouputs a list of beam ids
        '''

        cought_nodes = self.node_laso(poly_points=polygon_points,
                                      only_main_nodes=False)
        cought_beams = np.empty(shape=(0))

        for beam in range(np.shape(self.beam_array)[0]):
            nodes_in_beam = self._fetch_beam_nodes(beam)
            if any(np.in1d(nodes_in_beam, cought_nodes)):
                cought_beams = np.append(
                    cought_beams,
                    beam
                )

        return cought_beams
    '''
         _   _       _ _
     ___| |_| |_ ___|_| |_ _ _ ___ ___ ___
    | .'|  _|  _|  _| | . | | |  _| -_|_ -|
    |__,|_| |_| |_| |_|___|___|_| |___|___|
    '''

    boundary_array = np.empty((0, 4),
                              dtype=np.int32)

    def create_boundary(self,
                        node,
                        bd_type: tuple[int],
                        set_unremovable: bool = False):
        '''
        Creates a boundary
        - 1 => x - translation
        - 2 => y - translation
        - 3 => z - rotation
        '''

        node_idx = self._near_node_index(node)

        self.boundary_array = np.append(
            self.boundary_array,
            np.reshape(
                np.append(node_idx, bd_type),
                (1, 4)
            ),
            axis=0
        )

        if set_unremovable and node_idx not in self.non_removable_nodes:
            self._set_unremovable(node_idx)

    force_array = np.empty((0, 3),
                           dtype=np.float64)

    def create_force(self,
                     node,
                     force_vector: tuple):
        '''Creates a force'''

        node_idx = self._near_node_index(node)

        self.force_array = np.append(
            self.force_array,
            np.reshape(
                np.append(node_idx, force_vector),
                (1, 3)
            ),
            axis=0
        )

        self._set_unremovable(node_idx)

    initial_displacement_array = np.empty((0, 3),
                                          dtype=np.float64)

    def create_initial_displacement(self,
                                    node,
                                    displacement_vector: tuple):
        "Creates initial displacement definition"

        node_idx = self._near_node_index(node)

        self.initial_displacement_array = np.append(
            self.initial_displacement_array,
            np.array([[node_idx, displacement_vector[0],
                       displacement_vector[1]]]),
            axis=0
        )

        self._set_unremovable(node_idx)

    final_displacement_array = np.empty((0, 3),
                                        dtype=np.float64)

    def set_final_displacement(self,
                               node,
                               displacement_vector: tuple):
        "Sets wanted displacement"

        node_idx = self._near_node_index(node)

        self.final_displacement_array = np.append(
            self.final_displacement_array,
            np.array([[node_idx,
                       displacement_vector[0],
                       displacement_vector[1]]]),
            axis=0
        )

        self._set_unremovable(node_idx)

    '''
    WIDTH DEFINITION
    '''

    minimal_beam_width: float
    beam_height: float

    mechanism_volume: float

    beam_width_array = np.empty(shape=(0),
                                dtype=np.float64)

    def set_width_array(self,
                        input_width):
        '''
        Sets mesh beam widths
        '''
        remove_beams = self.beam_array[input_width < self.minimal_beam_width]

        remove_nodes = np.empty((0),
                                dtype=np.int32)

        for i, beam in enumerate(self.beam_array):
            if beam in remove_beams:
                remove_nodes = np.append(
                    remove_nodes,
                    beam[:-1]
                )
                mid_nodes = self.beam_mid_nodes_array[
                    np.sum(self.beam_array[:i, -1]):
                    np.sum(self.beam_array[:i, -1]) + beam[-1]
                ]
                remove_nodes = np.append(
                    remove_nodes,
                    mid_nodes
                )

        remove_nodes = np.unique(remove_nodes)

        if np.size(
                np.intersect1d(
                    remove_nodes,
                    self.non_removable_nodes
                )
        ) != 0:
            return False  # Width assign terminated unsucessfully

        self.beam_width_array = input_width
        self.beam_width_array[input_width < self.minimal_beam_width] = 0

        return True  # Width assign terminated successfully

    def calculate_mechanism_volume(self):

        length_array = np.empty((0), dtype=np.float64)

        for beam in self.beam_array:
            dx, dy = (self.node_array[beam[0]] -
                      self.node_array[beam[1]])
            length = np.sqrt(dx**2 + dy**2)
            length_array = np.append(length_array, length)
        mechanism_area = np.sum(length_array * self.beam_width_array)
        self.mechanism_volume = mechanism_area * self.beam_height

        return self.mechanism_volume

    def write_beginning_state(self):

        '''Writes beginning state of the construction'''

        with open(os.path.join(
                os.getcwd(),
                'mesh_setup.pkl'
        ),
                  'wb') as case_setup:
            pickle.dump(self, case_setup, pickle.HIGHEST_PROTOCOL)


def read_mesh_state():
    with open(os.path.join(
            os.getcwd(),
            'mesh_setup.pkl'
    ),
                'rb') as case_setup:
        return pickle.load(case_setup)


class SimpleMeshCreator(Mesh):

    '''
    A simple, automated mesh creaton based on given:
    - x dimension
    - y dimension
    - number of divisions (x_div, y_div)
    - support definitions
    '''

    def __init__(self,
                 _material,
                 _max_element_size,
                 length: float,
                 heigth: float,
                 divisions: tuple,
                 support_definition: str = None):

        super().__init__(_material, _max_element_size)

        '''
        Initialization
        '''

        for vert_coord in np.linspace(0,
                                      heigth,
                                      divisions[1] + 1,
                                      endpoint=True):
            for hor_coord in np.linspace(0,
                                         length,
                                         divisions[0] + 1,
                                         endpoint=True):
                self.create_node((hor_coord, vert_coord),
                                 main_node=True)

        for y_node in range(divisions[1] + 1):
            for x_node in range(divisions[0] + 1):
                current_node_id = x_node + y_node*(divisions[0] + 1)

                if x_node < divisions[0]:
                    self.create_beam(current_node_id,
                                     current_node_id + 1)
                if y_node < divisions[1]:
                    self.create_beam(current_node_id,
                                     current_node_id + (divisions[0] + 1))

                if support_definition == 'fd' \
                   and y_node < divisions[1] \
                   and x_node < divisions[0]:
                    self.create_beam(current_node_id,
                                     current_node_id + 1 + (divisions[0] + 1))

                if support_definition == 'bd' \
                   and y_node < divisions[1] \
                   and x_node > 0:
                    self.create_beam(current_node_id,
                                     current_node_id - 1 + (divisions[0] + 1))

                if support_definition == 'x' \
                   and y_node < divisions[1] \
                   and x_node < divisions[0]:
                    self.create_node(
                        np.average(
                            self.node_array[
                                [current_node_id,
                                 current_node_id + 1 + (divisions[0] + 1)], :],
                            axis=0
                        ),
                        main_node=True
                    )

                    created_mid_node_index = np.shape(self.node_array)[0] - 1

                    self.create_beam(current_node_id,
                                     created_mid_node_index)
                    self.create_beam(created_mid_node_index,
                                     current_node_id + 1 + (divisions[0] + 1))
                    self.create_beam(current_node_id + (divisions[0] + 1),
                                     created_mid_node_index)
                    self.create_beam(created_mid_node_index,
                                     current_node_id + 1)
