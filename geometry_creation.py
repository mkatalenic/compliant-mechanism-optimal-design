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

    def __init__(mdf,
                 material: tuple,
                 max_finite_element_size: float):
        '''
        Mesh initialization
        '''

        mdf.material = material  # (E, Poasson)

        mdf.max_element_size = max_finite_element_size

    '''
    NODE DEFINITIONS
    '''

    node_array = np.empty((0, 2), dtype=np.float64)
    non_removable_nodes = np.empty(0, dtype=np.int32)
    main_nodes = np.empty(0, dtype=np.int32)

    def create_node(mdf,
                    coordinates: tuple,
                    main_node: bool = False,
                    removable: bool = True):

        current_node_index = int(np.shape(mdf.node_array)[0])

        mdf.node_array = np.append(
            mdf.node_array,
            np.reshape(coordinates, (1, 2)),
            axis=0
        )

        if not removable:
            mdf.non_removable_nodes = np.append(
                mdf.non_removable_nodes,
                current_node_index
            )

        if main_node:
            mdf.main_nodes = np.append(
                mdf.main_nodes,
                current_node_index
            )

    def _set_unremovable(mdf,
                         node):
        '''Sets node as non removable'''

        mdf.non_removable_nodes = np.append(
            mdf.non_removable_nodes,
            node
        )

    '''
    NODE FETCHING
    '''

    def _near_node_index(mdf,
                         node_def):
        '''Near node coordinates to node index'''

        if type(node_def) == tuple:
            for idx, node in enumerate(
                    np.isclose(
                        mdf.node_array,
                        node_def
                    )
            ):
                if np.alltrue(node):
                    return idx

        elif type(node_def) == int:
            return node_def

    def node_laso(mdf,
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
            used_array = mdf.node_array[mdf.main_node_array]
            contained_node_id = mdf.main_node_array
        else:
            used_array = mdf.node_array
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

    def create_beam(mdf,
                    first_node,
                    last_node):
        '''Creates a beam '''

        f_node = mdf._near_node_index(first_node)
        l_node = mdf._near_node_index(last_node)

        beam_size = np.sqrt(
            np.sum(
                (mdf.node_array[f_node] - mdf.node_array[l_node])**2
            )
        )

        no_nodes_per_beam = int(beam_size / mdf.max_element_size)

        mdf.beam_array = np.append(
            mdf.beam_array,
            [f_node, l_node, no_nodes_per_beam],
            axis=0
        )

        for new_node in np.linspace(
                mdf.node_array[first_node],
                mdf.node_array[last_node],
                no_nodes_per_beam + 1,
                False
        )[1:]:

            mdf.create_node(new_node)

            current_node_index = int(np.shape(mdf.node_array)[0])

            mdf.beam_mid_nodes_array = np.append(
                mdf.beam_mid_nodes_array,
                current_node_index
            )

    '''
         _   _       _ _
     ___| |_| |_ ___|_| |_ _ _ ___ ___ ___
    | .'|  _|  _|  _| | . | | |  _| -_|_ -|
    |__,|_| |_| |_| |_|___|___|_| |___|___|
    '''

    boundary_array = np.empty((0, 4),
                              dtype=np.int32)

    def create_boundary(mdf,
                        node,
                        bd_type: tuple[int],
                        set_unremovable: bool = False):
        '''
        Creates a boundary
        - 1 => x - translation
        - 2 => y - translation
        - 3 => z - rotation
        '''

        node_idx = mdf._near_node_index(node)

        mdf.boundary_array = np.append(
            mdf.boundary_array,
            np.reshape(
                np.append(node_idx, bd_type),
                (1, 4)
            ),
            axis=0
        )

        if set_unremovable & node_idx not in mdf.non_removable_nodes:
            mdf._set_unremovable(node_idx)

    force_array = np.empty((0, 3),
                           dtype=np.float64)

    def create_force(mdf,
                     node,
                     force_vector: tuple):
        '''Creates a force'''

        node_idx = mdf._near_node_index(node)

        mdf.force_array = np.append(
            mdf.force_array,
            np.reshape(
                np.append(node_idx, force_vector),
                (1, 3)
            ),
            axis=0
        )

        mdf._set_unremovable(node_idx)

    initial_displacement_array = np.empty((0, 3),
                                          dtype=np.float64)

    def create_initial_displacement(mdf,
                                    node,
                                    displacement_vector: tuple):
        "Creates initial displacement definition"

        node_idx = mdf._near_node_index(node)

        mdf.initial_displacement_array = np.append(
            mdf.initial_displacement_array,
            np.array(node_idx, displacement_vector[0], displacement_vector[1]),
            axis=0
        )

        mdf._set_unremovable(node_idx)

    final_displacement_array = np.empty((0, 3),
                                        dtype=np.float64)

    def set_final_displacement(mdf,
                               node,
                               displacement_vector: tuple):
        "Sets wanted displacement"

        node_idx = mdf._near_node_index(node)

        mdf.final_displacement_array = np.append(
            mdf.final_displacement_array,
            np.array(node_idx, displacement_vector[0], displacement_vector[1]),
            axis=0
        )

        mdf._set_unremovable(node_idx)
