#!/usr/bin/env python3

"""
MESH DESIGN FRAMEWORK

   >=>       >==>
 >=>  >=>  >=>
>=>   >=> >=>
 >=>  >=>  >=>
     >=>     >==>
  >=>
"""

from __future__ import annotations

import numpy as np
import os
import pickle


def point_boundary_intersection(
    point_of_interest: Coordinates,
    first_boundary_point: Coordinates,
    second_boundary_point: Coordinates,
) -> bool:
    """
    Checks if the horizontal line starting at the given point
    intersects the boundary
    """

    P = point_of_interest

    if first_boundary_point.y == second_boundary_point.y:
        A = first_boundary_point
        B = second_boundary_point
        if P.y == A.y and min(A.x, B.x) < P.x < max(A.x, B.x):
            return True
        else:
            return False

    elif first_boundary_point.y > second_boundary_point.y:
        B = first_boundary_point
        A = second_boundary_point
    else:
        A = first_boundary_point
        B = second_boundary_point

    if P.y == A.y or P.y == B.y:
        P = Coordinates(P.x, P.y + 1e-8)

    if P.y < A.y or P.y > B.y:
        return False
    elif P.x >= max(A.x, B.x):
        return False
    else:
        if P.x < min(A.x, B.x):
            return True
        else:
            if A.x != B.x:
                angle_XAB = (B.y - A.y) / (B.x - A.x)
            else:
                angle_XAB = 1e10
            if A.x != P.x:
                angle_XAP = (P.y - A.y) / (P.x - A.x)
            else:
                angle_XAP = 1e10

            if angle_XAP >= angle_XAB:
                return True
            else:
                return False


from my_typing import (
    Material,
    Coordinates,
    Boundary,
    Force,
    NodeDisplacement,
    MeshDimensions,
    MeshSupportDefinitions,
)


class Mesh:
    """Mesh object creation framework"""

    def __init__(self, material: Material, max_finite_element_size: float):
        """Initialize all of the used mesh attributes"""

        # (E, Poisson)
        self.material = material

        self.minimal_beam_width: float
        self.beam_height: float

        self.mechanism_volume: float

        self.max_element_size = max_finite_element_size

        # Node array init
        self.node_array = np.empty((0, 2), dtype=np.float64)
        self.non_removable_nodes = np.empty(0, dtype=np.int32)
        self.main_nodes = np.empty(0, dtype=np.int32)

        # Boundary init
        self.boundary_array = np.empty((0, 4), dtype=np.int32)

        # Beam array init
        self.beam_array = np.empty((0, 3), dtype=np.int32)
        self.beam_mid_nodes_array = np.empty((0), dtype=np.int32)

        self.beam_width_array = np.empty(shape=(0), dtype=np.float64)
        self.beam_width_beginning_map = np.empty(shape=(0), dtype=bool)

        # Start and end conditions init
        self.initial_displacement_array = np.empty((0, 3), dtype=np.float64)

        self.force_array = np.empty((0, 3), dtype=np.float64)

        self.final_displacement_array = np.empty((0, 3), dtype=np.float64)

    def _set_unremovable(self, node):
        """Sets node as non removable"""

        self.non_removable_nodes = np.append(self.non_removable_nodes, node)
        self.non_removable_nodes = np.unique(self.non_removable_nodes)

    def create_node(
        self, coordinates: Coordinates, main_node: bool = False, removable: bool = True
    ):
        """
        Creates a node.
        set if removable or main
        """

        current_node_index = int(np.shape(self.node_array)[0])

        self.node_array = np.append(
            self.node_array,
            np.array([[coordinates.x, coordinates.y]]),
            axis=0,
        )

        if not removable:
            self._set_unremovable(current_node_index)

        if main_node:
            self.main_nodes = np.append(self.main_nodes, current_node_index)

    def _near_node_index(self, node_def):
        """Near node coordinates to node index"""

        if type(node_def) == tuple:
            used_array = self.node_array

            closest_node_index = np.argmin(
                np.sqrt(
                    np.sum(
                        np.square(
                            used_array
                            - np.repeat(
                                np.array(node_def).reshape((1, 2)),
                                np.shape(used_array)[0],
                                axis=0,
                            )
                        ),
                        axis=1,
                    )
                ),
                axis=0,
            )

            return closest_node_index

        else:
            return node_def

    def node_laso(self, poly_points: list, only_main_nodes: bool = True) -> np.ndarray:
        """
        Collects points that are inside of a given polygon

        The polygon is constructed of consecutive points
        with the last given point connecting to the first given point.

        This function uses the "ray-casting-method"
        to determine if a node is inside of the given polygon.
        """

        polygon_array = np.append(
            np.array(poly_points), np.array(poly_points)[0].reshape(1, 2), axis=0
        )
        if only_main_nodes:
            used_array = self.node_array[self.main_nodes]
            contained_node_id = self.main_nodes
        else:
            used_array = self.node_array
            contained_node_id = np.arange(np.shape(used_array)[0])

        counter_array = []
        for point in used_array:
            counter = 0
            for first, second in zip(polygon_array[:-1], polygon_array[1:]):
                counter += point_boundary_intersection(
                    Coordinates(point[0], point[1]),
                    Coordinates(first[0], first[1]),
                    Coordinates(second[0], second[1]),
                )
            counter_array.append(counter)

        return contained_node_id[[result % 2 != 0 for result in counter_array]]

    def create_beam(self, first_node, last_node):
        """Creates a beam"""

        f_node = self._near_node_index(first_node)
        l_node = self._near_node_index(last_node)

        beam_size = np.sqrt(
            np.sum((self.node_array[f_node] - self.node_array[l_node]) ** 2)
        )

        no_nodes_per_beam = int(beam_size / self.max_element_size)

        self.beam_array = np.append(
            self.beam_array,
            np.array([[f_node, l_node, no_nodes_per_beam * 2 - 1]]),
            axis=0,
        )

        for new_node in np.linspace(
            self.node_array[first_node],
            self.node_array[last_node],
            no_nodes_per_beam * 2,
            False,
        )[1:]:

            self.create_node(Coordinates(new_node[0], new_node[1]))

            current_node_index = int(np.shape(self.node_array)[0] - 1)

            self.beam_mid_nodes_array = np.append(
                self.beam_mid_nodes_array, current_node_index
            )

    def _fetch_beam_nodes(self, beam_to_fetch: int) -> np.ndarray:
        """
        Fetches beam midnodes and outputs them
        together with the first and last node
        """

        out_nodes = self.beam_array[beam_to_fetch][:-1]
        mid_nodes_until_current = np.sum(self.beam_array[:beam_to_fetch, -1])

        beam_mid_nodes = self.beam_mid_nodes_array[
            mid_nodes_until_current : mid_nodes_until_current
            + self.beam_array[beam_to_fetch, -1]
        ]

        return np.insert(out_nodes, 1, beam_mid_nodes)

    def beam_laso(self, polygon_points: list, only_main_nodes: bool=False) -> np.ndarray:
        """
        Ouputs a list of beam ids
        """

        cought_nodes = self.node_laso(poly_points=polygon_points, only_main_nodes=only_main_nodes)
        cought_beams = np.empty(shape=(0), dtype=int)

        if only_main_nodes:
            for beam in range(self.beam_array.shape[0]):
                nodes_in_beam = self._fetch_beam_nodes(beam)
                if all(np.in1d(nodes_in_beam[[0, -1]], cought_nodes)):
                    cought_beams = np.append(cought_beams, beam)
        else:
            for beam in range(self.beam_array.shape[0]):
                nodes_in_beam = self._fetch_beam_nodes(beam)
                if any(np.in1d(nodes_in_beam, cought_nodes)):
                    cought_beams = np.append(cought_beams, beam)

        return cought_beams

    def create_boundary(self, node, bd_type: Boundary, set_unremovable: bool = False):
        """
        Creates a boundary
        x_trans = (1, 0 ,0)
        y_trans = (0, 1 ,0)
        z_rotat = (0, 0 ,1)
        """

        node_idx = self._near_node_index(node)

        self.boundary_array = np.append(
            self.boundary_array,
            np.array(
                [
                    [
                        node_idx,
                        bd_type.x_translation,
                        bd_type.y_translation,
                        bd_type.z_rotation,
                    ]
                ]
            ),
            axis=0,
        )

        if set_unremovable and node_idx not in self.non_removable_nodes:
            self._set_unremovable(node_idx)

    def create_force(self, node, force_vector: Force):
        """Creates a force"""

        node_idx = self._near_node_index(node)

        self.force_array = np.append(
            self.force_array,
            np.array([[node_idx, force_vector.x_component, force_vector.y_component]]),
            axis=0,
        )

        self._set_unremovable(node_idx)

    def create_initial_displacement(self, node, displacement_vector: NodeDisplacement):
        "Creates initial displacement definition"

        node_idx = self._near_node_index(node)

        self.initial_displacement_array = np.append(
            self.initial_displacement_array,
            np.array(
                [
                    [
                        node_idx,
                        displacement_vector.x_component,
                        displacement_vector.y_component,
                    ]
                ]
            ),
            axis=0,
        )

        self._set_unremovable(node_idx)

    def set_final_displacement(self, node, displacement_vector: NodeDisplacement):
        "Sets wanted displacement"

        node_idx = self._near_node_index(node)

        self.final_displacement_array = np.append(
            self.final_displacement_array,
            np.array(
                [
                    [
                        node_idx,
                        displacement_vector.x_component,
                        displacement_vector.y_component,
                    ]
                ]
            ),
            axis=0,
        )

        self._set_unremovable(node_idx)

    def calculate_mechanism_volume(self) -> float:
        """Calculates mechanism volume of used mesh beams"""

        length_array = np.empty((0), dtype=np.float64)

        for beam in self.beam_array:
            dx, dy = self.node_array[beam[0]] - self.node_array[beam[1]]
            length = np.sqrt(dx**2 + dy**2)
            length_array = np.append(length_array, length)
        mechanism_area = np.sum(length_array * self.beam_width_array)
        self.mechanism_volume = float(mechanism_area * self.beam_height)

        return self.mechanism_volume

    def remove_beams(self, polygon_points: list) -> None:
        removed_beams = self.beam_laso(polygon_points)
        if len(self.beam_width_beginning_map) == 0:
            self.beam_width_beginning_map = np.full(self.beam_array.shape[0], True)
        self.beam_width_beginning_map[removed_beams] = False

    def set_widths_array_with_interface(
            self,
            input_width: float | np.ndarray,
            interface_polygon_points: list,
            interface_width: float
    ):
        interface_beams = self.beam_laso(interface_polygon_points)
        self.beam_width_beginning_map[interface_beams] = False
        self.set_width_array(input_width)

        self.beam_width_beginning_map[interface_beams] = True

        self.beam_width_array[interface_beams] = interface_width

        self.calculate_mechanism_volume()

        return True  # Width assign terminated successfully

    def set_width_array(self, input_width: float | np.ndarray):
        """
        Sets mesh beam widths
        """

        if len(self.beam_width_beginning_map) == 0:
            self.beam_width_beginning_map = np.full(self.beam_array.shape[0], True)

        if self.beam_width_array.size == 0:
            self.beam_width_array = np.zeros_like(
                self.beam_width_beginning_map, dtype=float
            )

        out_width = np.empty((0), dtype=float)

        if type(input_width) == float:
            input_width_size = len(self.beam_width_beginning_map)
            out_width = np.full(input_width_size, input_width)
            out_width[[not state for state in self.beam_width_beginning_map]] = 0.0
        elif type(input_width) == np.ndarray:
            out_width = np.zeros_like(self.beam_width_beginning_map, dtype=float)
            out_width[self.beam_width_beginning_map] = input_width

        remove_beams_idx = np.arange(self.beam_width_array.size)[
            out_width < self.minimal_beam_width
        ]
        non_zero_beams_idx = np.arange(self.beam_width_array.size)[
            out_width >= self.minimal_beam_width
        ]

        remove_nodes = np.empty((0), dtype=np.int32)
        left_nodes = np.empty((0), dtype=np.int32)

        for beam in remove_beams_idx:
            remove_nodes = np.append(remove_nodes, self._fetch_beam_nodes(beam))
        for beam in non_zero_beams_idx:
            left_nodes = np.append(left_nodes, self._fetch_beam_nodes(beam))
        remove_nodes = np.unique(remove_nodes)
        left_nodes = np.unique(left_nodes)

        if (
            np.size(np.intersect1d(left_nodes, self.non_removable_nodes))
            != np.unique(self.non_removable_nodes).size
        ):
            return False  # Width assign terminated unsucessfully

        out_width[out_width < self.minimal_beam_width] = 0.0
        self.beam_width_array = out_width

        self.calculate_mechanism_volume()

        return True  # Width assign terminated successfully

    def write_beginning_state(self):

        """Writes beginning state of the construction"""

        with open(os.path.join(os.getcwd(), "mesh_setup.pkl"), "wb") as case_setup:
            pickle.dump(self, case_setup, pickle.HIGHEST_PROTOCOL)


def read_mesh_state(pickled_mesh_location: str | None) -> Mesh:
    """Reads the beginning mesh state from a .pkl file"""
    if pickled_mesh_location is None:
        with open(os.path.join(os.getcwd(), "mesh_setup.pkl"), "rb") as case_setup:
            return pickle.load(case_setup)
    else:
        with open(os.path.join(os.getcwd(), pickled_mesh_location), "rb") as case_setup:
            return pickle.load(case_setup)


def simple_mesh_creator(
    material: Material,
    max_el_size: float,
    dimensions: MeshDimensions,
    support_definition: MeshSupportDefinitions = None,
) -> Mesh:
    """Returns a rectangular mesh with optional diagonal supports"""

    mesh = Mesh(material, max_el_size)

    for vert_coord in np.linspace(
        0, dimensions.heigth, dimensions.heigth_divisions + 1, endpoint=True
    ):
        for hor_coord in np.linspace(
            0, dimensions.length, dimensions.length_divisions + 1, endpoint=True
        ):
            mesh.create_node(Coordinates(hor_coord, vert_coord), main_node=True)

    for y_node in range(dimensions.heigth_divisions + 1):
        for x_node in range(dimensions.length_divisions + 1):
            current_node_id = x_node + y_node * (dimensions.length_divisions + 1)

            if x_node < dimensions.length_divisions:
                mesh.create_beam(current_node_id, current_node_id + 1)
            if y_node < dimensions.heigth_divisions:
                mesh.create_beam(
                    current_node_id, current_node_id + (dimensions.length_divisions + 1)
                )

            if (
                support_definition == "fd"
                and y_node < dimensions.heigth_divisions
                and x_node < dimensions.length_divisions
            ):
                mesh.create_beam(
                    current_node_id,
                    current_node_id + 1 + (dimensions.length_divisions + 1),
                )

            if (
                support_definition == "bd"
                and y_node < dimensions.heigth_divisions
                and x_node > 0
            ):
                mesh.create_beam(
                    current_node_id,
                    current_node_id - 1 + (dimensions.length_divisions + 1),
                )

            if (
                support_definition == "x"
                and y_node < dimensions.heigth_divisions
                and x_node < dimensions.length_divisions
            ):
                node_x, node_y = np.average(
                        mesh.node_array[
                            np.array(
                                [
                                    current_node_id,
                                    current_node_id
                                    + 1
                                    + (dimensions.length_divisions + 1),
                                ]
                            ),
                            :,
                        ],
                        axis=0,
                )
                mesh.create_node(
                    Coordinates(node_x, node_y),
                    main_node=True,
                )

                created_mid_node_index = np.shape(mesh.node_array)[0] - 1

                mesh.create_beam(current_node_id, created_mid_node_index)
                mesh.create_beam(
                    created_mid_node_index,
                    current_node_id + 1 + (dimensions.length_divisions + 1),
                )
                mesh.create_beam(
                    current_node_id + (dimensions.length_divisions + 1),
                    created_mid_node_index,
                )
                mesh.create_beam(created_mid_node_index, current_node_id + 1)

    return mesh
