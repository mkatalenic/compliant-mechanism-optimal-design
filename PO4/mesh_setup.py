#!/usr/bin/env python3

import numpy as np
import os
import sys
from math import tan, radians, cos, sin
sys.path.append("..")

import geometry_creation as gc
from my_typing import (
    Material,
    MeshDimensions,
    Boundary,
    Force,
    NodeDisplacement,
)

used_mesh = gc.simple_mesh_creator(
    material=Material(2.980e9, 0.2),
    max_el_size=2e-3,
    dimensions=MeshDimensions(60e-3, 60e-3, 3, 3),
    support_definition="x",
)

used_mesh.beam_height = 8e-3
used_mesh.minimal_beam_width = 5e-4

# Set force upper right corner
used_mesh.create_force((60, 60), Force(1, -30))

# Set down-left corner BC (u_x=1, u_y=0, u_zz=1)
used_mesh.create_boundary((0, 0), Boundary(0, 1, 0), set_unremovable=True)

# Set down-right corner BC (u_x=0, u_y=0, u_zz=1)
used_mesh.create_boundary((60e-3, 0), Boundary(1, 1, 0), set_unremovable=True)

# lower boundery
bound_nodes = used_mesh.node_laso(
        [
            (1e-3, 1e-3),
            (59e-3, 1e-3),
            (59e-3, -1e-3),
            (1e-3, -1e-3)
        ],
        only_main_nodes=True)
for node in bound_nodes:
    used_mesh.create_boundary(node, Boundary(0, 1, 0), set_unremovable=False)

# final node positions
nagib = radians(20)
selected_nodes = used_mesh.node_laso(
        [
            (-1e-3, 61e-3),
            (61e-3, 61e-3),
            (61e-3, 59e-3),
            (-1e-3, 59e-3)
        ],
        only_main_nodes=True)

selected_node_size = len(selected_nodes)
for node_no, node in enumerate(selected_nodes):
    x_displacement = node_no * 60e-3 / selected_node_size - node_no * 60e-3 / selected_node_size * cos(nagib)
    y_displacement = - node_no * 60e-3 / selected_node_size * sin(nagib)
    used_mesh.set_final_displacement(node, NodeDisplacement(
        x_displacement, y_displacement
    ))


MAXIMAL_BEAM_WIDTH = 3e-3
used_mesh.set_width_array(MAXIMAL_BEAM_WIDTH)
MAXIMAL_MESH_VOLUME = used_mesh.calculate_mechanism_volume()

BEGINNING_DESIGN_VECTOR = np.full(used_mesh.beam_width_array.size, 0.5 * MAXIMAL_BEAM_WIDTH)

BEAM_INTERFACE = used_mesh.beam_laso(
    [
        (-1e-3, 61e-3),
        (61e-3, 61e-3),
        (61e-3, 59e-3),
        (-1e-3, 59e-3),
    ], only_main_nodes=True
)

def create_and_write_mesh():
    """Pokretanje i zapisivanje mreže"""
    used_mesh.write_beginning_state()  # Spremanje početne konfiguracije mreže


if __name__ == "__main__":
    create_and_write_mesh()
