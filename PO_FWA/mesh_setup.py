#!/usr/bin/env python3

import numpy as np
import os
import sys
from math import radians, cos, sin
sys.path.append("..")

from calculix_manipulation import load_widths_from_info
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

MAXIMAL_BEAM_WIDTH = 3e-3
used_mesh.set_width_array(MAXIMAL_BEAM_WIDTH)

selected_node_size = len(selected_nodes)
for node_no, node in enumerate(selected_nodes):
    x_displacement = node_no * 60e-3 / selected_node_size - node_no * 60e-3 / selected_node_size * cos(nagib)
    y_displacement = - node_no * 60e-3 / selected_node_size * sin(nagib)
    used_mesh.set_final_displacement(node, NodeDisplacement(
        x_displacement, y_displacement
    ))

all_best_widths = np.empty((0, sum(used_mesh.beam_width_beginning_map)))
for log_file in os.listdir(os.path.join(os.getcwd(), 'logs')):

    iterations, widths = load_widths_from_info(
        used_mesh,
        log_txt_location='logs',
        log_name=log_file)

    all_best_widths = np.append(
        all_best_widths,
        np.array([widths[-1]]),
        axis=0
    )

all_best_widths_mapped = np.empty((all_best_widths.shape[0], used_mesh.beam_width_beginning_map.size), dtype=float)
for case, widths in enumerate(all_best_widths):
    all_best_widths_mapped[case][used_mesh.beam_width_beginning_map==True] = widths
    all_best_widths_mapped[case][used_mesh.beam_width_beginning_map==False] = 0.

# micanje greda koje je 8/10 GGS optimizacija proglasilo nepotrebnim
used_mesh.beam_width_beginning_map[used_mesh.beam_width_beginning_map == True] = np.sum(all_best_widths <= used_mesh.minimal_beam_width, axis=0) < 6
MAXIMAL_BEAM_WIDTH = 3e-3
used_mesh.set_width_array(MAXIMAL_BEAM_WIDTH)

BEGINNING_DESIGN_VECTOR = all_best_widths_mapped[:, used_mesh.beam_width_beginning_map]

MAXIMAL_BEAM_WIDTH = float(np.max(all_best_widths))
used_mesh.set_width_array(MAXIMAL_BEAM_WIDTH)
MAXIMAL_MESH_VOLUME = used_mesh.calculate_mechanism_volume()

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
