#!/usr/bin/env python3

import numpy as np
import sys

sys.path.append("..")

import geometry_creation as gc

from my_typing import (
    Material,
    MeshDimensions,
    Boundary,
    Force,
    NodeDisplacement,
)

MAXIMAL_BEAM_WIDTH = 5e-3

# kljesta_mesh = gc.simple_mesh_creator(
#     material=Material(2.980e9, 0.2),
#     max_el_size=2e-3,
#     dimensions=MeshDimensions(100e-3, 25e-3, 12, 4),
#     support_definition="fd",
# )
kljesta_mesh = gc.simple_mesh_creator(
    material=Material(2.980e9, 0.2),
    max_el_size=2e-3,
    dimensions=MeshDimensions(100e-3, 25e-3, 6, 4),
    support_definition="fd",
)

kljesta_mesh.beam_height = 8e-3
kljesta_mesh.minimal_beam_width = 1e-6

kljesta_mesh.remove_beams(
    [(70e-3, -1e-3), (70e-3, 12e-3), (110e-3, 12e-3), (110e-3, -1e-3)]
)

kljesta_mesh.set_width_array(MAXIMAL_BEAM_WIDTH)

# Create symetry boundary condition (u_y=0, u_zz=0)
for node in kljesta_mesh.node_laso(
    [(1e-3, 1e-3), (71e-3, 1e-3), (71e-3, -1e-3), (1e-3, -1e-3)], only_main_nodes=True
):
    kljesta_mesh.create_boundary(node, Boundary(0, 1, 1), set_unremovable=False)

# Symetry BC for Load node
# Set down-left corner BC (u_y=0, u_zz=0) and unremovable
kljesta_mesh.create_boundary((0, 0), Boundary(0, 1, 1), set_unremovable=True)

# Set up-left corner BC (u_x=0, u_y=0)
kljesta_mesh.create_boundary((0, 25e-3), Boundary(1, 1, 0), set_unremovable=False)

# Set up-mid BC (u_x=0, u_y=0) and unremovable
kljesta_mesh.create_boundary((50e-3, 25e-3), Boundary(1, 1, 0), set_unremovable=True)

# Driving force
kljesta_mesh.create_force((0, 0), Force(100, 0))  # x, y  # F_x, F_y

for node in kljesta_mesh.node_laso(
    [(70e-3, 12.3e-3), (70e-3, 12.6e-3), (110e-3, 12.6e-3), (110e-3, 12.3e-3)]
):
    # Reaction forces
    kljesta_mesh.create_force(node, Force(0, 5))
    # Final disp
    kljesta_mesh.set_final_displacement(node, NodeDisplacement(0, -6.25e-3))


def create_and_write_mesh():
    kljesta_mesh.write_beginning_state()  # Spremanje početne konfiguracije mreže


if __name__ == "__main__":
    create_and_write_mesh()
