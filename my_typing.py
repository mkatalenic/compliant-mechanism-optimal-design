#!/usr/bin/env python3

from collections import namedtuple
from typing import Literal

# Variables as named tuple
# For easier navigation and code readability
Material = namedtuple("Material", "E Poisson")
Coordinates = namedtuple("CartesianCoordinates", "x y")
Boundary = namedtuple("BoundaryType", "x_translation y_translation z_rotation")
Force = namedtuple("ForceVector", "x_component y_component")
NodeDisplacement = namedtuple("DisplacementVector", "x_component y_component")
MeshDimensions = namedtuple(
    "MeshDimensions", ["length", "heigth", "length_divisions", "heigth_divisions"]
)
Stress = namedtuple(
    "StressMatrix",
    ["sigma_x", "sigma_y", "sigma_z", "tau_xy", "tau_yz", "tau_xz"],
)
MeshSupportDefinitions = Literal["fd", "bd", "x", None]
