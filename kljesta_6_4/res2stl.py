#!/usr/bin/env python3

import numpy as np
import sys

sys.path.append("..")

import matplotlib.pyplot as plt

from indago import GGS, PSO, FWA
import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

import openpyscad as oscad

def res2stl(
        calculix_case: cm.calculix_manipulator,
        used_beams,
        output_name: 'str',
        mirror=False
):
    beam_height = calculix_case.used_mesh.beam_height*1e3
    beam_widths = calculix_case.calculated_widths[-1]*1e3

    beam_start_end = calculix_case.used_mesh.beam_array[used_beams, :2]
    first_nodes = calculix_case.used_mesh.node_array[beam_start_end[:,0]]*1e3
    last_nodes = calculix_case.used_mesh.node_array[beam_start_end[:,1]]*1e3
    beam_lengths = np.sqrt(
        np.sum((
            last_nodes - first_nodes
            )**2,
               axis=1)
    )
    beams_angle = np.rad2deg(np.arcsin(
        (
            last_nodes - first_nodes
        )[:,1] / beam_lengths
    ))

    beam = oscad.Cube([beam_lengths[0], beam_widths[0], beam_height]).translate([0, -beam_widths[0]/2, 0])
    mesh_stl_out = beam.rotate([0, 0, beams_angle[0]]).translate([first_nodes[0][0], first_nodes[0][1], 0])

    for idx in range(1, beam_widths.size):
        bw = beam_widths[idx]
        bl = beam_lengths[idx]

        beam = oscad.Cube([bl, bw, beam_height]).translate([0, -bw/2, 0])

        mesh_stl_out += beam.rotate([0, 0, beams_angle[idx]]).translate([first_nodes[idx][0], first_nodes[idx][1], 0])

    if mirror == 'x':
        mesh_stl_out += mesh_stl_out.mirror([0,1,0])
    if mirror == 'y':
        mesh_stl_out += mesh_stl_out.mirror([1,0,0])
    if mirror == 'xy':
        mesh_stl_out += mesh_stl_out.mirror([1,0,0])
        mesh_stl_out += mesh_stl_out.mirror([0,1,0])

    mesh_stl_out.write('kljesta.scad')

if __name__=='__main__':

    kljesta_mesh = cm.calculix_manipulator(
        gc.read_mesh_state()
    )

    kljesta_mesh.used_mesh.set_width_array(
        np.zeros(
            np.shape(
                kljesta_mesh.used_mesh.beam_array
            )[0]
        )
    )

    # Remove unwanted beams (removes beam inside of polygon)
    removed_beams = kljesta_mesh.used_mesh.beam_laso(
        [
            (70e-3, -1e-3),
            (70e-3, 12e-3),
            (110e-3, 12e-3),
            (110e-3, -1e-3)
        ]
    )

    # Beam to be optimized (varying thickness)
    used_beams = np.array([
        x for x in range(kljesta_mesh.used_mesh.beam_array.shape[0]) if x not in removed_beams
    ])

    used_case_name = 'ccx_files_7'

    kljesta_mesh.load_from_info(
        used_beams.size,
        log_txt_location=used_case_name
    )

    kljesta_mesh.used_mesh.beam_height = 8e-3

    res2stl(kljesta_mesh,
            used_beams,
            'kljesta_6_4',
            mirror='x')
