#!/usr/bin/env python3

import numpy as np

import geometry_creation as gc
import calculix_manipulation as cm

import openpyscad as oscad


def res2stl(
        pickled_mesh_location: str,
        log_file_location: str,
        output_name: str,
        mirror=False
):
    used_mesh = gc.read_mesh_state(pickled_mesh_location)
    beam_height = used_mesh.beam_height*1e3
    _, all_beam_widths = cm.load_widths_from_info(used_mesh,
                                       log_txt_location=log_file_location)
    beam_widths = all_beam_widths[-1]
    used_beams = np.arange(beam_widths.size)[beam_widths >= used_mesh.minimal_beam_width]
    beam_widths = beam_widths[beam_widths >= used_mesh.minimal_beam_width]

    beam_start_end = used_mesh.beam_array[used_beams, :2]
    first_nodes = used_mesh.node_array[beam_start_end[:,0]]*1e3
    last_nodes = used_mesh.node_array[beam_start_end[:,1]]*1e3
    beam_lengths = np.sqrt(
        np.sum((
            last_nodes - first_nodes
            )**2,
               axis=1)
    )
    beams_angle = np.rad2deg(np.arcsin(
        (
            last_nodes - first_nodes
        )[:, 1] / beam_lengths
    ))

    beam = oscad.Cube([beam_lengths[0], beam_widths[0], beam_height]).translate([0, -beam_widths[0]/2, 0])
    mesh_stl_out = beam.rotate([0, 0, beams_angle[0]]).translate([first_nodes[0][0], first_nodes[0][1], 0])

    for idx in range(1, beam_widths.size):
        bw = beam_widths[idx]
        bl = beam_lengths[idx]

        beam = oscad.Cube([bl, bw*1e3, beam_height]).translate([0, -bw/2, 0])

        mesh_stl_out += beam.rotate([0, 0, beams_angle[idx]]).translate([first_nodes[idx][0], first_nodes[idx][1], 0])

    if mirror == 'x':
        mesh_stl_out += mesh_stl_out.mirror([0,1,0])
    if mirror == 'y':
        mesh_stl_out += mesh_stl_out.mirror([1,0,0])
    if mirror == 'xy':
        mesh_stl_out += mesh_stl_out.mirror([1,0,0])
        mesh_stl_out += mesh_stl_out.mirror([0,1,0])

    mesh_stl_out.write(f'{output_name}.scad')

if __name__=='__main__':

    res2stl('FWA_PO/mesh_setup.pkl',
            'FWA_PO/ccx_files',
            'FWA_PO',
            mirror=None)
