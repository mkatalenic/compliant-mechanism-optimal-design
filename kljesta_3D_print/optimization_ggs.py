#!/usr/bin/env python3

import sys
import os
import shutil

import numpy as np

sys.path.append("..")

from indago import GGS
import matplotlib.pyplot as plt
import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

mesh = gc.SimpleMeshCreator((2980e6, 0.2),
                            10e-3,
                            100e-3,
                            25e-3,
                            (6, 2),
                            'x')

mesh.minimal_beam_width = 2e-3
mesh.beam_height = 8e-3


mesh.set_width_array(
    np.ones(
        np.shape(
            mesh.beam_array
        )[0]
    ) * 5e-3
)

max_volume = mesh.calculate_mechanism_volume()

mesh.set_width_array(
    np.zeros(
        np.shape(
            mesh.beam_array
        )[0]
    )
)

# remove unwanted beams
removed_beams = mesh.beam_laso(
    [
        (70e-3, -1e-3),
        (70e-3, 12e-3),
        (110e-3, 12e-3),
        (110e-3, -1e-3)
    ]
)

used_beams = [
    x for x in range(np.shape(mesh.beam_array)[0]) if x not in removed_beams
]

for node in mesh.node_laso(
        [
            (-1e-3, 1e-3),
            (67e-3, 1e-3),
            (67e-3, -1e-3),
            (-1e-3, -1e-3)
        ],
        only_main_nodes=False):
    mesh.create_boundary(node,
                         (0, 1, 1),
                         set_unremovable=True)

mesh.create_boundary(
    (0, 25e-3),
    (1, 1, 0),
    set_unremovable=True
)

mesh.create_initial_displacement(
    (0, 0),
    (5e-3, 0)
)

# mesh.create_force(
#     (0, 0),
#     (10, 0)
# )

mesh.set_final_displacement(
    (3/2*100e-3, 0),
    (6e-3, 0)
)

mesh.set_final_displacement(
    (6/5*100e-3, 12.5e-3),
    (0, -6e-3)
)

mesh.set_final_displacement(
    (100e-3, 12.5e-3),
    (-2e-3, -8e-3)
)

mesh.write_beginning_state()
ccx_manipulator = cm.calculix_manipulator(mesh)

import random
import string


def generateRandomAlphaNumericString(length):
    # Generate alphanumeric string
    letters = string.ascii_lowercase + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str

# drawer = cv.mesh_drawer()
# drawer.from_object(ccx_manipulator.used_mesh)
# drawer.my_figure.suptitle('Optimizacija klješta')


def min_fun(beam_widths, unique_str=None):
    #print(beam_widths)

    ccx_manipulator.used_mesh.beam_width_array[used_beams] = beam_widths
    ccx_results = ccx_manipulator.run_ccx(
        unique_str,
        delete_after_completion=True,
        von_mises_instead_of_principal=True
    )

    
    if ccx_results:
        (displacement, vm_stress) = ccx_results

        volume_weight = 0.05

        # error_weights = (1 - volume_weight)/np.size(
        #     ccx_manipulator.final_displacement_node_positions
        # )
        #print(ccx_manipulator.used_mesh.final_displacement_array[:, 1:])
        #print(displacement[ccx_manipulator.final_displacement_node_positions])
        u_goal = ccx_manipulator.used_mesh.final_displacement_array[:, 1:]
        u_calc = displacement[ccx_manipulator.final_displacement_node_positions]
        if u_goal.shape != u_calc.shape:
            print('!' * 20)
            print('shape error!', u_goal.shape, u_calc.shape)
            print(beam_widths)
            print('!' * 20)
            # input(' >> Press return to continue.')
        
        errors = ccx_manipulator.used_mesh.final_displacement_array[:, 1:] -\
            displacement[ccx_manipulator.final_displacement_node_positions]
        # print(errors)

        mechanism_volume = ccx_manipulator.used_mesh.calculate_mechanism_volume()

        dimensionless_vol_w_weights = mechanism_volume/max_volume*volume_weight

        # x_errors_w_weights = np.sum(errors[:, 0]**2) * error_weights / 100e-3 * 6
        # y_errors_w_weights = np.sum(errors[:, 1]**2) * error_weights / 25e-3 * 2
        
        x_errors_w_weights = np.sum(errors[:, 0]**2) #/ 100e-3 #* 6
        y_errors_w_weights = np.sum(errors[:, 1]**2) #/ 25e-3 #* 2

        

        # print(50*'-')
        # print(f'mechanism_volume: {mechanism_volume}')
        # print(f'weighted_sum: {out_weighted_sum}')
        # print('widths:')
        # print(beam_widths)
        # print(50*'-')

        # print(dimensionless_vol_w_weights, x_errors_w_weights, y_errors_w_weights, 0)
        return dimensionless_vol_w_weights, x_errors_w_weights, y_errors_w_weights, 0
    else:
        
        return 0, 0, 0, 1

dims = np.size(used_beams)
optimizer = GGS()
optimizer.dimensions = dims
optimizer.lb = np.ones((optimizer.dimensions)) * 0
optimizer.ub = np.ones((optimizer.dimensions)) * 1e-2
optimizer.iterations = 1000
optimizer.maximum_evaluations = 20000

optimizer.evaluation_function = min_fun
optimizer.objectives = 3
optimizer.objective_labels = ['vol', 'x_err', 'y_err']
# optimizer.objective_weights = [0.7, 0.3]
optimizer.constraints = 1
optimizer.constraint_labels = ['valid_sim']

# optimizer.safe_evaluation = True
optimizer.params['n'] = 101
optimizer.params['k_max'] = 2

optimizer.eval_fail_behavior = 'ignore'
# optimizer.eval_retry_attempts = 10
# optimizer.eval_retry_recede = 0.05

optimizer.number_of_processes = 'maximum'

optimizer.forward_unique_str = True
optimizer.monitoring = 'basic'

valid = False
while not valid:
    x0 = optimizer.lb + np.random.random(dims) * (optimizer.ub - optimizer.lb)
    #x0 = np.random.random(dims) * 4e-3 + 1.8e-3
    if os.path.exists('ccx_files/x0'):
        shutil.rmtree('ccx_files/x0')
    r = min_fun(x0, 'x0')
    print(r)
    valid = not r[3]
    
# r = min_fun(x0, 'test')            
# print(r)
optimizer.X0 = x0

results = optimizer.optimize()

optimizer.results.plot_convergence()
axes = plt.gcf().axes
axes[0].set_yscale('log')
axes[1].set_yscale('symlog')
plt.savefig('optimization.png')