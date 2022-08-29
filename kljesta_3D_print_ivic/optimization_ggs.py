#!/usr/bin/env python3

import sys
import os
import shutil

import numpy as np

sys.path.append("..")

from indago import GGS, PSO
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

mesh = gc.SimpleMeshCreator((2.980e9, 0.2), # Young modulus, Poisson
                            5e-3, # Maximum element size
                            100e-3, # Domain width
                            25e-3, # Domain height
                            (6, 2), # Frame grid division
                            'x' # Frame grid additional support
                            )

mesh.minimal_beam_width = 2e-4 # (variable thickness) Beams with lower widths are removed
mesh.beam_height = 8e-3 # z thickenss (fixed)

# Beam thickness initialization (in order to calculate refernt volume)
mesh.set_width_array(
    np.ones(
        np.shape(
            mesh.beam_array
        )[0]
    ) * 5e-3
)
max_volume = mesh.calculate_mechanism_volume()

# Remove unwanted beams (removes beam inside of polygon)
removed_beams = mesh.beam_laso(
    [
        (70e-3, -1e-3),
        (70e-3, 12e-3),
        (110e-3, 12e-3),
        (110e-3, -1e-3)
    ]
)

# Set thickness to zero for all beams
mesh.set_width_array(np.zeros(mesh.beam_array.shape[0]))

# Beam to be optimized (varying thickness)
used_beams = [
    x for x in range(mesh.beam_array.shape[0]) if x not in removed_beams
]

# Create symetry boundary condition (u_y=0, u_zz=0) and make nodes unremovable
for node in mesh.node_laso(
        [
            (-1e-3, 1e-3),
            (51e-3, 1e-3),
            (51e-3, -1e-3),
            (-1e-3, -1e-3)
        ],
        only_main_nodes=True):
    mesh.create_boundary(node, (0, 1, 1), set_unremovable=True)

# Set corner BC (u_x=0, u_y=0) and unremovable
mesh.create_boundary(
    (0, 25e-3),
    (1, 1, 0),
    set_unremovable=True
)

# mesh.create_initial_displacement(
#     (0, 0),
#     (5e-3, 0)
# )

mesh.create_force(
    (0, 0), # x, y
    (100, 0) # F_x, F_y
)

# mesh.set_final_displacement(
#     (2/3*100e-3, 0), # x, y
#     (6e-3, 0) # u_x, u_y
# )

mesh.set_final_displacement(
    (5/6*100e-3, 12.5e-3),
    (0, -6.25e-3)
)

mesh.set_final_displacement(
    (100e-3, 12.5e-3),
    (0, -6.25e-3)
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

def min_fun(beam_widths, unique_str=None, debug=False):
    #print(beam_widths)

    if os.path.exists(f'ccx_files/{unique_str}'):
        shutil.rmtree(f'ccx_files/{unique_str}')
        
    ccx_manipulator.used_mesh.beam_width_array[used_beams] = beam_widths
    ccx_results = ccx_manipulator.run_ccx(
        unique_str,
        von_mises_instead_of_principal=True
    )

    
    if ccx_results:        
        volume = ccx_manipulator.used_mesh.calculate_mechanism_volume() / max_volume
        
        # Ovo mi baš nije sasvim jasno:
        displacement, vm_stress = ccx_results
        u_goal = ccx_manipulator.used_mesh.final_displacement_array[:, 1:]
        u_calc = displacement[ccx_manipulator.final_displacement_node_positions]

        if u_goal.shape != u_calc.shape:
            print('!' * 20)
            print('shape error!', u_goal.shape, u_calc.shape)
            # print(beam_widths)
            print('!' * 20)
            # input('Press return to continue.')
        
        errors = u_calc - u_goal
        
        m = np.abs(u_goal) > 0
        errors[m] = np.abs(errors[m] / u_goal[m])
        
        # x_err = np.sum(errors[:, 0]**2) #/ 100e-3 #* 6
        # y_err = np.sum(errors[:, 1]**2) #/ 25e-3 #* 2
        x_err = np.average(errors[:, 0]) #/ 100e-3 #* 6
        y_err = np.average(errors[:, 1]) #/ 25e-3 #* 2

        if debug:
            print(f'{u_goal=}')
            print(f'{u_calc=}')
            print(f'{m=}')
            # print(f'{displacement=}')
            # print(f'{ccx_manipulator.final_displacement_node_positions=}')
            
        return volume, x_err, y_err, 0
    
    else:        
        return np.nan, np.nan, np.nan, 1

def eval_penalty(beam_widths, debug=False):
    
    unique_str = generateRandomAlphaNumericString(20)
    volume, x_err, y_err, valid = min_fun(beam_widths, unique_str, debug)
    fit = volume * 0.001 + x_err + x_err * 0.001 + y_err * 0.998
    
    print(beam_widths)
    print(volume, x_err, y_err, valid, fit)
    if valid:
        return fit
    else:
        return 1e10
    
dims = np.size(used_beams)
optimizer = GGS()
optimizer.dimensions = dims
optimizer.lb = np.ones((optimizer.dimensions)) * 0
optimizer.ub = np.ones((optimizer.dimensions)) * 10 * 1e-3
optimizer.iterations = 1000
optimizer.maximum_evaluations = 20000

optimizer.evaluation_function = min_fun
optimizer.objectives = 3
optimizer.objective_labels = ['vol', 'x_err', 'y_err']
optimizer.objective_weights = [0.001, 0.001, 0.998]
optimizer.constraints = 1
optimizer.constraint_labels = ['valid_sim']

# optimizer.safe_evaluation = True
optimizer.params['n'] = 51
optimizer.params['k_max'] = 2

optimizer.eval_fail_behavior = 'ignore'
# optimizer.eval_retry_attempts = 10
# optimizer.eval_retry_recede = 0.05

optimizer.number_of_processes = 'maximum'

optimizer.forward_unique_str = True
optimizer.monitoring = 'dashboard'

valid = False
while not valid:
    x0 = optimizer.ub * 0.8
    # x0 = optimizer.lb + np.random.random(dims) * (optimizer.ub - optimizer.lb)
    # x0[129] = 0
    #x0 = np.random.random(dims) * 4e-3 + 1.8e-3
    if os.path.exists('ccx_files/x0'):
        shutil.rmtree('ccx_files/x0')
    r = min_fun(x0, 'x0', debug=True)
    print(r)
    valid = not r[3]

# input('Press return to continue.')
test_dir = './ccx_files'


def post_iteration_processing(it, candidates, best):
    if candidates[0] <= best:
        # Keeping only overall best solution
        # if os.path.exists(f'{test_dir}/best'):
        #     shutil.rmtree(f'{test_dir}/best')
        # os.rename(f'{test_dir}/{candidates[0].unique_str}',
        # f'{test_dir}/best')

        # Keeping best solution of each iteration (if it is the best overall)

        if os.path.exists(f'{test_dir}/best_it{it}'):
            shutil.rmtree(f'{test_dir}/best_it{it}')

        os.rename(f'{test_dir}/{candidates[0].unique_str}',
                  f'{test_dir}/best_it{it}')

        # Log keeps track of new best solutions in each iteration
        with open(f'{test_dir}/log.txt', 'a') as log:
            X = ', '.join(f'{x:13.6e}' for x in candidates[0].X)
            O = ', '.join(f'{o:13.6e}' for o in candidates[0].O)
            C = ', '.join(f'{c:13.6e}' for c in candidates[0].C)
            log.write(f'{it:6d} X:[{X}], O:[{O}], C:[{C}]' +
                      f' fitness:{candidates[0].f:13.6e}\n')

        # Remove the best from candidates
        # (since its directory is already renamed)
        candidates = np.delete(candidates, 0)

    # Remove candidates' directories
    for c in candidates:
        shutil.rmtree(f'{test_dir}/{c.unique_str}')


    return

if True:
    optimizer.X0 = x0
    optimizer.post_iteration_processing = post_iteration_processing
    results = optimizer.optimize()
    print(results)
    
    optimizer.results.plot_convergence()
    axes = plt.gcf().axes
    # axes[0].set_yscale('log')
    # axes[1].set_yscale('symlog')
    plt.savefig('optimization.png')
    
    x0 = results.X

if False:
    
    opt = minimize(eval_penalty, x0, 
                   method='SLSQP',
                   tol=1e-12,
                   bounds=[_lbub for _lbub in zip(optimizer.lb, optimizer.ub)],
                   options={'eps':1e-4,
                            'ftol':1e-12,}
                   )
