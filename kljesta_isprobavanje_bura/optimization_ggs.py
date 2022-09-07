#!/usr/bin/env python3

import sys
import os
import shutil

import numpy as np

sys.path.append("..")

from indago import GGS, PSO, FWA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

if os.path.exists('ccx_files'):
    shutil.rmtree('ccx_files')

if os.path.exists('img'):
    shutil.rmtree('img')

if os.path.exists('mesh_setup.pkl'):
    os.remove('mesh_setup.pkl')

mesh = gc.SimpleMeshCreator((2.980e9, 0.2), # Young modulus, Poisson
                            2e-3, # Maximum element size
                            100e-3, # Domain width
                            25e-3, # Domain height
                            (12, 4), # Frame grid division
                            'x' # Frame grid additional support
                            )

mesh.beam_height = 8e-3 # z thickenss (fixed)
mesh.minimal_beam_width = 1e-6 # (variable thickness) Beams with lower widths are removed

# Beam thickness initialization (in order to calculate refernt volume)
mesh.set_width_array(
    np.ones(
        np.shape(
            mesh.beam_array
        )[0]
    ) * 10e-3
)

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

# Referent volume calculation
w = np.full(mesh.beam_array.shape[0], 0, dtype=float)
w[used_beams] = 10e-3
mesh.set_width_array(w)
max_volume = mesh.calculate_mechanism_volume()


# Create symetry boundary condition (u_y=0, u_zz=0) and make nodes unremovable
for node in mesh.node_laso(
        [
            (1e-3, 1e-3),
            (71e-3, 1e-3),
            (71e-3, -1e-3),
            (1e-3, -1e-3)
        ],
        only_main_nodes=True):
    mesh.create_boundary(node, (0, 1, 1), set_unremovable=False)

# Symetry BC for Load node
# Set down-left corner BC (u_x=0, u_y=0) and unremovable
mesh.create_boundary(
    (0, 0),
    (0, 1, 1),
    set_unremovable=True
)

# Set up-left corner BC (u_x=0, u_y=0)
mesh.create_boundary(
    (0, 25e-3),
    (1, 1, 0),
    set_unremovable=False
)

# Set up-mid BC (u_x=0, u_y=0) and unremovable
mesh.create_boundary(
    (50e-3, 25e-3),
    (1, 1, 0),
    set_unremovable=True
)

# Driving force
mesh.create_force(
    (0, 0), # x, y
    (-100, 0) # F_x, F_y
)

for node in mesh.node_laso(
  [
   (70e-3, 12.3e-3),
   (70e-3, 12.6e-3),
   (110e-3, 12.6e-3),
   (110e-3, 12.3e-3)
   ]
):
    # Reaction forces
    mesh.create_force(node, (0, 5))
    # Final disp
    mesh.set_final_displacement(node, (0, -6.25e-3))


mesh.write_beginning_state()
ccx_manipulator = cm.calculix_manipulator(mesh)

def min_fun(beam_widths, unique_str=None, debug=False):
    #print(beam_widths)

    if os.path.exists(f'ccx_files/{unique_str}'):
        shutil.rmtree(f'ccx_files/{unique_str}')

    ccx_manipulator.used_mesh.beam_width_array[used_beams] = beam_widths
    ccx_results, used_nodes_read = ccx_manipulator.run_ccx(
        unique_str,
        von_mises_instead_of_principal=False
    )


    if ccx_results:
        volume = ccx_manipulator.used_mesh.calculate_mechanism_volume() / max_volume

        displacement, vm_stress = ccx_results
        u_goal = ccx_manipulator.used_mesh.final_displacement_array[:, 1:]
        u_calc = displacement[ccx_manipulator.final_displacement_node_positions]

        if u_goal.shape != u_calc.shape:
            print('!' * 20)
            print('shape error!', u_goal.shape, u_calc.shape)
            # print(beam_widths)
            print('!' * 20)
            # input('Press return to continue.')

        np.savez_compressed(f'ccx_files/{unique_str}/data.npz', displacement=displacement, vm_stress=vm_stress, used_nodes_read=used_nodes_read)
        # ccx_manipulator.used_mesh.current_displacement = displacement
        # ccx_manipulator.used_mesh.current_stress = vm_stress

        errors = u_calc - u_goal

        y_err_diff = np.abs(errors[0, 1] - errors[1, 1]) / np.average(np.abs(u_goal[:, 1]))

        m = np.abs(u_goal) > 0
        errors[m] = errors[m] / u_goal[m]

        errors = np.abs(errors)

        # x_err = np.sum(errors[:, 0]**2) #/ 100e-3 #* 6
        # y_err = np.sum(errors[:, 1]**2) #/ 25e-3 #* 2
        x_err = np.average(errors[:, 0]) #/ 100e-3 #* 6
        y_err = np.average(errors[:, 1]) #/ 25e-3 #* 2

        stress_cnstr = vm_stress.max() - 10e6 # <= 0

        if debug:
            print(f'{u_goal=}')
            print(f'{u_calc=}')
            print(f'{m=}')
            # print(f'{displacement=}')
            # print(f'{ccx_manipulator.final_displacement_node_positions=}')

        return volume, x_err, y_err, y_err_diff, 0, stress_cnstr

    else:
        return np.nan, np.nan, np.nan, np.nan, 1, 0


dims = np.size(used_beams)
optimizer = GGS()
optimizer.dimensions = dims
optimizer.lb = 0 * 1e-3
optimizer.ub = 10 * 1e-3
optimizer.iterations = 1000
optimizer.maximum_evaluations = 200000

optimizer.evaluation_function = min_fun
optimizer.objectives = 4
optimizer.objective_labels = ['vol', 'x_err', 'y_err', 'y_err_diff']
# optimizer.objective_weights = [0.001, 0.001, 0.99, 0.008]
# optimizer.objective_weights = [0, 0, 0.99, 0.01] # Ovo radi vrlo dobro
optimizer.objective_weights = [1e-2, 0, 1, 1]
optimizer.constraints = 2
optimizer.constraint_labels = ['invalid_sim', 'stress']

# optimizer.safe_evaluation = True
optimizer.params['n'] = 101
optimizer.params['k_max'] = 2

optimizer.eval_fail_behavior = 'ignore'
# optimizer.eval_retry_attempts = 10
# optimizer.eval_retry_recede = 0.05

optimizer.number_of_processes = 'maximum'

optimizer.forward_unique_str = True
optimizer.monitoring = 'dashboard'
# optimizer.monitoring = 'basic'


valid = False
while not valid:

    ub_perc = int(sys.argv[2]) / int(sys.argv[1])
    x0 = np.full(optimizer.dimensions, optimizer.ub * ub_perc)
    # x0 = optimizer.lb + np.random.random(dims) * (optimizer.ub - optimizer.lb)
    # x0[129] = 0
    #x0 = np.random.random(dims) * 4e-3 + 1.8e-3
    if os.path.exists('ccx_files/x0'):
        shutil.rmtree('ccx_files/x0')
    r = min_fun(x0, 'x0', debug=False)
    # print(r)
    valid = not r[4]

# input('Press return to continue.')
test_dir = './ccx_files'

# postavke animacije
drawer = cv.mesh_drawer()
drawer.from_object(mesh)

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

        drawer.my_ax.clear()
        drawer.my_info_ax.clear()
        drawer.my_res_ax.clear()
        drawer.my_fitness_ax.clear()

        npz = np.load(f'{test_dir}/best_it{it}/data.npz')

        kljesta_info = {
            'Iteracija': int(it),
            'h ' + '[m]': f'{mesh.beam_height:.5E}',
            'Volumen': f'{candidates[0].O[0]*100:.2f}%',
            'x error': f'{candidates[0].O[1]*100:.2f}%',
            'y error': f'{candidates[0].O[2]*100:.2f}%',
            'y error difference': f'{candidates[0].O[3]*100:.2f}%'
        }

        ccx_manipulator.used_mesh.beam_width_array[used_beams] = candidates[0].X

        drawer.make_drawing(kljesta_info,
                            ccx_manipulator.used_mesh,
                            npz['displacement'],
                            npz['vm_stress'],
                            npz['used_nodes_read'],
                            (10e6, 0),
                            displacement_scale=1,
                            beam_names=False)

        drawer.plot_obj_constr_fitness(it,
                                       candidates[0].O,
                                       candidates[0].C,
                                       candidates[0].f)

        if it != 0:
            drawer.check_and_make_copies_best(it-1)

        drawer.save_drawing(f'best_{it}')

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










"""
SCIPY minimize
"""

def eval_penalty(beam_widths, debug=False):

    unique_str = 'sim_' + generateRandomAlphaNumericString(20)
    volume, x_err, y_err, y_err_diff, invalid = min_fun(beam_widths, unique_str, debug)


    fit = volume * 0.001 + x_err + x_err * 0.001 + y_err * 0.997 + y_err_diff * 0.001
    if invalid:
        fit = 1e10
        print(f'Invalid simulation in {unique_str}')
        print(beam_widths)
        np.savez_compressed(f'debug_{unique_str}.npz', x=beam_widths)
        input('Pres return to continue')

    print(beam_widths)
    print([volume, x_err, y_err, y_err_diff], invalid, fit)

    # Remove candidates' directories
    if os.path.exists(f'{test_dir}/{unique_str}'):
        shutil.rmtree(f'{test_dir}/{unique_str}')

    return fit

if False:
    # SLSQP
    bounds = [_lbub for _lbub in zip(optimizer.lb, optimizer.ub)]
    # print(f'{bounds=}')
    opt = minimize(eval_penalty, x0,
                   method='SLSQP',
                   bounds=bounds,
                   options={'eps': 1e-4, # dx_i precision
                            'ftol': 1e-20,
                            'disp': True,
                            'maxiter': 20000,
                   }
                   )
    print(opt)

if False:
    # L-BFGS-B
    bounds = [_lbub for _lbub in zip(optimizer.lb, optimizer.ub)]
    # print(f'{bounds=}')
    opt = minimize(eval_penalty, x0,
                   method='L-BFGS-B',
                   bounds=bounds,
                   options={'eps': 1e-4, # dx_i precision
                            'gtol': 1e-50,
                            'ftol': 1e-50,
                            'maxcor': 50,
                            'makls': 20,
                   })
    print(opt)
