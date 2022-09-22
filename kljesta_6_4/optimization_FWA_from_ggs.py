#!/usr/bin/env python3

import sys
import os
import shutil

import numpy as np

sys.path.append("..")

import matplotlib.pyplot as plt

from indago import GGS, PSO, FWA
import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv

# skopirani mesh_setup.pkl od 10_5
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


x0_cases = np.empty((0, used_beams.size))

for case in range(1,11):

    kljesta_mesh.load_from_info(
        used_beams.size,
        log_txt_location=f'ccx_files_{case}'
    )

    x0_cases = np.append(
        x0_cases,
        [kljesta_mesh.calculated_widths[-1,:]],
        axis=0
    )

    kljesta_mesh.used_mesh.beam_height = 8e-3

    kljesta_mesh.load_best_ccx_solutions(best_it_location=f'ccx_files_{case}')

new_used_beams = used_beams[
    [np.average(x0_cases, axis=0)[beam] > kljesta_mesh.used_mesh.minimal_beam_width
     for beam in range(used_beams.size)]
]

# Referent volume calculation
w = np.full(kljesta_mesh.used_mesh.beam_array.shape[0], 0, dtype=float)
w[new_used_beams] = 5e-3
kljesta_mesh.used_mesh.set_width_array(w)
max_volume = kljesta_mesh.used_mesh.calculate_mechanism_volume()

def min_fun(beam_widths, unique_str=None, debug=False):

    if os.path.exists(f'ccx_files/{unique_str}'):
        shutil.rmtree(f'ccx_files/{unique_str}')

    w = np.full(kljesta_mesh.used_mesh.beam_array.shape[0], 0, dtype=float)
    w[new_used_beams] = beam_widths
    if kljesta_mesh.used_mesh.set_width_array(w):
        ccx_results, used_nodes_read = kljesta_mesh.run_ccx(
            unique_str,
            von_mises_instead_of_principal=False
        )
    else:
        ccx_results = False


    if ccx_results:
        volume = kljesta_mesh.used_mesh.calculate_mechanism_volume() / max_volume

        displacement, vm_stress = ccx_results
        u_goal = kljesta_mesh.used_mesh.final_displacement_array[:, 1:]
        u_calc = displacement[kljesta_mesh.final_displacement_node_positions]

        if u_goal.shape != u_calc.shape:
            print('!' * 20)
            print('shape error!', u_goal.shape, u_calc.shape)
            # print(beam_widths)
            print('!' * 20)
            # input('Press return to continue.')

        np.savez_compressed(f'ccx_files/{unique_str}/data.npz', displacement=displacement, vm_stress=vm_stress, used_nodes_read=used_nodes_read)
        # kljesta_mesh.used_mesh.current_displacement = displacement
        # kljesta_mesh.used_mesh.current_stress = vm_stress

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
            # print(f'{kljesta_mesh.final_displacement_node_positions=}')

        return volume, x_err, y_err, y_err_diff, 0, stress_cnstr

    else:
        return np.nan, np.nan, np.nan, np.nan, 1, 0

dims = new_used_beams.size
optimizer = FWA()
optimizer.dimensions = dims
optimizer.lb = 0 * 1e-3
optimizer.ub = 5 * 1e-3
optimizer.iterations = 1000
optimizer.maximum_evaluations = 200000

optimizer.evaluation_function = min_fun
optimizer.objectives = 4
optimizer.objective_labels = ['vol', 'x_err', 'y_err', 'y_err_diff']
optimizer.objective_weights = [1e-3, 1e-4, 1, 1]
optimizer.constraints = 2
optimizer.constraint_labels = ['invalid_sim', 'stress']

optimizer.eval_fail_behavior = 'ignore'

optimizer.number_of_processes = 'maximum'

optimizer.forward_unique_str = True
# optimizer.monitoring = 'dashboard'
optimizer.monitoring = 'basic'

# optimizer.params['swarm_size'] = 9 # number of PSO particles; default swarm_size=dimensions
# optimizer.params['inertia'] = 0.8 # PSO parameter known as inertia weight w (should range from 0.5 to 1.0), the other available options are 'LDIW' (w linearly decreasing from 1.0 to 0.4) and 'anakatabatic'; default inertia=0.72
# optimizer.params['cognitive_rate'] = 1.0 # PSO parameter also known as c1 (should range from 0.0 to 2.0); default cognitive_rate=1.0
# optimizer.params['social_rate'] = 1.0 # PSO parameter also known as c2 (should range from 0.0 to 2.0); default social_rate=1.0

# valid = False
# while not valid:
x0 = x0_cases[:, np.average(x0_cases, axis=0) > kljesta_mesh.used_mesh.minimal_beam_width]
    # x0 = optimizer.lb + np.random.random(dims) * (optimizer.ub - optimizer.lb)
    # x0[129] = 0
    #x0 = np.random.random(dims) * 4e-3 + 1.8e-3
    # if os.path.exists('ccx_files/x0'):
        # shutil.rmtree('ccx_files/x0')
    # r = min_fun(x0, 'x0', debug=False)
    # # print(r)
    # valid = not r[4]

drawer = cv.mesh_drawer()
drawer.from_object(kljesta_mesh.used_mesh)

test_dir = 'ccx_files'
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
            'h ' + '[m]': f'{kljesta_mesh.used_mesh.beam_height:.5E}',
            'Volumen': f'{candidates[0].O[0]*100:.2f}%',
            'x error': f'{candidates[0].O[1]*100:.2f}%',
            'y error': f'{candidates[0].O[2]*100:.2f}%',
            'y error difference': f'{candidates[0].O[3]*100:.2f}%'
        }

        w = np.full(kljesta_mesh.used_mesh.beam_array.shape[0], 0, dtype=float)
        w[new_used_beams] = candidates[0].X
        kljesta_mesh.used_mesh.set_width_array(w)

        drawer.make_drawing(kljesta_info,
                            kljesta_mesh.used_mesh,
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

        drawer.save_drawing(f'best_{it}')

        drawer.check_and_make_copies_best()

    # Remove the best from candidates
    # (since its directory is already renamed)
    candidates = np.delete(candidates, 0)

    # Remove candidates' directories
    for c in candidates:
        shutil.rmtree(f'{test_dir}/{c.unique_str}')

    return

if True:
    # ub_perc = float(sys.argv[2]) / float(sys.argv[1])
    # x0 = np.full(optimizer.dimensions, optimizer.ub * ub_perc)
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
