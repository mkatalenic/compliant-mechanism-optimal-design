#!/usr/bin/env python3

import numpy as np

import geometry_creation as gc
import calculix_manipulation as cm
import case_visualisation as cv
import os
import shutil

'''
Vrijednosti se zadaju u mikrometrima, gramima i sekundama
E = 1 N/m^2 = 10^-3 g/micrometer*s^2
F = 1 N = 10^9 g*micrometer/s^2
'''

example_mesh = gc.SimpleMeshCreator((160e3, 0.29),
                                    10,
                                    200,
                                    100,
                                    (6, 4),
                                    'x')

example_mesh.minimal_beam_width = 5e-5

# example_mesh.beam_height = 0.053
# example_mesh.beam_height = 53

debele = [112, 113, 115, 114, 116, 111, 148, 118, 121, 119, 122, 117,
          75, 76, 78, 77, 74, 79, 81, 82, 83, 84, 85, 87, 80, 41, 40,
          42, 44, 47, 46, 45, 48, 43, 50, 51, 53, 54, 49, 11, 13, 14, 12,
          17, 16, 15, 19, 20, 21, 63, 64, 26, 27, 58, 100, 106, 107,
          145, 146, 151, 133, 134, 132, 136, 139, 135, 137]

tanke = [153, 144, 147, 138, 101, 57, 9, 18, 23, 28, 66, 142, 60, 68]

example_mesh.set_width_array(
    np.zeros(
        np.shape(
            example_mesh.beam_array
        )[0]
    )
)

example_mesh.beam_width_array[debele] = 5

example_mesh.create_boundary(
    (200, 100),
    (1, 1, 0),
    set_unremovable=True
)

example_mesh.create_boundary(
    (0, 100),
    (1, 1, 0),
    set_unremovable=True
)

example_mesh.create_force(
    (100, 0),
    (0, 500)
)

example_mesh.set_final_displacement(
    (100, 100),
    (0, -0.29)
)

example_mesh.write_beginning_state()

ccx_helper = cm.calculix_manipulator(example_mesh)

from indago import PSO


def evaluation(all_widths, unique_str=None):

    heigth = all_widths[0]
    widths = all_widths[1:]

    ccx_helper.used_mesh.beam_height = heigth
    ccx_helper.used_mesh.beam_width_array[tanke] = widths
    ccx_results = ccx_helper.run_ccx(unique_str)

    if ccx_results is False:
        return np.nan, np.nan, np.nan, np.nan
    else:
        (displacement, _) = ccx_results
        y_error = np.sum(
            (ccx_helper.used_mesh.final_displacement_array[:, 1:][:, 1] -
             displacement[ccx_helper.final_diplacement_node_positions][:, 1])**2,
            axis=0
        )
        x_error = abs(float(
            displacement[ccx_helper.final_diplacement_node_positions][:, 0]
        )) - 1e-6
        mech_volume = ccx_helper.used_mesh.calculate_mechanism_volume()

        max_y_error = abs(y_error) - 1e-3

        return y_error, mech_volume, x_error, max_y_error


optimizer = PSO()
optimizer.dimensions = np.size(tanke) + 1
optimizer.lb = np.ones((optimizer.dimensions)) * 1e-4
optimizer.lb[0] = 0.053
optimizer.ub = np.ones((optimizer.dimensions)) * 5
optimizer.ub[0] = 10
optimizer.iterations = 1000

optimizer.evaluation_function = evaluation
optimizer.objectives = 2
optimizer.objective_labels = ['y_error', 'mechanism volume']
optimizer.objective_weights = [0.7, 0.3]
optimizer.constraints = 2
optimizer.constraint_labels = ['x_error', 'max possible y_error']

# optimizer.safe_evaluation = True

optimizer.eval_fail_behavior = 'retry'
optimizer.eval_retry_attempts = 10
optimizer.eval_retry_recede = 0.05

optimizer.number_of_processes = 'maximum'

optimizer.forward_unique_str = True

optimizer.monitoring = 'dashboard'

test_dir = './ccx_files'
drawing = cv.mesh_drawer()


def post_iteration_processing(it, candidates, best):
    if candidates[0] <= best:
        # Keeping only overall best solution
        # if os.path.exists(f'{test_dir}/best'):
        #     shutil.rmtree(f'{test_dir}/best')
        # os.rename(f'{test_dir}/{candidates[0].unique_str}', f'{test_dir}/best')

        # Keeping best solution of each iteration (if it is the best overall)
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


optimizer.post_iteration_processing = post_iteration_processing
results = optimizer.optimize()
