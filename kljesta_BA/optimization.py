#!/usr/bin/env python3

import shutil
import os

import matplotlib.pyplot as plt

import mesh_setup

mesh_setup.create_and_write_mesh()


from optimization_configuration import (
    OPTIMIZATION_DIMENSIONS,
    OPTIMIZATION_UPPER_BOUND,
    OPTIMIZATION_ITERATIONS,
    OPTIMIZATION_MAXIMUM_EVALUATIONS,
    OPTIMIZATION_OBJECTIVES_AND_WEIGHTS,
    OPTIMIZATION_CONSTRAINTS_LABELS,
    OPTIMIZER_SPECIFIC_PARAMETERS,
    OPTIMIZATION_STARTING_DESIGN_VECTOR,
)

from optimization_configuration import minimization_function, post_iteration_processing

"""Brisanje nepotrebnih direktorija"""
for directory in ["ccx_files", "img"]:
    if os.path.exists(directory):
        shutil.rmtree(directory)

for created_file in ["mesh_setup.pkl", "optimization.png"]:
    if os.path.exists(created_file):
        os.remove(created_file)

from indago import PSO

optimizer = PSO()
optimizer.dimensions = OPTIMIZATION_DIMENSIONS
optimizer.lb = 0
optimizer.ub = OPTIMIZATION_UPPER_BOUND
optimizer.iterations = OPTIMIZATION_ITERATIONS
optimizer.maximum_evaluations = OPTIMIZATION_MAXIMUM_EVALUATIONS


optimizer.evaluation_function = minimization_function
optimizer.objectives = len(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS)
optimizer.objective_labels = list(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.keys())
optimizer.objective_weights = list(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.values())
optimizer.constraints = len(OPTIMIZATION_CONSTRAINTS_LABELS)
optimizer.constraint_labels = OPTIMIZATION_CONSTRAINTS_LABELS

optimizer.params = OPTIMIZER_SPECIFIC_PARAMETERS

optimizer.eval_fail_behavior = "ignore"

optimizer.number_of_processes = "maximum"

optimizer.forward_unique_str = True

x0 = OPTIMIZATION_STARTING_DESIGN_VECTOR

optimizer.X0 = x0
optimizer.post_iteration_processing = post_iteration_processing
optimizer.monitoring = "dashboard"
results = optimizer.optimize()
print(results)

optimizer.results.plot_convergence()
axes = plt.gcf().axes
plt.savefig("optimization.png")

x0 = results.X
