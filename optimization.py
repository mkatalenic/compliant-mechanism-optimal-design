#!/usr/bin/env python3

import shutil
import os
import sys

from pyparsing import OpAssoc


import matplotlib.pyplot as plt

from indago import GGS, PSO, SSA, BA, FWA, MRFO


class OptimFactory:
    def __init__(self) -> None:
        self.GGS = GGS()
        self.PSO = PSO()
        self.SSA = SSA()
        self.BA = BA()
        self.FWA = FWA()
        self.MRFO = MRFO()


def remove_old_dirs(path_to_config_dir) -> None:
    """Brisanje nepotrebnih direktorija"""
    for directory in ["ccx_files", "img"]:
        full_dir_path = os.path.join(os.getcwd(), path_to_config_dir, directory)
        if os.path.exists(full_dir_path):
            shutil.rmtree(full_dir_path)

    for created_file in ["mesh_setup.pkl", "optimization.png"]:
        full_file_path = os.path.join(os.getcwd(), path_to_config_dir, created_file)
        if os.path.exists(full_file_path):
            os.remove(full_file_path)


if __name__ == "__main__":

    OF = OptimFactory()

    path_to_config_dir = sys.argv[1]
    os.chdir(os.path.join(os.getcwd(), path_to_config_dir))
    sys.path.append(".")

    remove_old_dirs(path_to_config_dir)

    import mesh_setup

    mesh_setup.create_and_write_mesh()

    from optimization_configuration import (
        OPTIMIZATOR_NAME,
        OPTIMIZATION_DIMENSIONS,
        OPTIMIZATION_UPPER_BOUND,
        OPTIMIZATION_ITERATIONS,
        OPTIMIZATION_MAXIMUM_EVALUATIONS,
        OPTIMIZATION_OBJECTIVES_AND_WEIGHTS,
        OPTIMIZATION_CONSTRAINTS_LABELS,
        OPTIMIZER_SPECIFIC_PARAMETERS,
        OPTIMIZATION_STARTING_DESIGN_VECTOR,
        minimization_function,
        post_iteration_processing,
    )

    optim_setup = {
        "dimensions": OPTIMIZATION_DIMENSIONS,
        "lb": 0,
        "ub": OPTIMIZATION_UPPER_BOUND,
        "iterations": OPTIMIZATION_ITERATIONS,
        "maximum_evaluations": OPTIMIZATION_MAXIMUM_EVALUATIONS,
        "evaluation_function": minimization_function,
        "objectives": len(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS),
        "objective_labels": list(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.keys()),
        "objective_weights": list(OPTIMIZATION_OBJECTIVES_AND_WEIGHTS.values()),
        "constraints": len(OPTIMIZATION_CONSTRAINTS_LABELS),
        "constraint_labels": OPTIMIZATION_CONSTRAINTS_LABELS,
        "params": OPTIMIZER_SPECIFIC_PARAMETERS,
        "eval_fail_behavior": "ignore",
        "number_of_processes": "maximum",
        "forward_unique_str": True,
        "X0": OPTIMIZATION_STARTING_DESIGN_VECTOR,
        "post_iteration_processing": post_iteration_processing,
        "monitoring": "basic",
    }

    if OPTIMIZATOR_NAME in OF.__dict__.keys():
        optimizer = getattr(OF, OPTIMIZATOR_NAME)

        for setup_key, setup_value in optim_setup.items():
            optimizer.__dict__[setup_key] = setup_value

        results = optimizer.optimize()
        print(results)

        optimizer.results.plot_convergence()
        axes = plt.gcf().axes
        plt.savefig("optimization.png")

        x0 = results.X
