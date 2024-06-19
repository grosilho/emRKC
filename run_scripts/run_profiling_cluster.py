import os
import numpy as np
import subprocess
import re

from run_simulations_cluster import execute_with_dependencies, count_number_simulation, set_options, get_slurm_settings


def main():
    # Problem description
    domains_names = ["03_fastl_LA"]
    refinements = [2]
    fibrosiss = [True]
    ionic_models = ["HH", "CRN", "TTP"]

    # Numerical methods
    es_classes = ["RKC1"]
    integrators = ["exp_mES", "mES", "IMEXEXP"]
    splittings = ["exp_nonstiff"]
    safe_adds = [0]

    # read an initial value or use the hardcoded one?
    read_init_val = True

    # run more than once, for more accurate profiling results
    iter_indeces = list(range(0, 1))

    # Step sizes
    base_dt = 1.0
    min_k = 6
    max_k = 6
    ks = np.array(list(range(min_k, max_k + 1)))
    dts = base_dt * 2.0 ** (-ks)

    # This data is used to estimate the running time, used in the slurm job submission
    tend = 50.0

    # Some options
    overwrite_existing_results = True
    dry_run = False

    # dependencies
    job_number = 49755164  # start only after job_number has finished. Put 0 to start immediately
    dependencies = True  # if true, start a smiluation only if previous one has finished

    # Some options that usually we do not change
    o_freq = 0
    log_level = 30

    # In general we do not touch frow here on
    # ------------------------------------------------------------------------------

    tot_sim = len(iter_indeces) * (
        count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, splittings, dts) * (1 if "IMEXEXP" in integrators else 0)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, dts) * (1 if "mES" in integrators else 0)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, splittings, dts)
        * (len(integrators) - (1 if "IMEXEXP" in integrators else 0) - (1 if "mES" in integrators else 0))
    )
    print(f"About to launch {tot_sim} simulations.")

    simulations_counter = 0
    total_runtime_estimated = 0

    for iter_index in iter_indeces:
        if "IMEXEXP" in integrators:
            es_class = "none"
            safe_add = 0
            integrator = "IMEXEXP"
            for domain_name in domains_names:
                for fibrosis in fibrosiss:
                    for refinement in refinements:
                        for ionic_model in ionic_models:
                            for splitting in splittings:
                                for k, dt in zip(ks, dts):
                                    print(f"\nSimulation {simulations_counter+1} of {tot_sim}")
                                    ntasks, ntaskspernode, steps_per_sec = get_slurm_settings(domain_name, refinement)
                                    output_file_name = "IMEXEXP_k_" + str(k) + "_ntasks_" + str(ntasks) + "_read_init_val_" + str(read_init_val) + "_prof_" + str(iter_index)
                                    options = set_options(
                                        integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level
                                    )
                                    options["--output_root"] = options["--output_root"][:-1] + "_profiling/"
                                    dir_and_output_file_name = (
                                        options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--refinements"]) + "/" + options["--ionic_model"] + "/" + output_file_name
                                    )
                                    base_python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                    runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                    total_runtime_estimated += runtime
                                    if runtime > 0:
                                        simulations_counter += 1

        if "mES" in integrators:
            integrator = "mES"
            splitting = "stiff_nonstiff"
            for domain_name in domains_names:
                for fibrosis in fibrosiss:
                    for refinement in refinements:
                        for ionic_model in ionic_models:
                            for safe_add in safe_adds:
                                for es_class in es_classes:
                                    for k, dt in zip(ks, dts):
                                        print(f"\nSimulation {simulations_counter+1} of {tot_sim}")
                                        ntasks, ntaskspernode, steps_per_sec = get_slurm_settings(domain_name, refinement)
                                        output_file_name = (
                                            "mES_"
                                            + es_class
                                            + "_safe_add_"
                                            + str(safe_add)
                                            + "_k_"
                                            + str(k)
                                            + "_ntasks_"
                                            + str(ntasks)
                                            + "_read_init_val_"
                                            + str(read_init_val)
                                            + "_prof_"
                                            + str(iter_index)
                                        )
                                        options = set_options(
                                            integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level
                                        )
                                        options["--output_root"] = options["--output_root"][:-1] + "_profiling/"
                                        dir_and_output_file_name = (
                                            options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--refinements"]) + "/" + options["--ionic_model"] + "/" + output_file_name
                                        )
                                        base_python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                        runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                        total_runtime_estimated += runtime
                                        if runtime > 0:
                                            simulations_counter += 1

        if "exp_mES" in integrators:
            integrator = "exp_mES"
            for domain_name in domains_names:
                for fibrosis in fibrosiss:
                    for refinement in refinements:
                        for ionic_model in ionic_models:
                            for safe_add in safe_adds:
                                for es_class in es_classes:
                                    for splitting in splittings:
                                        for k, dt in zip(ks, dts):
                                            print(f"\nSimulation {simulations_counter+1} of {tot_sim}")
                                            ntasks, ntaskspernode, steps_per_sec = get_slurm_settings(domain_name, refinement)
                                            output_file_name = (
                                                integrator
                                                + "_"
                                                + es_class
                                                + "_safe_add_"
                                                + str(safe_add)
                                                + "_k_"
                                                + str(k)
                                                + "_ntasks_"
                                                + str(ntasks)
                                                + "_read_init_val_"
                                                + str(read_init_val)
                                                + "_prof_"
                                                + str(iter_index)
                                            )
                                            options = set_options(
                                                integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level
                                            )
                                            options["--output_root"] = options["--output_root"][:-1] + "_profiling/"
                                            dir_and_output_file_name = (
                                                options["--output_root"]
                                                + options["--domain_name"]
                                                + "/"
                                                + "ref_"
                                                + str(options["--refinements"])
                                                + "/"
                                                + options["--ionic_model"]
                                                + "/"
                                                + output_file_name
                                            )
                                            base_python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                            runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                            total_runtime_estimated += runtime
                                            if runtime > 0:
                                                simulations_counter += 1

    print(f"\nLaunched {simulations_counter} simulations, forecast was {tot_sim} simulations.")
    print(f"Total runtime estimated: {total_runtime_estimated}")


if __name__ == "__main__":
    main()
