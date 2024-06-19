import numpy as np

from run_simulations import execute, count_number_simulation, set_options


def main():
    # Define problems
    domains_names = ["03_fastl_LA"]
    refinements = [2]
    ionic_models = ["HH", "CRN", "TTP"]
    fibrosiss = [False]

    # Define integration methods
    es_classes = ["RKC1"]  # "RKW1"
    integrators = ["mES", "exp_mES", "IMEXEXP"]
    splittings = ["exp_nonstiff"]  # useless since now is hardcoded depending on the integrator. I should remove this option...
    safe_adds = [0]

    # Define step sizes
    base_dt = 1.0
    min_k = 3
    max_k = 3
    ks = np.array(list(range(min_k, max_k + 1)))
    dts = base_dt * 2.0 ** (-ks)

    # Some options
    overwrite_existing_results = False
    dry_run = False
    n_proc = 12

    # read an initial value or use the hardcoded one?
    read_init_vals = [False]

    # From here on usually we do not touch anything
    # ---------------------------------------------------------------------------------------------------------------------
    o_freq = 0
    base_docker_command = "docker exec -ti -w /src/Stabilized_integrators_FeNICSx my_dolfinx_daint_container mpirun -n " + str(n_proc) + " "

    tot_sim = (
        count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, splittings, dts, read_init_vals)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, dts, read_init_vals)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, splittings, dts, read_init_vals)
        * (len(integrators) - (1 if "IMEXEXP" in integrators else 0) - (1 if "mES" in integrators else 0))
    )
    print(f"Launching {tot_sim} simulations.")

    simulations_counter = 0

    if "IMEXEXP" in integrators:
        es_class = "none"
        safe_add = 0
        integrator = "IMEXEXP"
        for read_init_val in read_init_vals:
            for domain_name in domains_names:
                for fibrosis in fibrosiss:
                    for refinement in refinements:
                        for ionic_model in ionic_models:
                            for splitting in splittings:
                                for k, dt in zip(ks, dts):
                                    print(f"Running simulation {simulations_counter+1} of {tot_sim}")
                                    output_file_name = "IMEXEXP_k_" + str(k) + "_read_init_val_" + str(read_init_val)
                                    options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val)
                                    options["--output_root"] = options["--output_root"][:-1] + "_profiling/"
                                    dir_and_output_file_name = (
                                        options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--refinements"]) + "/" + options["--ionic_model"] + "/" + output_file_name
                                    )
                                    python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                    base_command = base_docker_command + python_command
                                    executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                    if executed:
                                        simulations_counter += 1
        integrators.remove("IMEXEXP")

    if "mES" in integrators:
        integrator = "mES"
        splitting = "stiff_nonstiff"
        for read_init_val in read_init_vals:
            for domain_name in domains_names:
                for fibrosis in fibrosiss:
                    for refinement in refinements:
                        for ionic_model in ionic_models:
                            for safe_add in safe_adds:
                                for es_class in es_classes:
                                    for k, dt in zip(ks, dts):
                                        print(f"Running simulation {simulations_counter+1} of {tot_sim}")
                                        output_file_name = "mES_" + es_class + "_safe_add_" + str(safe_add) + "_k_" + str(k) + "_read_init_val_" + str(read_init_val)
                                        options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val)
                                        options["--output_root"] = options["--output_root"][:-1] + "_profiling/"
                                        dir_and_output_file_name = (
                                            options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--refinements"]) + "/" + options["--ionic_model"] + "/" + output_file_name
                                        )
                                        python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                        base_command = base_docker_command + python_command
                                        executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                        if executed:
                                            simulations_counter += 1
        integrators.remove("mES")

    for read_init_val in read_init_vals:
        for domain_name in domains_names:
            for fibrosis in fibrosiss:
                for refinement in refinements:
                    for ionic_model in ionic_models:
                        for safe_add in safe_adds:
                            for es_class in es_classes:
                                for integrator in integrators:
                                    for splitting in splittings:
                                        for k, dt in zip(ks, dts):
                                            print(f"Running simulation {simulations_counter+1} of {tot_sim}")
                                            output_file_name = integrator + "_" + es_class + "_safe_add_" + str(safe_add) + "_k_" + str(k) + "_read_init_val_" + str(read_init_val)
                                            options = set_options(
                                                integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val
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
                                            python_command = f"python3 -m cProfile -o {dir_and_output_file_name}.prof Solve_Monodomain.py"
                                            base_command = base_docker_command + python_command
                                            executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                            if executed:
                                                simulations_counter += 1

    print(f"Launched {simulations_counter} simulations, forecast was {tot_sim} simulations.")


if __name__ == "__main__":
    main()
