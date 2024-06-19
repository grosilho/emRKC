import os
import numpy as np


def count_number_simulation(*lst_lst):
    p = 1
    for lst in lst_lst:
        p *= len(lst)
    return p


def set_options(integrator, ionic_model, dt, space_disc, order, es_class, safe_add, fibrosis, refinements, domain_name, o_freq, output_file_name, read_init_val, end_time):
    options = dict()
    options["--output_root"] = "./results_monodomain/" if not fibrosis else "./results_monodomain_fibrosis/"
    options["--integrator"] = integrator
    options["--ionic_model"] = ionic_model
    options["--dt"] = dt
    options["--end_time"] = end_time
    options["--space_disc"] = space_disc
    options["--order"] = order
    options["--output_file_name"] = output_file_name
    options["--es_class"] = es_class
    options["--safe_add"] = safe_add
    if fibrosis:
        options["--fibrosis"] = ""
    options["--pre_refinements"] = refinements
    options["--domain_name"] = domain_name
    options["--o_freq"] = o_freq
    if read_init_val:
        options["--read_init_val"] = ""
    options["--istim_dur"] = 0.0 if read_init_val else -1.0

    return options


def options_command(options):
    cmd = ""
    for key, val in options.items():
        cmd = cmd + " " + key + " " + str(val)
    return cmd


def execute(base_command, options, dry_run, overwrite):
    opts = options_command(options)
    command = base_command + opts

    base_dir = options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--pre_refinements"]) + "/" + options["--ionic_model"] + "/"
    output_file = base_dir + options["--output_file_name"] + ".db"

    file_exists = True if os.path.isfile(output_file) else False

    executed = False
    if overwrite or not file_exists:
        print(f"Options: {opts}")
        executed = True
        if not dry_run:
            os.system(command)

    return executed


def main():
    # Define problems
    domains_names = ["cuboid_2D"]
    refinements = [0, 1, 2]
    ionic_models = ["TTP", "CRN", "HH"]
    fibrosiss = [False, True]

    # Define integration methods
    es_classes = ["RKC1"]  # "RKW1"
    integrators = ["mES", "exp_mES", "IMEXEXP"]  # ,'exp_mES_prog'
    splittings = ["exp_nonstiff"]  # useless since now is hardcoded depending on the integrator. I should remove this option...
    safe_adds = [0]

    # Define step sizes
    base_dt = 1.0
    min_k = 0
    max_k = 9
    ks = np.array(list(range(min_k, max_k + 1)))
    dts = base_dt * 2.0 ** (-ks)

    # Some options
    overwrite_existing_results = True
    dry_run = False
    n_proc = 12

    # read an initial value or use the hardcoded one?
    read_init_val = False

    # From here on usually we do not touch anything
    # ---------------------------------------------------------------------------------------------------------------------
    o_freq = 0
    base_command = "docker exec -ti -w /src/Stabilized_integrators_FeNICSx my_dolfinx_daint_container mpirun -n " + str(n_proc) + " python3 Solve_Monodomain.py"

    tot_sim = (
        count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, splittings, dts)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, dts)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, splittings, dts)
        * (len(integrators) - (1 if "IMEXEXP" in integrators else 0) - (1 if "mES" in integrators else 0))
    )
    print(f"Launching {tot_sim} simulations.")

    simulations_counter = 0

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
                                print(f"Running simulation {simulations_counter+1} of {tot_sim}")
                                output_file_name = "IMEXEXP_splitting_" + splitting + "_k_" + str(k)
                                options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val)
                                executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                if executed:
                                    simulations_counter += 1
        integrators.remove("IMEXEXP")

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
                                    print(f"Running simulation {simulations_counter+1} of {tot_sim}")
                                    output_file_name = "mES_" + es_class + "_safe_add_" + str(safe_add) + "_splitting_stiff_nonstiff_k_" + str(k)
                                    options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val)
                                    executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                    if executed:
                                        simulations_counter += 1
        integrators.remove("mES")

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
                                        output_file_name = integrator + "_" + es_class + "_safe_add_" + str(safe_add) + "_splitting_" + splitting + "_k_" + str(k)
                                        options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val)
                                        executed = execute(base_command, options, dry_run, overwrite_existing_results)
                                        if executed:
                                            simulations_counter += 1

    print(f"Launched {simulations_counter} simulations, forecast was {tot_sim} simulations.")


if __name__ == "__main__":
    main()
