from run_simulations import set_options, execute
import numpy as np


def main():
    # Compute reference solutions for multiple problems
    space_disc = "FD"
    order = 2
    domains_names = ["cuboid_1D_small"]
    refinements = [2]
    ionic_models = ["TTP"]  # ["BS", "TTP", "CRN", "HH"]
    fibrosiss = [False]

    # parameters used to compute the solutions
    dt = 1e-5
    end_time = 10.0
    integrator = "exp_mES"
    es_class = "RKC1"
    safe_add = 0

    # Some options
    dry_run = False
    overwrite_existing_results = True

    # Some rarely changed options
    output_file_name = "ref_sol_end_time_" + str(np.round(end_time).astype(int))
    o_freq = -1
    n_proc = 1

    # read an initial value or use the hardcoded one?
    read_init_val = True

    # We dont change anything from here on
    # -------------------------------------------------------------------------------------------------------------------------
    if space_disc == "FEM":
        base_command = "docker exec -ti -w /src/Stabilized_integrators_FeNICSx my_dolfinx_daint_container mpirun -n " + str(n_proc) + " python3 Solve_Monodomain.py"
    else:
        base_command = "python Solve_Monodomain.py"

    tot_sim = len(domains_names) * len(refinements) * len(ionic_models) * len(fibrosiss)
    sim_count = 0

    for domain_name in domains_names:
        for fibrosis in fibrosiss:
            for refinement in refinements:
                for ionic_model in ionic_models:
                    options = set_options(integrator, ionic_model, dt, space_disc, order, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, end_time)
                    sim_count += 1
                    print(f"Computing reference solution {sim_count} of {tot_sim}")
                    execute(base_command, options, dry_run, overwrite_existing_results)


if __name__ == "__main__":
    main()
