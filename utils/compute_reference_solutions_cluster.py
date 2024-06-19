from run_simulations_cluster import set_options, execute_with_dependencies


def main():
    # for 03_fastl_LA we have all ionic models and no fibrosis, for fibrosis only CRN and TTP

    # Compute reference solutions for multiple problems
    domains_names = ["03_fastl_LA"]
    refinements = [2]
    fibrosiss = [True]
    ionic_models = ["HH"]  # ["HH", "CRN", "TTP"]

    # parameters used to compute the solution
    dt = 1e-4
    integrator = "exp_mES"
    es_class = "RKW1"
    splitting = "exp_nonstiff"
    safe_add = 0

    # read an initial value or use the hardcoded one?
    read_init_val = True

    # This data is used to estimate the running time, used in the slurm job submission
    tend = 50.0

    # Some options
    overwrite_existing_results = False
    dry_run = False

    # Other options, rarely changed
    output_file_name = "ref_sol"
    o_freq = -1
    log_level = 30

    job_number = 0
    dependencies = True

    base_python_command = "python3 Solve_Monodomain.py"

    tot_sim = len(domains_names) * len(refinements) * len(ionic_models) * len(fibrosiss)
    sim_count = 0
    total_runtime_estimated = 0

    for domain_name in domains_names:
        for fibrosis in fibrosiss:
            for refinement in refinements:
                for ionic_model in ionic_models:
                    options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level)
                    sim_count += 1
                    print(f"\nComputing reference solution {sim_count} of {tot_sim}")
                    runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                    total_runtime_estimated += runtime

    print(f"\nTotal runtime estimated: {total_runtime_estimated}")


if __name__ == "__main__":
    main()
