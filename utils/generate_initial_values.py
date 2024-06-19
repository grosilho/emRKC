import os
import numpy as np


def set_options(integrator, ionic_model, space_disc, dt, fibrosis, pre_refinements, domain_name, end_time, order):
    options = dict()
    options["--output_root"] = "./results_monodomain/" if not fibrosis else "./results_monodomain_fibrosis/"
    options["--integrator"] = integrator
    options["--ionic_model"] = ionic_model
    options["--space_disc"] = space_disc
    options["--dt"] = dt
    options["--end_time"] = end_time
    if fibrosis:
        options["--fibrosis"] = ""
    options["--pre_refinements"] = pre_refinements
    options["--domain_name"] = domain_name
    options["--o_freq"] = -2
    options["--order"] = order
    options["--output_file_name"] = "init_val_" + space_disc

    return options


def options_command(options):
    cmd = ""
    for key, val in options.items():
        cmd = cmd + " " + key + " " + str(val)
    return cmd


def execute(base_command, options, dry_run):
    opts = options_command(options)
    command = base_command + opts

    print(f"Options: {opts}")
    if not dry_run:
        os.system(command)


def main():
    # Define problems
    domains_names = ["cube_1D"]
    pre_refinements = [0, 1, 2]  # , 3, 4]
    ionic_models = ["TTP", "CRN", "HH", "BS"]
    space_discs = ["DCT"]
    end_time = 6.0
    order = 4

    fibrosis = False
    integrator = "IMEXEXP"
    dt = 0.01

    # Some options
    dry_run = False

    # From here on usually we do not touch anything
    # ---------------------------------------------------------------------------------------------------------------------
    n_proc = 1
    # mac:  my_dolfinx_daint_monodomain
    # linux: my_dolfinx_daint_container_monodomain_new
    base_command = (
        "docker exec -w /src/Stabilized_integrators -it my_dolfinx_daint_monodomain mpirun -n "
        + str(n_proc)
        + " python3 Solve_Monodomain.py"
    )

    for space_disc in space_discs:
        for domain_name in domains_names:
            for pre_refinement in pre_refinements:
                for ionic_model in ionic_models:
                    options = set_options(
                        integrator, ionic_model, space_disc, dt, fibrosis, pre_refinement, domain_name, end_time, order
                    )
                    execute(base_command, options, dry_run)


if __name__ == "__main__":
    main()
