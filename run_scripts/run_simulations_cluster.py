import os
import numpy as np
import subprocess
import re


def count_number_simulation(*lst_lst):
    p = 1
    for lst in lst_lst:
        p *= len(lst)
    return p


def set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinements, domain_name, o_freq, output_file_name, read_init_val, log_level):
    options = dict()
    options["--output_root"] = "./results_monodomain/" if not fibrosis else "./results_monodomain_fibrosis/"
    options["--integrator"] = integrator
    options["--ionic_model"] = ionic_model
    options["--dt"] = dt
    options["--splitting"] = splitting
    options["--output_file_name"] = output_file_name
    options["--es_class"] = es_class
    options["--safe_add"] = safe_add
    options["--log_level"] = log_level
    if fibrosis:
        options["--fibrosis"] = ""
    options["--refinements"] = refinements
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


def estimate_run_time(tend, dt, steps_per_sec, integrator):
    seconds = tend / dt / steps_per_sec
    fac = 1.5 if "exp_mES" in integrator else 5.0
    seconds = max(fac * seconds + 600, 1200)
    hours = int(np.floor(seconds / 3600.0))
    minutes = int(np.ceil(60 * (seconds / 3600.0 - hours)))
    hours_str = ("0" if hours < 10 else "") + str(hours)
    minutes_str = ("0" if minutes < 10 else "") + str(minutes)
    time_str = hours_str + ":" + minutes_str + ":" + "00"
    hours_exact = seconds / 3600.0
    return time_str, hours_exact


def execute_with_dependencies(base_python_command, options, dry_run, overwrite, tend, job_number, dependencies):
    base_dir = options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--refinements"]) + "/" + options["--ionic_model"] + "/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    output_file = base_dir + options["--output_file_name"] + ".db"
    file_exists = os.path.isfile(output_file)
    if overwrite or not file_exists:
        ntasks, ntaskspernode, steps_per_sec = get_slurm_settings(options["--domain_name"], options["--refinements"])
        time_str, hours_exact = estimate_run_time(tend, options["--dt"], steps_per_sec, options["--integrator"])
        print(f"Slurm options: ntasks = {ntasks}, ntaskspernode = {ntaskspernode}, steps_per_sec = {steps_per_sec}, time = {time_str}")
        opts = options_command(options)
        print(f"Simulation options: {opts}")
        log_file = base_dir + options["--output_file_name"] + ".log"
        dependency_str = "" if (job_number == 0 or not dependencies) else f"\n#SBATCH --dependency=afterany:{job_number}"
        prev_job_number = job_number
        script = f'#!/bin/bash -l\
                    \n#SBATCH --job-name="myjob"\
                    \n#SBATCH --account="s1074"\
                    \n#SBATCH --time={time_str}\
                    \n#SBATCH --ntasks={ntasks}\
                    \n#SBATCH --ntasks-per-node={ntaskspernode}\
                    \n#SBATCH --output={log_file}\
                    \n#SBATCH --cpus-per-task=1\
                    \n#SBATCH --ntasks-per-core=1\
                    \n#SBATCH --constraint=gpu\
                    \n#SBATCH --hint=nomultithread\
                    {dependency_str}\
                    \nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\
                    \nexport CRAY_CUDA_MPS=1\
                    \nsrun {base_python_command+opts}\
                    \n'

        if not dry_run:
            with open("script.sh", "w") as file:
                file.write(script)
            res = subprocess.check_output("sbatch script.sh", shell=True)  # output similar to: Submitted batch job 48953310
            match = re.search(r"\d+", res.decode())
            if match:
                job_number = int(match.group())
            else:
                raise Exception("Could not find the job number")
            os.system("rm script.sh")
            print(f"Submitted batch job {job_number}, runtime estimated {time_str}" + (f", dependent on job {prev_job_number}" if prev_job_number != 0 else ""))

        return hours_exact, job_number
    else:
        return 0.0, job_number


def get_slurm_settings(domain_name, refinement):
    if domain_name == "cuboid_3D":
        if refinement == 0:
            ntasks = 24
            steps_per_sec = 100
        elif refinement == 1:
            ntasks = 48
            steps_per_sec = 25
        elif refinement == 2:
            ntasks = 120
            steps_per_sec = 7
    elif domain_name == "03_fastl_LA":
        if refinement == 2:
            ntasks = 240
            steps_per_sec = 32
            # ntasks = 12
            # steps_per_sec = 0.2

    max_ntaskspernode = 12
    ntaskspernode = min(max_ntaskspernode, ntasks)

    return ntasks, ntaskspernode, steps_per_sec


def main():
    # Problem description
    domains_names = ["cuboid_3D"]
    refinements = [2]
    fibrosiss = [False]
    ionic_models = ["HH", "CRN", "TTP"]

    # Numerical methods
    es_classes = ["RKC1"]
    integrators = ["mES", "exp_mES", "IMEXEXP"]  # , "exp_mES_prog"]
    splittings = ["exp_nonstiff"]  # ,"exp_stiff_nonstiff"]
    safe_adds = [0]

    # read an initial value or use the hardcoded one?
    read_init_val = False

    # Step sizes
    base_dt = 1.0
    min_k = 0
    max_k = 9
    ks = np.array(list(range(min_k, max_k + 1)))
    dts = base_dt * 2.0 ** (-ks)

    # This data is used to estimate the running time, used in the slurm job submission
    tend = 25.0

    # Some options
    overwrite_existing_results = False
    dry_run = False

    # dependencies
    job_number = 0  # start only after job_number has finished. Put 0 to start immediately
    dependencies = True  # if true, start a smiluation only if previous one has finished

    # Some options that usually we do not change
    o_freq = 0
    log_level = 40

    # In general we do not touch frow here on
    # ------------------------------------------------------------------------------
    base_python_command = "python3 Solve_Monodomain.py"

    tot_sim = (
        count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, splittings, dts) * (1 if "IMEXEXP" in integrators else 0)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, dts) * (1 if "mES" in integrators else 0)
        + count_number_simulation(domains_names, fibrosiss, refinements, ionic_models, safe_adds, es_classes, splittings, dts)
        * (len(integrators) - (1 if "IMEXEXP" in integrators else 0) - (1 if "mES" in integrators else 0))
    )
    print(f"About to launch {tot_sim} simulations.")

    simulations_counter = 0
    total_runtime_estimated = 0

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
                                output_file_name = "IMEXEXP_splitting_" + splitting + "_k_" + str(k)
                                options = set_options(integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level)
                                runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                total_runtime_estimated += runtime
                                if runtime > 0:
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
                                    print(f"\nSimulation {simulations_counter+1} of {tot_sim}")
                                    output_file_name = "mES_" + es_class + "_safe_add_" + str(safe_add) + "_splitting_stiff_nonstiff_k_" + str(k)
                                    options = set_options(
                                        integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level
                                    )
                                    runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                    total_runtime_estimated += runtime
                                    if runtime > 0:
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
                                        print(f"\nSimulation {simulations_counter+1} of {tot_sim}")
                                        output_file_name = integrator + "_" + es_class + "_safe_add_" + str(safe_add) + "_splitting_" + splitting + "_k_" + str(k)
                                        options = set_options(
                                            integrator, ionic_model, dt, splitting, es_class, safe_add, fibrosis, refinement, domain_name, o_freq, output_file_name, read_init_val, log_level
                                        )
                                        runtime, job_number = execute_with_dependencies(base_python_command, options, dry_run, overwrite_existing_results, tend, job_number, dependencies)
                                        total_runtime_estimated += runtime
                                        if runtime > 0:
                                            simulations_counter += 1

    print(f"\nLaunched {simulations_counter} simulations, forecast was {tot_sim} simulations.")
    print(f"Total runtime estimated: {total_runtime_estimated}")


if __name__ == "__main__":
    main()
