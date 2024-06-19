import os
import numpy as np
import subprocess
import re


def count_number_simulation(*lst_lst):
    p = 1
    for lst in lst_lst:
        p *= len(lst)
    return p


def set_options(integrator, ionic_model_name, dt, fibrosis, pre_refinements, post_refinements, domain_name, output_file_name, log_level):
    options = dict()
    options["--output_root"] = "./results_scalability/" if not fibrosis else "./results_scalability_fibrosis/"
    options["--integrator"] = integrator
    options["--ionic_model_name"] = ionic_model_name
    options["--dt"] = dt
    options["--output_file_name"] = output_file_name
    options["--log_level"] = log_level
    if fibrosis:
        options["--fibrosis"] = ""
    options["--pre_refinements"] = pre_refinements
    options["--post_refinements"] = post_refinements
    options["--domain_name"] = domain_name

    return options


def options_command(options):
    cmd = ""
    for key, val in options.items():
        cmd = cmd + " " + key + " " + str(val)
    return cmd


def execute_with_dependencies(base_python_command, ntasks, time_str, options, dry_run, overwrite, job_number, dependencies):
    base_dir = options["--output_root"] + options["--domain_name"] + "/" + "ref_" + str(options["--pre_refinements"]) + "/" + options["--ionic_model_name"] + "/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    output_file = base_dir + options["--output_file_name"] + ".db"
    file_exists = os.path.isfile(output_file)
    if overwrite or not file_exists:
        ntaskspernode = 12
        print(f"Slurm options: ntasks = {ntasks}, ntaskspernode = {ntaskspernode}, time = {time_str}")
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

        return job_number
    else:
        return job_number


def main():
    # Problem description
    # domain_name = "cuboid_3D"
    # domain_name = "01_strocchi_LV"
    domain_name = "03_fastl_LA"
    pre_refinements = 2
    post_refinements = 0
    fibrosis = False
    ionic_model_name = "CRN"

    # N processors
    time_str = "05:00:00"
    # ntasks = [12, 24, 48]  # [128, 256]  # , 512, 1024, 2048]
    # ntasks = [256, 512, 1024, 2048]
    # ntasks = [120, 240, 480, 960, 1920]
    # ntasks = [16, 32, 64, 128, 256, 512, 768, 1024, 2048]
    ntasks = [2048]

    # Numerical methods
    # integrator = "exp_mES"
    integrator = "IMEXEXP"

    # Step size
    dt = 0.1

    # Some options
    overwrite_existing_results = True
    dry_run = False

    # dependencies
    job_number = 0  # start only after job_number has finished. Put 0 to start immediately
    dependencies = True  # if true, start a smiluation only if previous one has finished

    # In general we do not touch frow here on
    # ------------------------------------------------------------------------------
    base_python_command = "python3 Solve_Monodomain_New.py"
    log_level = 30

    for ntask in ntasks:
        output_file_name = f"{integrator}_post_ref_{post_refinements}_n_tasks_{ntask}"
        options = set_options(integrator, ionic_model_name, dt, fibrosis, pre_refinements, post_refinements, domain_name, output_file_name, log_level)
        job_number = execute_with_dependencies(base_python_command, ntask, time_str, options, dry_run, overwrite_existing_results, job_number, dependencies)


if __name__ == "__main__":
    main()
