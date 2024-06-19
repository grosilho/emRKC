import logging
import argparse
import os
from pathlib import Path

import sys

sys.path.insert(0, '..')

from utils.data_management import database

from problem_classes.MonodomainODE import MonodomainODE, MultiscaleMonodomainODE

from integrators.TimeIntegrator import TimeIntegrator
from integrators.standard import explicit_Euler, exponential_explicit_implicit_Euler
from integrators.ExplicitStabilized import (
    explicit_stabilized,
    multirate_explicit_stabilized,
    exponential_multirate_explicit_stabilized,
    exponential_multirate_explicit_stabilized_progress,
)


def get_default_problem_params():
    problem_params = dict()
    problem_params["space_disc"] = "DCT"
    problem_params["order"] = 2
    problem_params["mass_lumping"] = True  # has effect only for space_disc = FEM with order = 1
    problem_params["pre_refinements"] = 0
    problem_params["post_refinements"] = 0
    problem_params["fibrosis"] = False
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    problem_params["meshes_fibers_root_folder"] = executed_file_dir + "/../meshes_fibers_fibrosis/results"
    problem_params["domain_name"] = "cuboid_2D_small"
    problem_params["init_time"] = 0.0
    problem_params["end_time"] = -1.0
    problem_params["apply_stimulus"] = True
    problem_params["lin_solv_max_iter"] = None
    problem_params["lin_solv_rtol"] = None
    problem_params["ionic_model_name"] = "HH"
    problem_params["read_init_val"] = False
    problem_params["init_val_name"] = "init_val"
    problem_params["enable_output"] = False
    problem_params["output_V_only"] = True
    problem_params["output_root"] = executed_file_dir + "results/"
    problem_params["output_file_name"] = "monodomain"
    problem_params["ref_sol"] = "none"

    return problem_params


def get_default_integrator_params():
    int_params = dict()
    int_params["o_freq"] = 1
    int_params["dt"] = 0.1
    int_params["integrator"] = "emES"
    int_params["log_level"] = 30

    int_params["rho_freq"] = 1e30
    int_params["damping"] = 0.05
    int_params["safe_add"] = 0
    int_params["ES_class"] = "RKC"
    int_params["ES_s_outer"] = 0
    int_params["ES_m_inner"] = 0

    return int_params


def get_problem_and_stepper(problem_params, int_params):
    if int_params["integrator"] == "EE":
        problem = MonodomainODE(**problem_params)
        EE = explicit_Euler(int_params, problem)
        stepper = EE
    elif int_params["integrator"] == "IMEX-RL":
        problem_params["splitting"] = "fast_slow_exponential"
        problem = MultiscaleMonodomainODE(**problem_params)
        IMEXEXP = exponential_explicit_implicit_Euler(int_params, problem)
        stepper = IMEXEXP
    elif int_params["integrator"] == "ES":
        problem = MonodomainODE(**problem_params)
        ES = explicit_stabilized(int_params, problem)
        stepper = ES
    elif int_params["integrator"] == "mES":
        problem_params["splitting"] = "fast_slow"
        problem = MultiscaleMonodomainODE(**problem_params)
        ES = multirate_explicit_stabilized(int_params, problem)
        stepper = ES
    elif int_params["integrator"] == "emES":
        problem_params["splitting"] = "fast_slow_exponential"
        problem = MultiscaleMonodomainODE(**problem_params)
        ES = exponential_multirate_explicit_stabilized(int_params, problem)
        stepper = ES
    elif int_params["integrator"] == "emES_prog":
        problem_params["splitting"] = "fast_slow_exponential"
        problem = MultiscaleMonodomainODE(**problem_params)
        ES = exponential_multirate_explicit_stabilized_progress(int_params, problem)
        stepper = ES
    else:
        raise Exception("Unknown integrator.")

    return problem, stepper


def modify_params(params, args):
    for key, val in args.items():
        if key in params:
            params[key] = val
    return params


def main():
    problem_params = get_default_problem_params()
    int_params = get_default_integrator_params()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # integrator args
    parser.add_argument(
        "--integrator",
        default="emES",
        type=str,
        help="The integration scheme: IMEX-RL, EE, ES, mES, emES. IMEX-RL is the IMEX Rush-Larsen method,\
              EE the Explicit Euler and ES stands for Explicit Stabilized. The class of ES method (e.g. RKC)\
              is defined by the ES_class parameter. mES stands for multirate Explicit Stabilized. emES stands for exponential multirate Explicit Stabilized.",
    )
    parser.add_argument("--dt", default=0.1, type=float, help="The step size.")
    parser.add_argument(
        "--end_time",
        default=0.0,
        type=float,
        help="The end time in ms. If zero or not provided, the default 50ms is used. Note that initial time is set at t0=0ms",
    )
    parser.add_argument(
        "--apply_stimulus",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply or not the stimulus protool. If false, an interesting initial value should be readen from disk, otherwise you are going to simulate zeros.",
    )
    parser.add_argument(
        "--ES_class",
        default="RKC",
        type=str,
        help="Explicit Stabilized method: RKC, RKU, RKW, HSRKU. Only valid for ES, mES and emES integrators.",
    )
    parser.add_argument(
        "--ES_s_outer",
        default=0,
        type=int,
        help="If >0, fix the outer number of stages s to that value. If 0, the number of stages is computed automatically.",
    )
    parser.add_argument(
        "--ES_m_inner",
        default=0,
        type=int,
        help="If >0 and the outer stages s are fixed, fixes the inner number of stages m. If 0 or outer s are not fixed, the number of stages m is computed automatically.",
    )
    parser.add_argument(
        "--o_freq",
        default=0,
        type=int,
        help="Output frequency (write sol. to disk), every o_freq steps. \
            If -1 then saves final solution as reference solution for error computations, if -2 saves it to be used as initial value for another simulation.",
    )
    parser.add_argument(
        "--rho_freq",
        default=1e30,
        type=int,
        help="Spectral radii are updated every rho_freq steps. Use a large value to compute it only once at the beginning of the simulation.",
    )
    parser.add_argument(
        "--safe_add",
        default=0,
        type=int,
        help="Add safe_add additional stages to outer stages s, for robustness regarding rapidly growing spectral radii. Usually 0,1 or 2 at most.",
    )
    parser.add_argument("--log_level", default=20, type=int, help="Logging level: 10 debug, 20 info, 30 warning.")
    # problem args
    parser.add_argument(
        "--space_disc",
        default="FEM",
        type=str,
        help="Space discretization method: FEM (finite element method) or DCT (discrete cosine transform).",
    )
    parser.add_argument(
        "--enable_output",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Activate or deactivate xdmf output: True or False",
    )
    parser.add_argument("--output_file_name", default="monodomain", type=str, help="Output file name.")
    parser.add_argument(
        "--output_root",
        default="results/",
        type=str,
        help="Output root folder, relative to the location of this file.",
    )
    parser.add_argument(
        "--ref_sol",
        default="ref_sol",
        type=str,
        help="Name of reference solution. If it is found, it will be used to compute errors at the end.",
    )
    parser.add_argument(
        "--ionic_model_name",
        default="TTP",
        type=str,
        help="Ionic_model: HH (Hodgkin-Huxley 1952), CRN (Courtemanche-Ramirez-Nattel 1998), \
            TTP (tenTusscher-Panfilov 2006), TTP_SMOOTH (a smoothed version of TTP), BS (the bistable, or Nagumo, model)",
    )
    parser.add_argument(
        "--domain_name",
        default="cuboid_2D_small",
        type=str,
        help="Domain name: cuboid_2D, cuboid_3D, truncated_ellipsoid,.... See codes for all avalable options.",
    )
    parser.add_argument(
        "--read_init_val",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Read the initial value from file.",
    )
    parser.add_argument(
        "--init_val_name",
        default="init_val",
        type=str,
        help="Name of the initial value to be readen, when read_initial_value is true.",
    )

    parser.add_argument(
        "--pre_refinements",
        default=0,
        type=int,
        help="When space_disc=FEM, loads a mesh which has already been pre-refined pre_refinements times.\
              When space_disc=DCT, builds a uniform mesh with mesh size 2**(pre_refinements) smaller than a baseline.",
    )
    parser.add_argument(
        "--post_refinements",
        default=0,
        type=int,
        help="When space_disc=FEM, refines the mesh post_refinements times, after loading it. No effect when space_disc=DCT.",
    )
    parser.add_argument(
        "--fibrosis",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="With or without fibrosis, available only for space_disc=FEM.",
    )
    parser.add_argument(
        "--order",
        default=1,
        type=int,
        help="Order of FEM or DCT discretization. Any order for FEM, orders 2 and 4 for DCT.",
    )
    parser.add_argument(
        "--mass_lumping",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="When using FEm and order is 1, enables of disables mass lumping.",
    )

    args = vars(parser.parse_args())

    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    args["output_root"] = executed_file_dir + "/../" + args["output_root"]

    problem_params = modify_params(problem_params, args)
    int_params = modify_params(int_params, args)

    run_simulation(problem_params, int_params)


def run_simulation(problem_params, int_params):
    dt = int_params["dt"]
    o_freq = int_params["o_freq"]

    logger_level = int_params["log_level"]
    logging.basicConfig(level=logger_level)
    logger_rho = logging.getLogger("rho_estimator")
    logger_rho.setLevel(logger_level)
    logger_step = logging.getLogger("step")
    logger_step.setLevel(logger_level)
    logger_stat = logging.getLogger("stat")
    logger_stat.setLevel(logger_level)
    show_bar = logger_level == logging.WARNING
    logger = {"log_rho": logger_rho, "log_step": logger_step, "log_stat": logger_stat, "show_bar": show_bar}

    problem, stepper = get_problem_and_stepper(problem_params, int_params)

    step = stepper.step
    time_int = TimeIntegrator(problem, step, logger)
    t_end, u_end, cpu_time, CV = time_int.solve(dt, o_freq)

    step_stats = stepper.get_stats()

    del stepper

    error_availabe, error_L2, rel_error_L2 = problem.compute_errors(u_end)

    sim_data = dict()
    sol_data = dict()

    tot_dofs = u_end.getSize()
    mesh_dofs = u_end[0].getSize()
    dofs_stats, dofs_stats_avg = problem.parabolic.get_dofs_stats()

    rank = 0 if not hasattr(problem.parabolic, "comm") else problem.parabolic.comm.Get_rank()

    norm_u_end = abs(u_end)

    if rank == 0:
        sim_data["cpu_time"] = cpu_time

        sol_data["error_L2_availabe"] = error_availabe
        sol_data["error_L2"] = error_L2
        sol_data["rel_error_L2"] = rel_error_L2
        sol_data["CV"] = CV
        sol_data["tot_dofs"] = tot_dofs
        sol_data["mesh_dofs"] = mesh_dofs
        sol_data["avg_dofs_per_proc"] = dofs_stats_avg[0]
        sol_data["avg_loc_dofs_per_proc"] = dofs_stats_avg[1]
        sol_data["avg_ghost_dofs_per_proc"] = dofs_stats_avg[2]
        sol_data["avg_perc_ghost_dofs_per_proc"] = dofs_stats_avg[3]

        file_name = problem.output_folder / Path(problem.output_file_name)
        data_man = database(file_name)
        problem_params_no_comm = problem_params
        if hasattr(problem_params_no_comm, "comm"):
            del problem_params_no_comm["comm"]
        data_man.write_dictionary("problem_params", problem_params_no_comm)
        data_man.write_dictionary("int_params", int_params)
        data_man.write_dictionary("sim_data", sim_data)
        data_man.write_dictionary("sol_data", sol_data)
        data_man.write_dictionary("step_stats", step_stats)

        # print some info
        print(f"Solved in {sim_data['cpu_time']:0.8f} seconds")
        print(f"Norm of solution: {norm_u_end}")
        print(f'CV = {sol_data["CV"]}')
        print(f"Step stats dict: {step_stats}")


if __name__ == "__main__":
    main()
