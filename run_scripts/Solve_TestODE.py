from pySDC.projects.Monodomain.problem_classes.TestODE import TestODE, MultiscaleTestODE
from integrators.ExplicitStabilized import explicit_stabilized
from integrators.ExplicitStabilized import exponential_multirate_explicit_stabilized
from integrators.standard import explicit_Euler, exponential_explicit_implicit_Euler
from integrators.TimeIntegrator import TimeIntegrator

import logging
from utils.data_management import database
import argparse
from pathlib import Path
import os


def get_default_problem_params():
    problem_params = dict()
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    problem_params["end_time"] = -1.0
    problem_params["enable_output"] = False
    problem_params["output_root"] = executed_file_dir + "/results_monodomain/"
    problem_params["output_file_name"] = "monodomain"
    problem_params["lmbda_laplacian"] = -5.0
    problem_params["lmbda_gating"] = -10.0
    problem_params["lmbda_others"] = -1.0

    return problem_params


def get_default_integrator_params():
    int_params = dict()
    int_params["o_freq"] = 1
    int_params["dt"] = 0.1
    int_params["integrator"] = "exp_mES"
    int_params["log_level"] = 30

    int_params["rho_freq"] = 1e30
    int_params["damping"] = 0.05
    int_params["safe_add"] = 0
    int_params["es_class"] = "RKC1"
    int_params["es_s_outer"] = 0  # if given, or not zero, then the algorithm fixes s of the outer stabilized scheme to this value.
    int_params["es_s_inner"] = 0

    return int_params


def get_problem_and_stepper(problem_params, int_params):
    if int_params["integrator"] == "EE":
        problem = TestODE(**problem_params)
        EE = explicit_Euler(int_params, problem)
        stepper = EE
    elif int_params["integrator"] == "IMEXEXP":
        problem_params["splitting"] = "exp_nonstiff"
        problem = MultiscaleTestODE(**problem_params)
        IMEXEXP = exponential_explicit_implicit_Euler(int_params, problem)
        stepper = IMEXEXP
    elif int_params["integrator"] == "ES":
        problem = TestODE(**problem_params)
        ES = explicit_stabilized(int_params, problem)
        stepper = ES
    elif int_params["integrator"] == "mES":
        raise Exception("Test equation for mES not implemented")
    elif int_params["integrator"] == "exp_mES":
        problem_params["splitting"] = "exp_nonstiff"
        problem = MultiscaleTestODE(**problem_params)
        ES = exponential_multirate_explicit_stabilized(int_params, problem)
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
    parser.add_argument("--integrator", default="exp_mES", type=str, help="integration scheme: IMEXEXP, EE, ES, mES, exp_mES")
    parser.add_argument("--dt", default=0.5, type=float, help="step size")
    parser.add_argument("--end_time", default=-1.0, type=float, help="end time. If negative, a default one is used")
    parser.add_argument("--es_class", default="RKC1", type=str, help="explicit stabilized method: RKC1, RKU1, RKW1, HSRKU1")
    parser.add_argument("--es_s_outer", default=0, type=int, help="if given, fix the outer number of stages")
    parser.add_argument("--es_s_inner", default=0, type=int, help="if given, fixes the inner number of stage (if also the outer is fixed)")
    parser.add_argument("--o_freq", default=0, type=int, help="output frequency, every o_freq steps. If -1 then saves final solution as reference solution")
    parser.add_argument("--rho_freq", default=1e30, type=int, help="spectral radii are updated every rho_freq steps.")
    parser.add_argument("--safe_add", default=0, type=int, help="add safe_add additional stages")
    parser.add_argument("--log_level", default=20, type=int, help="log level: 10 debug, 20 info, 30 warning")
    # problem args
    parser.add_argument("--enable_output", default=False, action=argparse.BooleanOptionalAction, help="activate or deactivate xdmf output: True or False")
    parser.add_argument("--output_file_name", default="monodomain", type=str, help="output file name")
    parser.add_argument("--output_root", default="results_tmp/", type=str, help="output root folder")

    args = vars(parser.parse_args())

    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    args["output_root"] = executed_file_dir + "/" + args["output_root"]

    problem_params = modify_params(problem_params, args)
    int_params = modify_params(int_params, args)

    t_end, u_end, sim_data, sol_data, step_stats = run_simulation(problem_params, int_params)

    norm_u_end = abs(u_end)
    print(f"Solved in {sim_data['cpu_time']:0.8f} seconds")
    print(f"Norm of solution: {norm_u_end}")
    print(f"Step stats dict: {step_stats}")


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

    sim_data["cpu_time"] = cpu_time

    sol_data["error_L2_availabe"] = error_availabe
    sol_data["error_L2"] = error_L2
    sol_data["rel_error_L2"] = rel_error_L2

    file_name = problem.output_folder / Path(problem.output_file_name)
    data_man = database(file_name)
    data_man.write_dictionary("problem_params", problem_params)
    data_man.write_dictionary("int_params", int_params)
    data_man.write_dictionary("sim_data", sim_data)
    data_man.write_dictionary("sol_data", sol_data)
    data_man.write_dictionary("step_stats", step_stats)

    return t_end, u_end, sim_data, sol_data, step_stats


if __name__ == "__main__":
    main()
