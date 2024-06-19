import numpy as np
from explicit_stabilized_classes.rho_estimator import rho_estimator
import logging


class explicit_Euler:
    def __init__(self, params, problem):
        self.params = params

        if "verbose" in params:
            self.verbose = params["verbose"]
        else:
            self.verbose = False

        self.logger_step = logging.getLogger("step")

        # updates spectral radius every rho_freq steps
        if "rho_freq" not in params:
            self.rho_freq = 5
        else:
            self.rho_freq = params["rho_freq"]
        self.rho_count = 0

        self.P = problem

        if hasattr(self.P, "rho") and callable(self.P.rho):
            self.rho = self.P.rho
        else:
            self.rho_estimator = rho_estimator(self.P)
            self.rho = self.rho_estimator.rho

        self.f0 = self.P.dtype_u(init=self.P.init, val=0.0)

        self.steps_counter = 0
        self.substeps_avg = 0

    def update_n_substeps(self, u, t, dt, fu=None):
        if self.rho_count % self.rho_freq == 0:
            self.estimated_rho = self.rho(y=u, t=t, fy=fu)
            if self.estimated_rho > 0.0:
                self.max_dt = min(0.9 * 2.0 / self.estimated_rho, dt)
            else:
                self.max_dt = dt
            self.n_substeps = int(np.ceil(dt / self.max_dt))
            self.rho_count = 1
        else:
            self.rho_count += 1

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, fh=self.f0)
        self.update_n_substeps(u, t, dt, self.f0)

        if self.logger_step.level <= logging.INFO:
            self.logger_step.info(
                f"t = {t:1.2e}, dt = {dt:1.2e}, n = {self.n_substeps}, max_dt = {self.max_dt:1.2e}, rho = {self.estimated_rho:1.2e}, |y| = {abs(u):1.2e}"
            )

        u.axpy(self.max_dt, self.f0)
        for j in range(2, self.n_substeps + 1):
            P.eval_f(u, t + self.max_dt * (j - 1), fh=self.f0)
            u.axpy(self.max_dt, self.f0)

        self.steps_counter += 1
        self.substeps_avg += self.n_substeps

        return u

    # def step(self, u, t, dt, *args):
    #     P = self.P

    #     P.eval_f(u, t, fh=self.f0)
    #     self.update_n_substeps(u, t, dt, self.f0)

    #     if self.logger_step.level <= logging.INFO:
    #         self.logger_step.info(f"t = {t:1.2e}, dt = {dt:1.2e}, n = {self.n_substeps}, max_dt = {self.max_dt:1.2e}, rho = {self.estimated_rho:1.2e}, |y| = {abs(u):1.2e}")

    #     u.axpy(dt, self.f0)

    #     self.steps_counter += 1
    #     self.substeps_avg += 1

    #     return u

    def get_stats(self):
        self.substeps_avg = self.substeps_avg / self.steps_counter
        step_stats = dict()
        step_stats["substeps_avg"] = self.substeps_avg

        return step_stats


class exponential_explicit_implicit_Euler:
    def __init__(self, params, problem):
        self.params = params

        if "verbose" in params:
            self.verbose = params["verbose"]
        else:
            self.verbose = False

        self.logger_step = logging.getLogger("step")

        self.P = problem

        self.fu = self.P.dtype_f(init=self.P.init, val=0.0)

    def step(self, u, t, dt, *args):
        P = self.P

        if self.logger_step.level <= logging.INFO:
            self.logger_step.info(f"t = {t:1.2e}, dt = {dt}, ||y|| = {abs(u)}")

        P.eval_phi_f_exp(u, dt, t, self.fu.exp)
        u.axpy(dt, self.fu.exp)

        P.eval_f(u, t, eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
        u.axpy(dt, self.fu.expl)

        P.solve_system(u, dt, u, t + dt, u)

        return u

    def get_stats(self):
        step_stats = dict()
        return step_stats
