from integrators.explicit_stabilized_classes.rho_estimator import rho_estimator
from integrators.explicit_stabilized_classes.es_methods import RKC, RKW, RKU, HSRKU, mES
import logging


class explicit_stabilized:
    def __init__(self, params, problem):
        if "verbose" in params:
            self.verbose = params["verbose"]
        else:
            self.verbose = False

        self.logger_step = logging.getLogger("stepper")

        self.damping = 0.05
        self.safe_add = 2
        if "damping" in params:
            self.damping = params["damping"]
        if "safe_add" in params:
            self.safe_add = params["safe_add"]

        self.es = eval(params["ES_class"])(self.damping, self.safe_add)
        self.s_prev = 0

        # updates spectral radius every rho_freq steps
        if "rho_freq" not in params:
            self.rho_freq = 5
        else:
            self.rho_freq = params["rho_freq"]
        self.rho_count = 0
        self.never_estimated_rho = True

        self.P = problem

        if hasattr(self.P, "rho") and callable(self.P.rho):
            self.rho = self.P.rho
        else:
            self.rho_estimator = rho_estimator(self.P)
            self.rho = self.rho_estimator.rho

        self.gj = self.P.dtype_u(init=self.P.init, val=0.0)
        self.gjm1 = self.P.dtype_u(init=self.P.init, val=0.0)
        self.gjm2 = self.P.dtype_u(init=self.P.init, val=0.0)

        self.steps_counter = 0
        self.s_avg = 0

    def update_stages_coefficients(self, u, t, dt, fu=None):
        if self.rho_count % self.rho_freq == 0:
            self.estimated_rho = self.rho(y=u, t=t, fy=fu)
            self.s = self.es.get_s(dt * self.estimated_rho)
            self.rho_count = 1
            if self.s != self.s_prev:
                self.es.update_coefficients(self.s)
                self.s_prev = self.s
        else:
            self.rho_count += 1

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, fh=self.gj)
        self.update_stages_coefficients(u, t, dt, self.gj)

        if self.logger_step.level <= logging.INFO:
            self.logger_step.info(
                f"t = {t:1.2e}, dt = {dt:1.2e}, s = {self.s}, rho = {self.estimated_rho:1.2e}, |y| = {abs(u):1.2e}"
            )

        self.gj.aypx(self.es.mu[0] * dt, u)
        for j in range(2, self.s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * self.es.c[j - 2], fh=self.gj)

            self.gj.axpby(self.es.nu[j - 1], self.es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(self.es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(self.es.kappa[j - 1], u)

        u, self.gj = self.gj, u

        self.steps_counter += 1
        self.s_avg += self.s

        return u

    def get_stats(self):
        if self.steps_counter > 0:
            self.s_avg = self.s_avg / self.steps_counter
        step_stats = dict()
        step_stats["s_avg"] = self.s_avg

        return step_stats


class multirate_explicit_stabilized(explicit_stabilized):
    def __init__(self, params, problem):
        super(multirate_explicit_stabilized, self).__init__(params, problem)

        self.scale_separation = True
        params["ES_class_outer"] = params["ES_class"]
        params["ES_class_inner"] = params["ES_class"]
        self.es = mES(
            params["ES_class_outer"],
            params["ES_class_inner"],
            self.damping,
            self.safe_add,
            scale_separation=self.scale_separation,
        )
        self.s_prev = 0
        self.m_prev = 0

        self.s_fixed = [0, 0]
        if "ES_s_outer" in params:
            self.s_fixed[0] = params["ES_s_outer"]
            # notice that that to fix s_inner then also s_outer must be fixed
            if "ES_m_inner" in params:
                self.s_fixed[1] = params["ES_m_inner"]

        if hasattr(self.P, "rho_nonstiff") and callable(self.P.rho_nonstiff):
            self.rho_expl = self.P.rho_nonstiff
        else:
            if not hasattr(self, "rho_estimator"):
                self.rho_estimator = rho_estimator(self.P)
            self.rho_expl = self.rho_estimator.rho_expl

        if hasattr(self.P, "rho_stiff") and callable(self.P.rho_stiff):
            self.rho_impl = self.P.rho_stiff
        else:
            if not hasattr(self, "rho_estimator"):
                self.rho_estimator = rho_estimator(self.P)
            self.rho_impl = self.rho_estimator.rho_impl

        if hasattr(self, "rho_estimator"):
            self.estimated_rho = self.rho_estimator.eigval
        else:

            class pair:
                def __init__(self, impl, expl):
                    self.impl = impl
                    self.expl = expl

            self.estimated_rho = pair(0.0, 0.0)

        self.rhs_stiff_args = problem.rhs_stiff_args

        self.fu = self.P.dtype_f(init=self.P.init, val=0.0)
        self.uj = self.P.dtype_u(init=self.P.init, val=0.0)
        self.ujm1 = self.P.dtype_u(init=self.P.init, val=0.0)
        self.ujm2 = self.P.dtype_u(init=self.P.init, val=0.0)

        self.eval_f_before_update_stages = True

        self.m_avg = 0

    def update_stages_coefficients(self, u, t, dt, fu=None):
        if self.rho_count % self.rho_freq == 0:
            if self.s_fixed[0] == 0:
                self.estimated_rho.expl = self.rho_expl(y=u, t=t, fy=fu)
                self.estimated_rho.impl = self.rho_impl(y=u, t=t, fy=fu)
                self.s = self.es.get_s(dt * self.estimated_rho.expl)
                self.m = self.es.get_m(dt, self.estimated_rho.impl, self.s)
            elif self.s_fixed[1] == 0:
                self.s = self.s_fixed[0]
                self.estimated_rho.impl = self.rho_impl(y=u, t=t, fy=fu)
                self.m = self.es.get_m(dt, self.estimated_rho.impl, self.s)
            else:
                self.s = self.s_fixed[0]
                self.m = self.s_fixed[1]
                self.es.fix_eta(dt, self.s, self.m)
                self.estimated_rho = self.rho_estimator.eigval  # dummy variable

            self.rho_count = 1
            if self.s != self.s_prev or self.m != self.m_prev:
                self.es.update_coefficients(self.s, self.m)
                self.s_prev = self.s
                self.m_prev = self.m
        else:
            self.rho_count += 1

    def step(self, u, t, dt, *args):
        P = self.P

        if self.eval_f_before_update_stages:
            P.eval_f(u, t, eval_impl=True, eval_expl=True, eval_exp=False, fh=self.fu)
            self.update_stages_coefficients(u, t, dt, self.fu)
        else:
            self.update_stages_coefficients(u, t, dt)

        if self.logger_step.level <= logging.INFO:
            self.logger_step.info(
                f"t = {t:1.2e}, dt = {dt:1.2e}, s = {self.s}, m = {self.m}, rho_slow = {self.estimated_rho.expl:1.2e}, rho_fast = {self.estimated_rho.impl:1.2e}, |y| = {abs(u):1.2e}"
            )

        # computes f_eta and stores result in self.uj
        # we pass self.fu as it is reused
        if self.eval_f_before_update_stages:
            self.f_eta(u, t, fu=self.fu)
        else:
            self.f_eta(u, t)

        self.gj, self.uj = self.uj, self.gj

        self.gj *= self.es.mu[0] * dt
        self.gj += u
        for j in range(2, self.s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            self.f_eta(self.gjm1, t + dt * self.es.c[j - 2])  # computes f_eta and stores result in self.uj
            self.gj, self.uj = self.uj, self.gj

            self.gj.axpby(self.es.nu[j - 1], self.es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(self.es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(self.es.kappa[j - 1], u)

        self.gj, u = u, self.gj

        self.steps_counter += 1
        self.s_avg += self.s
        self.m_avg += self.m

        return u

    def f_eta(self, u, t, fu=None):
        P = self.P

        # in what follows we iterate only over the components of the vectors that are arguments of f_stiff or are modified by f_stiff.
        # there is no need to keep updated variables that do not interact throught f_stiff.
        # for some problems (as monodomain model) this avoids a lot of computations
        #

        if fu is None:
            P.eval_f(u, t, eval_impl=True, eval_expl=True, eval_exp=False, fh=self.fu)

        self.uj, self.fu.impl = self.fu.impl, self.uj
        self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
        self.uj.imul_sub(self.es.alpha[0] * self.es.eta, self.rhs_stiff_args)
        self.uj.iadd_sub(u, self.rhs_stiff_args)

        for j in range(2, self.m + 1):
            self.ujm2, self.ujm1, self.uj = self.ujm1, self.uj, self.ujm2

            P.eval_f(
                self.ujm1,
                t + self.es.eta * self.es.d[j - 2],
                eval_impl=True,
                eval_expl=False,
                eval_exp=False,
                fh=self.fu,
                zero_untouched_indeces=False,
            )
            self.uj, self.fu.impl = self.fu.impl, self.uj

            self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
            self.uj.axpby_sub(self.es.beta[j - 1], self.es.alpha[j - 1] * self.es.eta, self.ujm1, self.rhs_stiff_args)
            if j > 2:
                self.uj.axpy_sub(self.es.gamma[j - 1], self.ujm2, self.rhs_stiff_args)
            else:
                self.uj.axpy_sub(self.es.gamma[j - 1], u, self.rhs_stiff_args)

        self.uj.isub_sub(u, self.rhs_stiff_args)
        self.uj.imul_sub(1.0 / self.es.eta, self.rhs_stiff_args)

        # here we could just swap uj and fu.expl for the non rhs_stiff_args. It works when doing stuff fully in python,
        # but when using the c++ bindings for the ionic model it doesn't. Dont know why...
        self.uj.copy_sub(self.fu.expl, [i for i in range(P.size) if i not in self.rhs_stiff_args])

        return self.uj

    def get_stats(self):
        if self.steps_counter > 0:
            self.s_avg = self.s_avg / self.steps_counter
            self.m_avg = self.m_avg / self.steps_counter
        step_stats = dict()
        step_stats["s_avg"] = self.s_avg
        step_stats["m_avg"] = self.m_avg

        return step_stats


class exponential_multirate_explicit_stabilized(multirate_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_multirate_explicit_stabilized, self).__init__(params, problem)

        self.rhs_stiff_args = problem.rhs_stiff_args
        self.rhs_exp_indeces = problem.rhs_exp_indeces

        self.eval_f_before_update_stages = False

    def f_eta(self, u, t, fu=None):
        P = self.P

        # in what follows we iterate only over the components of the vectors that are arguments of f_stiff or are modified by f_stiff.
        # there is no need to keep updated variables that do not interact throught f_stiff.
        # for some problems (as monodomain model) this avoids a lot of computations
        #
        # eval phi once on u, update u=u+eta*phi(eta)f(u), compute f_eta on this updated u (exponentials are no more computed since are already fully added to u)

        P.eval_phi_f_exp(u, self.es.eta, t, self.fu.exp)
        self.ujm1.copy(u)
        self.ujm1.axpy_sub(self.es.eta, self.fu.exp, self.rhs_exp_indeces)

        P.eval_f(self.ujm1, t, eval_impl=True, eval_expl=True, eval_exp=False, fh=self.fu)

        self.uj, self.fu.impl = self.fu.impl, self.uj
        self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
        self.uj.aypx_sub(self.es.alpha[0] * self.es.eta, self.ujm1, self.rhs_stiff_args)

        for j in range(2, self.m + 1):
            self.ujm2, self.ujm1, self.uj = self.ujm1, self.uj, self.ujm2

            P.eval_f(
                self.ujm1,
                t + self.es.eta * self.es.d[j - 2],
                eval_impl=True,
                eval_expl=False,
                eval_exp=False,
                fh=self.fu,
                zero_untouched_indeces=False,
            )

            self.uj, self.fu.impl = self.fu.impl, self.uj
            self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
            self.uj.axpby_sub(self.es.beta[j - 1], self.es.alpha[j - 1] * self.es.eta, self.ujm1, self.rhs_stiff_args)
            self.uj.axpy_sub(self.es.gamma[j - 1], self.ujm2, self.rhs_stiff_args)

        self.uj.isub_sub(u, self.rhs_stiff_args)
        self.uj.imul_sub(1.0 / self.es.eta, self.rhs_stiff_args)

        if hasattr(type(u), "swap_sub"):
            self.uj.swap_sub(self.fu.expl, [i for i in range(P.size) if i not in self.rhs_stiff_args])
        else:
            self.uj.copy_sub(self.fu.expl, [i for i in range(P.size) if i not in self.rhs_stiff_args])
        # self.uj.iadd_sub(self.fu.exp, [i for i in range(P.size) if i not in self.rhs_stiff_args])
        self.uj.iadd_sub(self.fu.exp, [i for i in self.rhs_exp_indeces if i not in self.rhs_stiff_args])

        return self.uj


class exponential_multirate_explicit_stabilized_progress(multirate_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_multirate_explicit_stabilized_progress, self).__init__(params, problem)

        self.rhs_nonstiff_args = problem.rhs_nonstiff_args
        self.rhs_stiff_args = problem.rhs_stiff_args
        self.rhs_exp_args = problem.rhs_exp_args
        self.rhs_exp_indeces = problem.rhs_exp_indeces
        self.rhs_stiff_or_exp_args = [i for i in range(problem.size) if i in self.rhs_stiff_args + self.rhs_exp_args]

        self.eval_f_before_update_stages = False

    def f_eta(self, u, t, fu=None):
        P = self.P

        # in what follows we iterate only over the components of the vectors that are arguments of f_stiff or are modified by f_stiff.
        # there is no need to keep updated variables that do not interact throught f_stiff.
        # for some problems (as monodomain model) this avoids a lot of computations
        #
        # eval phi once on u, eval f.expl on u+eta*phi, eval f.impl and add phi(u) progressively

        P.eval_phi_f_exp(u, self.es.eta, t, self.fu.exp)
        self.uj.copy(u)
        self.uj.axpy_sub(self.es.eta, self.fu.exp, self.rhs_exp_indeces)

        P.eval_f(self.uj, t, eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
        P.eval_f(u, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)

        self.uj, self.fu.impl = self.fu.impl, self.uj
        self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
        self.uj.iadd_sub(self.fu.exp, self.rhs_stiff_args)
        self.uj.imul_sub(self.es.alpha[0] * self.es.eta, self.rhs_stiff_args)
        self.uj.iadd_sub(u, self.rhs_stiff_args)

        for j in range(2, self.m + 1):
            self.ujm2, self.ujm1, self.uj = self.ujm1, self.uj, self.ujm2

            P.eval_f(
                self.ujm1,
                t + self.es.eta * self.es.d[j - 2],
                eval_impl=True,
                eval_expl=False,
                eval_exp=False,
                fh=self.fu,
            )

            self.uj, self.fu.impl = self.fu.impl, self.uj

            self.uj.iadd_sub(self.fu.expl, self.rhs_stiff_args)
            self.uj.iadd_sub(self.fu.exp, self.rhs_stiff_args)
            self.uj.axpby_sub(self.es.beta[j - 1], self.es.alpha[j - 1] * self.es.eta, self.ujm1, self.rhs_stiff_args)
            if j > 2:
                self.uj.axpy_sub(self.es.gamma[j - 1], self.ujm2, self.rhs_stiff_args)
            else:
                self.uj.axpy_sub(self.es.gamma[j - 1], u, self.rhs_stiff_args)

        self.uj.isub_sub(u, self.rhs_stiff_args)
        self.uj.imul_sub(1.0 / self.es.eta, self.rhs_stiff_args)

        if hasattr(type(u), "swap_sub"):
            self.uj.swap_sub(self.fu.expl, [i for i in range(P.size) if i not in self.rhs_stiff_args])
        else:
            self.uj.copy_sub(self.fu.expl, [i for i in range(P.size) if i not in self.rhs_stiff_args])
        self.uj.iadd_sub(self.fu.exp, [i for i in range(P.size) if i not in self.rhs_stiff_args])

        return self.uj


# Here we list some other methods based on splitting instead of the modified equation.
# They work less efficiently and are not pusblished


class exponential_splitting_explicit_stabilized(explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_splitting_explicit_stabilized, self).__init__(params, problem)

        self.es = [
            params["ES_class_outer"](self.damping, self.safe_add),
            params["ES_class_inner"](self.damping, self.safe_add),
        ]
        self.s = [0, 0]
        self.s_prev = [0, 0]

        self.s_fixed = [0, 0]
        if "ES_s_outer" in params:
            self.s_fixed[0] = params["ES_s_outer"]
        if "ES_m_inner" in params:
            self.s_fixed[1] = params["ES_m_inner"]

        self.rho = [None, None]
        if hasattr(self.P, "rho_nonstiff") and callable(self.P.rho_nonstiff):
            self.rho_expl = self.P.rho_nonstiff
        else:
            self.rho_expl = self.rho_estimator.rho_expl
        if hasattr(self.P, "rho_stiff") and callable(self.P.rho_stiff):
            self.rho_impl = self.P.rho_stiff
        else:
            self.rho_impl = self.rho_estimator.rho_impl
        self.estimated_rho = [0, 0]
        self.rho_count = [0, 0]

        self.rhs_nonstiff_args = problem.rhs_nonstiff_args
        self.rhs_stiff_args = problem.rhs_stiff_args
        self.rhs_exp_args = problem.rhs_exp_args
        self.rhs_stiff_or_exp_args = [i for i in range(problem.size) if i in self.rhs_stiff_args + self.rhs_exp_args]

        self.fu = self.P.dtype_f(self.P.init, val=0.0)

    def update_stages_coefficients(self, u, t, dt, i, fu=None):
        if self.rho_count[i] % self.rho_freq == 0:
            if self.s_fixed[i] == 0:
                self.estimated_rho[i] = self.rho[i](y=u, t=t, fy=fu)
                self.s[i] = self.es[i].get_s(dt * self.estimated_rho[i])
            else:
                self.s[i] = self.s_fixed[i]

            self.rho_count[i] = 1
            if self.s[i] != self.s_prev[i]:
                self.es[i].update_coefficients(self.s[i])
                self.s_prev[i] = self.s[i]
        else:
            self.rho_count[i] += 1

        self.print_info(u, t, dt, i)

    def print_info(self, u, t, dt, i):
        self.logger_step.info(
            f"t = {t:1.2e}, dt = {dt:1.2e}, s[{i}] = {self.s[i]}, rho[{i}] = {self.estimated_rho[i]:1.2e}, |y| = {abs(u):1.2e} "
        )


class exponential_splitting_explicit_stabilized_V1(exponential_splitting_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_splitting_explicit_stabilized_V1, self).__init__(params, problem)

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 0, fu=self.fu)

        # integrate non stiff terms
        es = self.es[0]
        s = self.s[0]
        self.gj, self.fu.expl = self.fu.expl, self.gj
        self.gj *= es.mu[0] * dt
        self.gj += u
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
            self.gj, self.fu.expl = self.fu.expl, self.gj

            self.gj.axpby(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(es.kappa[j - 1], u)

        # performs exponential Euler step
        P.phi_one_f_eval(self.gj, dt, t, self.fu.exp)
        self.gj.axpy_sub(dt, self.fu.exp, self.rhs_exp_args)
        u, self.gj = self.gj, u

        P.eval_f(u, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 1, fu=self.fu)

        # integrate stiff terms
        es = self.es[1]
        s = self.s[1]
        self.gj, self.fu.impl = self.fu.impl, self.gj
        self.gj.imul_sub(es.mu[0] * dt, self.rhs_stiff_args)
        self.gj.iadd_sub(u, self.rhs_stiff_args)

        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
            self.gj, self.fu.impl = self.fu.impl, self.gj

            self.gj.axpby_sub(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1, self.rhs_stiff_args)
            if j > 2:
                self.gj.axpy_sub(es.kappa[j - 1], self.gjm2, self.rhs_stiff_args)
            else:
                self.gj.axpy_sub(es.kappa[j - 1], u, self.rhs_stiff_args)

        if hasattr(type(u), "swap_sub"):
            u.swap_sub(self.gj, self.rhs_stiff_args)
        else:
            u.copy_sub(self.gj, self.rhs_stiff_args)

        return u


class exponential_splitting_explicit_stabilized_V2(exponential_splitting_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_splitting_explicit_stabilized_V2, self).__init__(params, problem)

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 0, fu=self.fu)

        # integrate non stiff terms
        es = self.es[0]
        s = self.s[0]
        self.gj, self.fu.expl = self.fu.expl, self.gj
        self.gj *= es.mu[0] * dt
        self.gj += u
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
            self.gj, self.fu.expl = self.fu.expl, self.gj

            self.gj.axpby(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(es.kappa[j - 1], u)

        self.gj, u = u, self.gj

        P.eval_f(u, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 1, fu=self.fu)

        # integrate stiff terms
        es = self.es[1]
        s = self.s[1]
        self.gj, self.fu.impl = self.fu.impl, self.gj
        self.gj.imul_sub(es.mu[0] * dt, self.rhs_stiff_args)
        self.gj.iadd_sub(u, self.rhs_stiff_or_exp_args)
        P.phi_one_f_eval(self.gj, es.mu[0] * dt, t, self.fu.exp)
        self.gj.axpy_sub(es.mu[0] * dt, self.fu.exp, self.rhs_exp_args)
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
            self.gj, self.fu.impl = self.fu.impl, self.gj

            self.gj.axpby_sub(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1, self.rhs_stiff_or_exp_args)
            if j > 2:
                self.gj.axpy_sub(es.kappa[j - 1], self.gjm2, self.rhs_stiff_or_exp_args)
            else:
                self.gj.axpy_sub(es.kappa[j - 1], u, self.rhs_stiff_or_exp_args)

            P.phi_one_f_eval(self.gj, es.mu[j - 1] * dt, t + es.c[j - 2] * dt, self.fu.exp)
            self.gj.axpy_sub(es.mu[j - 1] * dt, self.fu.exp, self.rhs_exp_args)

        if hasattr(type(u), "swap_sub"):
            u.swap_sub(self.gj, self.rhs_stiff_or_exp_args)
        else:
            u.copy_sub(self.gj, self.rhs_stiff_or_exp_args)

        return u


class exponential_splitting_explicit_stabilized_V3(exponential_splitting_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_splitting_explicit_stabilized_V3, self).__init__(params, problem)

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 0, fu=self.fu)

        # integrate non stiff terms
        es = self.es[0]
        s = self.s[0]
        self.gj, self.fu.expl = self.fu.expl, self.gj
        self.gj *= es.mu[0] * dt
        self.gj += u
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
            self.gj, self.fu.expl = self.fu.expl, self.gj

            self.gj.axpby(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(es.kappa[j - 1], u)

        self.gj, u = u, self.gj

        P.eval_f(u, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 1, fu=self.fu)

        # integrate stiff terms
        es = self.es[1]
        s = self.s[1]
        P.phi_one_f_eval(u, es.mu[0] * dt, t, self.fu.exp)
        self.gj, self.fu.impl = self.fu.impl, self.gj
        self.gj.iadd_sub(self.fu.exp, self.rhs_exp_args)
        self.gj.aypx_sub(es.mu[0] * dt, u, self.rhs_stiff_or_exp_args)
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
            P.phi_one_f_eval(self.gjm1, es.mu[j - 1] * dt, t + dt * es.c[j - 2], self.fu.exp)
            self.gj, self.fu.impl = self.fu.impl, self.gj
            self.gj.iadd_sub(self.fu.exp, self.rhs_exp_args)
            self.gj.axpby_sub(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1, self.rhs_stiff_or_exp_args)
            if j > 2:
                self.gj.axpy_sub(es.kappa[j - 1], self.gjm2, self.rhs_stiff_or_exp_args)
            else:
                self.gj.axpy_sub(es.kappa[j - 1], u, self.rhs_stiff_or_exp_args)

        if hasattr(type(u), "swap_sub"):
            u.swap_sub(self.gj, self.rhs_stiff_or_exp_args)
        else:
            u.copy_sub(self.gj, self.rhs_stiff_or_exp_args)

        return u


class exponential_splitting_explicit_stabilized_V4(exponential_splitting_explicit_stabilized):
    def __init__(self, params, problem):
        super(exponential_splitting_explicit_stabilized_V4, self).__init__(params, problem)

    def step(self, u, t, dt, *args):
        P = self.P

        P.eval_f(u, t, eval_impl=True, eval_expl=True, eval_exp=False, fh=self.fu)
        self.update_stages_coefficients(u, t, dt, 0, fu=self.fu)
        self.update_stages_coefficients(u, t, dt, 1, fu=self.fu)

        # integrate non stiff terms
        es = self.es[0]
        s = self.s[0]
        self.gj, self.fu.expl = self.fu.expl, self.gj
        self.gj *= es.mu[0] * dt
        self.gj += u
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=False, eval_expl=True, eval_exp=False, fh=self.fu)
            self.gj, self.fu.expl = self.fu.expl, self.gj

            self.gj.axpby(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(es.kappa[j - 1], u)

        self.fu.expl, self.gj = self.gj, self.fu.expl
        self.fu.expl -= u
        self.fu.expl *= 1.0 / dt

        # integrate stiff terms
        es = self.es[1]
        s = self.s[1]
        P.phi_one_f_eval(u, es.mu[0] * dt, t, self.fu.exp)
        self.gj, self.fu.impl = self.fu.impl, self.gj
        self.gj.iadd_sub(self.fu.exp, self.rhs_exp_args)
        self.gj += self.fu.expl
        self.gj.aypx(es.mu[0] * dt, u)
        for j in range(2, s + 1):
            self.gjm2, self.gjm1, self.gj = self.gjm1, self.gj, self.gjm2

            P.eval_f(self.gjm1, t + dt * es.c[j - 2], eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fu)
            P.phi_one_f_eval(self.gjm1, es.mu[j - 1] * dt, t + dt * es.c[j - 2], self.fu.exp)
            self.gj, self.fu.impl = self.fu.impl, self.gj
            self.gj.iadd_sub(self.fu.exp, self.rhs_exp_args)
            self.gj += self.fu.expl
            self.gj.axpby(es.nu[j - 1], es.mu[j - 1] * dt, self.gjm1)
            if j > 2:
                self.gj.axpy(es.kappa[j - 1], self.gjm2)
            else:
                self.gj.axpy(es.kappa[j - 1], u)

        self.gj, u = u, self.gj

        return u
