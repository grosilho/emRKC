import time
from tqdm import tqdm
import logging


class TimeIntegrator:
    def __init__(self, problem, step):
        self.step = step
        self.problem = problem

    def solve(self, dt, o_freq):
        problem = self.problem
        log_stat = logging.getLogger("stat")

        u_n = problem.initial_value()
        if o_freq > 0:
            problem.write_solution(u_n, problem.t0)

        dofs_stats, dofs_stats_avg = problem.parabolic.get_dofs_stats()
        if not hasattr(problem.parabolic, "comm") or problem.parabolic.comm.rank == 0:
            log_stat.info(f"Total dofs: {u_n.getSize()}")
            for i, d in enumerate(dofs_stats):
                log_stat.info(
                    f"Processor {i}: tot_dofs = {d[0]:.2e}, n_loc_dofs = {d[1]:.2e}, n_ghost_dofs = {d[2]:.2e}, %ghost = {100*d[3]:.2f}"
                )
            log_stat.info(
                f"Averages: tot_dofs_per_proc = {dofs_stats_avg[0]:.2e}, n_loc_dofs_per_proc = {dofs_stats_avg[1]:.2e}, n_ghost_dofs_per_proc = {dofs_stats_avg[2]:.2e}, %ghost_per_proc = {100*dofs_stats_avg[3]:.2f}"
            )

        n = 1
        t = problem.t0
        Tend = problem.Tend
        last = t >= Tend
        f = problem.dtype_f(problem.init, val=0.0)

        show_bar = logging.getLogger("step").getEffectiveLevel() == logging.WARNING

        if show_bar:
            pbar = tqdm(total=int((Tend - t) / dt))

        tic = time.perf_counter()

        while not last:
            if t + (1.0 + 1e-8) * dt >= Tend:
                dt = Tend - t
                last = True

            u_n = self.step(u_n, t, dt, f)
            t += dt

            if o_freq > 0 and n % o_freq == 0:
                problem.write_solution(u_n, t)

            n += 1
            if show_bar:
                pbar.update(1)

        toc = time.perf_counter()
        cpu_time = toc - tic

        if show_bar:
            pbar.close()

        if o_freq == -1:
            problem.write_reference_solution(u_n)
        elif o_freq == -2:
            problem.write_reference_solution(u_n, all=True)

        return Tend, u_n, cpu_time
