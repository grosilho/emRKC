import numpy as np

from problem_classes.space_discretizazions.anysotropic_ND_FD import AnysotropicNDimFinDiff
from scipy.sparse.linalg import cg
from datatype_classes.FD_Vector import FD_Vector
from pathlib import Path
import os


class Parabolic_FD:
    def __init__(self, **problem_params):

        for key, val in problem_params.items():
            setattr(self, key, val)

        self.define_domain()
        self.define_coefficients()
        self.define_Laplacian()
        self.define_solver()
        self.define_stimulus()
        self.define_eval_points()

        assert self.bc in ['N', 'P'], "bc must be either 'N' or 'P'"

    def __del__(self):
        if self.enable_output:
            self.output_file.close()
            with open(
                self.output_file_path.parent / Path(self.output_file_name + '_txyz').with_suffix(".npy"), 'wb'
            ) as f:
                np.save(f, np.array(self.t_out))
                xyz = self.NDim_FD.grids
                for i in range(self.dim):
                    np.save(f, xyz[i])

    @property
    def vector_type(self):
        return FD_Vector

    @property
    def mesh_name(self):
        return "ref_" + str(self.pre_refinements)

    def define_domain(self):
        if "cube" in self.domain_name:
            self.dom_size = ([0.0, 100.0], [0.0, 100.0], [0.0, 100.0])
            self.dim = int(self.domain_name[5])
        else:  # cuboid
            if "smaller" in self.domain_name:
                self.dom_size = ([0.0, 10.0], [0.0, 4.5], [0.0, 2.0])
            elif "small" in self.domain_name:
                self.dom_size = ([0.0, 5.0], [0.0, 3.0], [0.0, 1.0])
            elif "very_large" in self.domain_name:
                self.dom_size = ([0.0, 280.0], [0.0, 112.0], [0.0, 48.0])
            elif "large" in self.domain_name:
                self.dom_size = ([0.0, 60.0], [0.0, 21.0], [0.0, 9.0])
            else:
                self.dom_size = ([0.0, 20.0], [0.0, 7.0], [0.0, 3.0])
            self.dim = int(self.domain_name[7])

        self.n_elems = 5.0 * np.max(self.dom_size) * 2**self.pre_refinements + 1
        self.n_elems = int(np.round(self.n_elems))

        if self.bc == 'P':
            self.n_elems -= 1

        self.dom_size = self.dom_size[: self.dim]

    def define_coefficients(self):
        self.chi = 140.0  # mm^-1
        self.Cm = 0.01  # uF/mm^2
        self.si_l = 0.17  # mS/mm
        self.se_l = 0.62  # mS/mm
        self.si_t = 0.019  # mS/mm
        self.se_t = 0.24  # mS/mm

        if "cube" in self.domain_name:
            # if self.pre_refinements == -1: # only for generating initial value
            # self.si_l *= 0.5
            # self.se_l *= 0.5
            # elif self.pre_refinements == 0:
            # self.si_l *= 0.25
            # self.se_l *= 0.25
            self.si_t = self.si_l
            self.se_t = self.se_l

        self.sigma_l = self.si_l * self.se_l / (self.si_l + self.se_l)
        self.sigma_t = self.si_t * self.se_t / (self.si_t + self.se_t)
        self.diff_l = self.sigma_l / self.chi / self.Cm
        self.diff_t = self.sigma_t / self.chi / self.Cm

        if self.dim == 1:
            self.diff = (self.diff_l,)
        elif self.dim == 2:
            self.diff = (self.diff_l, self.diff_t)
        else:
            self.diff = (self.diff_l, self.diff_t, self.diff_t)

    def define_Laplacian(self):
        self.NDim_FD = AnysotropicNDimFinDiff(
            dom_size=self.dom_size,
            nvars=self.n_elems,
            diff=self.diff,
            derivative=2,
            stencil_type='center',
            order=self.order,
            bc='neumann-zero' if self.bc == 'N' else 'periodic',
        )
        self.n_dofs = self.NDim_FD.A.shape[0]
        self.init = self.n_dofs

    @property
    def grids(self):
        return self.NDim_FD.grids

    @property
    def dx(self):
        return np.max(self.NDim_FD.dx)

    def define_stimulus(self):
        self.zero_stim_vec = 0.0
        # all remaining stimulus parameters are set in MonodomainODE

    def define_solver(self):
        # we suppose that the problem is symmetric
        # if self.dim <= 1:
        #     self.solver = lambda mat, vec, guess: spsolve(mat, vec)
        # else:
        lin_solv_rtol = self.lin_solv_rtol if self.lin_solv_rtol is not None else 1e-5
        self.solver = lambda mat, vec, guess: cg(
            mat, vec, x0=guess, atol=0, tol=lin_solv_rtol, maxiter=self.lin_solv_max_iter
        )[0]

    def solve_system(self, rhs, factor, u0, t, u_sol):
        u_sol.values[:] = self.solver(self.NDim_FD.Id - factor * self.NDim_FD.A, rhs.values, u0.values)

        return u_sol

    def add_disc_laplacian(self, uh, res):
        res.values += self.NDim_FD.A @ uh.values

    def define_eval_points(self):
        if "small" in self.domain_name:
            n_pts = 5
            a = np.array([[0.5, 0.5, 0.5]])
        else:
            n_pts = 10
            a = np.array([[1.5, 1.5, 1.5]])
        a = a[:, : self.dim]
        dom_size = np.array(self.dom_size)[:, 1]
        b = dom_size.reshape((1, self.dim))
        x = np.reshape(np.linspace(0.0, 1.0, n_pts), (n_pts, 1))
        self.eval_points = a + (b - a) * x

    def init_output(self, output_folder):
        self.output_folder = output_folder
        self.output_file_path = self.output_folder / Path(self.output_file_name).with_suffix(".npy")
        if self.enable_output:
            if self.output_file_path.is_file():
                os.remove(self.output_file_path)
            if not self.output_folder.is_dir():
                os.makedirs(self.output_folder)
            self.output_file = open(self.output_file_path, 'wb')
            self.t_out = []

    def write_solution(self, u, t, all):
        if self.enable_output:
            if not all:
                np.save(self.output_file, u[0].values.reshape(self.NDim_FD.shape))
                self.t_out.append(t)
            else:
                raise NotImplementedError("all=True not implemented for Parabolic_FD.write_solution")

    def write_reference_solution(self, uh, indeces):
        if self.output_file_path.is_file():
            os.remove(self.output_file_path)
        if not self.output_file_path.parent.is_dir():
            os.makedirs(self.output_file_path.parent)
        with open(self.output_file_path, 'wb') as file:
            [np.save(file, uh[i].values.reshape(self.NDim_FD.shape)) for i in indeces]

    def read_reference_solution(self, uh, indeces, ref_file_name):
        ref_sol_path = Path(self.output_folder) / Path(ref_file_name).with_suffix(".npy")
        if ref_sol_path.is_file():
            with open(ref_sol_path, 'rb') as f:
                for i in indeces:
                    uh[i].values[:] = np.load(f).ravel()
            return True
        else:
            print(f'did not find {ref_sol_path}')
            return False

    def eval_on_points(self, u):
        return None

    def stim_region(self, stim_center, stim_radius):
        grids = self.NDim_FD.grids
        coord_inside_stim_box = []
        for i in range(len(grids)):
            coord_inside_stim_box.append(abs(grids[i] - stim_center[i]) < stim_radius[i])

        inside_stim_box = True
        for i in range(len(grids)):
            inside_stim_box = np.logical_and(inside_stim_box, coord_inside_stim_box[i])

        return self.vector_type(inside_stim_box.ravel().astype(float))

    def compute_errors(self, uh, ref_sol):
        # Compute L2 error
        error_L2 = np.linalg.norm(uh.values - ref_sol.values)
        sol_norm_L2 = np.linalg.norm(ref_sol.values)
        rel_error_L2 = error_L2 / sol_norm_L2

        return error_L2, rel_error_L2

    def get_dofs_stats(self):
        tmp = self.vector_type(self.init)
        data = (
            tmp.n_loc_dofs + tmp.n_ghost_dofs,
            tmp.n_loc_dofs,
            tmp.n_ghost_dofs,
            tmp.n_ghost_dofs / (tmp.n_loc_dofs + tmp.n_loc_dofs),
        )
        return (data,), data
