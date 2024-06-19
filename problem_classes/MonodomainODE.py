from pathlib import Path
import logging
import numpy as np
from datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors
import problem_classes.ionicmodels.cpp as ionicmodels


class MonodomainODE:
    def __init__(self, **problem_params):
        self.logger = logging.getLogger("step")

        # the class for the spatial discretization of the parabolic part of monodomain
        if problem_params['space_disc'] == "FEM":
            from problem_classes.space_discretizazions.Parabolic_FEniCSx import Parabolic_FEniCSx as Parabolic
        elif problem_params['space_disc'] == "FD":
            from problem_classes.space_discretizazions.Parabolic_FD import Parabolic_FD as Parabolic
        elif problem_params['space_disc'] == "DCT":
            from problem_classes.space_discretizazions.Parabolic_DCT import Parabolic_DCT as Parabolic

        self.parabolic = Parabolic(**problem_params)

        # self.init = self.parabolic.init
        self.init = self.parabolic.init

        # store all problem params dictionary values as attributes
        for key, val in problem_params.items():
            setattr(self, key, val)

        self.define_ionic_model()
        self.define_stimulus()

        # initial and end time
        self.t0 = 0.0
        self.Tend = 50.0 if self.end_time <= 0.0 else self.end_time

        # dtype_u and dtype_f are super vectors of vector_type
        self.vector_type = self.parabolic.vector_type

        def dtype_u(init, val=None):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        def dtype_f(init, val=None):
            return VectorOfVectors(init, val, self.vector_type, self.size)

        # in the VectorOfVectors, the first index is the potential V, the other indices are the ionic model variables

        self.dtype_u = dtype_u
        self.dtype_f = dtype_f

        # init output stuff
        self.output_folder = (
            Path(self.output_root)
            / Path(self.parabolic.domain_name)
            / Path(self.parabolic.mesh_name)
            / Path(self.ionic_model_name)
        )
        self.parabolic.init_output(self.output_folder)

    def write_solution(self, uh, t):
        # write solution to file, only the potential V=uh[0], not the ionic model variables
        self.parabolic.write_solution(uh, t, not self.output_V_only)

    def write_reference_solution(self, uh, all=False):
        # write solution to file, only the potential V=uh[0] or all variables if all=True
        self.parabolic.write_reference_solution(uh, list(range(uh.size)) if all else [0])

    def read_reference_solution(self, uh, ref_file_name, all=False):
        # read solution from file, only the potential V=uh[0] or all variables if all=True
        # returns true if read was successful, false else
        return self.parabolic.read_reference_solution(uh, list(range(uh.size)) if all else [0], ref_file_name)

    def initial_value(self):
        # create initial value (as vector of vectors). Every variable is constant in space
        u0 = self.dtype_u(self.init, val=self.ionic_model.initial_values())

        # overwwrite the initial value with solution from file if desired
        if self.read_init_val:
            read_ok = self.read_reference_solution(u0, self.init_val_name, True)
            assert (
                read_ok
            ), f"ERROR: Could not read initial value from file {str(Path(self.output_folder) / Path(self.init_val_name).with_suffix(".npy"))}"

        return u0

    def compute_errors(self, uh):
        """
        Compute L2 error of uh[0] (potential V)
        Args:
            uh (VectorOfVectors): solution as vector of vectors

        Returns:
            computed (bool): if error computation was successful
            error (float): L2 error
            rel_error (float): relative L2 error
        """
        ref_sol_V = self.vector_type(init=self.init, val=0.0)
        read_ok = self.read_reference_solution([ref_sol_V], self.ref_sol, False)
        if read_ok:
            error_L2, rel_error_L2 = self.parabolic.compute_errors(uh[0], ref_sol_V)

            if self.comm.rank == 0:
                print(f"L2-errors: {error_L2}")
                print(f"Relative L2-errors: {rel_error_L2}")

            return True, error_L2, rel_error_L2
        else:
            self.logger.debug("Could not read reference solution from file. So errors are not computed.")
            return False, 0.0, 0.0

    def getSize(self):
        # return number of dofs in the mesh
        return self.parabolic.getSize()

    def eval_on_points(self, u):
        # evaluate the solution on a set of points (points are predefined/hardcoded in self.parabolic)
        return self.parabolic.eval_on_points(u)

    def define_ionic_model(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.parabolic.Cm

        if self.ionic_model_name in ["HodgkinHuxley", "HH"]:
            self.ionic_model = ionicmodels.HodgkinHuxley(self.scale_im)
        elif self.ionic_model_name in ["Courtemanche1998", "CRN"]:
            self.ionic_model = ionicmodels.Courtemanche1998(self.scale_im)
        elif self.ionic_model_name in ["TenTusscher2006_epi", "TTP"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi(self.scale_im)
        elif self.ionic_model_name in ["TTP_S", "TTP_SMOOTH"]:
            self.ionic_model = ionicmodels.TenTusscher2006_epi_smooth(self.scale_im)
        elif self.ionic_model_name in ["BiStable", "BS"]:
            self.ionic_model = ionicmodels.BiStable(self.scale_im)
        else:
            raise Exception("Unknown ionic model.")

        self.size = self.ionic_model.size

    def define_stimulus(self):
        self.scale_Iion = 0.01  # used to convert currents in uA/cm^2 to uA/mm^2
        # scale_im is applied to the rhs of the ionic model, so that the rhs is in units of mV/ms
        self.scale_im = self.scale_Iion / self.parabolic.Cm

        stim_dur = 2.0
        if self.apply_stimulus:
            if "cuboid" in self.parabolic.domain_name:
                self.stim_protocol = [[0.0, stim_dur]]  # list of stimulus time and stimulus duration
                self.stim_intensities = [50.0]  # list of stimulus intensities
                self.stim_centers = [[0.0, 0.0, 0.0]]  # list of stimulus centers
                r = 1.5  # stimulus radius, here uniform for all stimuli
                self.stim_radii = [[r, r, r]] * len(self.stim_protocol)  # list of stimulus radii
            elif "cube" in self.parabolic.domain_name:
                self.stim_protocol = [[0.0, 2.0], [570.0, 10.0]]
                self.stim_intensities = [50.0, 80.0]
                centers = [[0.0, 50.0, 50.0], [58.5, 0.0, 50.0]]
                self.stim_centers = [centers[i] for i in range(len(self.stim_protocol))]
                self.stim_radii = [[1.0, 50.0, 50.0], [1.5, 60.0, 50.0]]
            elif "03_fastl_LA" == self.parabolic.domain_name:
                stim_dur = 4.0
                stim_interval = [0, 280, 170, 160, 155, 150, 145, 140, 135, 130, 126, 124, 124, 124, 124]
                stim_times = np.cumsum(stim_interval)
                self.stim_protocol = [[stim_time, stim_dur] for stim_time in stim_times]
                self.stim_intensities = [80.0] * len(self.stim_protocol)
                centers = [[29.7377, 17.648, 45.8272], [60.5251, 27.9437, 41.0176]]
                self.stim_centers = [centers[i % 2] for i in range(len(self.stim_protocol))]
                r = 5.0
                self.stim_radii = [[r, r, r]] * len(self.stim_protocol)

            self.stim_protocol = np.array(self.stim_protocol)

            self.last_stim_index = -1

    def eval_f(self, u, t, fh=None):
        if fh is None:
            # create memory space for fh, if not already provided. For performance reasons, it is better to provide fh
            fh = self.dtype_f(init=self.init, val=0.0)

        # eval ionic model rhs on u and put result in fh. All indices of the super vector fh must be computed (list(range(self.size)) (see later other cases)
        self.eval_expr(self.ionic_model.f, u, fh, list(range(self.size)), False)
        # apply stimulus protocol
        if self.apply_stimulus:
            fh.val_list[0] += self.Istim(t)

        # eval diffusion by adding the laplacian of u[0] to fh[0]
        self.parabolic.add_disc_laplacian(u[0], fh[0])

        return fh

    def Istim(self, t):
        tol = 1e-8
        for i, (stim_time, stim_dur) in enumerate(self.stim_protocol):
            if (t + stim_dur * tol >= stim_time) and (t + stim_dur * tol < stim_time + stim_dur):
                if i != self.last_stim_index:
                    self.last_stim_index = i
                    # the stimuls region is given by a vector of zeros and ones, where the ones are the spatial locations to be stimulated
                    self.space_stim = self.parabolic.stim_region(self.stim_centers[i], self.stim_radii[i])
                    # Multiply be the stimulus intensity. The stimulus intensity is also multiplied by scale_im to convert it to uA/mm^2 (as the rhs of the ionic model from Myokit is in another unit)
                    self.space_stim *= self.scale_im * self.stim_intensities[i]
                return self.space_stim

        # if no stimulus is applied at this time t, return zero
        return self.parabolic.zero_stim_vec

    def eval_expr(self, expr, u, fh, indeces, zero_untouched_indeces=True):
        # evaluate a C++ function expr on the super vector u and put the result in the super vector fh
        # indeces indicate which of the indeces in the super vector fh will be modified (indeces depends on expr)
        # zero_untouched_indeces indicates if the untouched indeces in fh must be zeroed
        if expr is not None:
            expr(u.np_list, fh.np_list)

        if zero_untouched_indeces:
            non_indeces = [i for i in range(self.size) if i not in indeces]
            for i in non_indeces:
                fh[i].zero()

    def apply_mass_matrix(self, x, y=None):
        # computes y = M x on parabolic part and not on ionic model part. Hence only on index 0
        if y is None:
            y = x.copy()
        else:
            # we copy x to the solution y because the mass matrix is applied only on the parabolic part (index 0)
            # for the other indeces we have a pure ODE and the mass matrix is the identity, so a simple copy is enough
            y.copy(x)

        self.parabolic.apply_mass_matrix(x.val_list[0], y.val_list[0])

        return y


class MultiscaleMonodomainODE(MonodomainODE):
    def __init__(self, **problem_params):
        super(MultiscaleMonodomainODE, self).__init__(**problem_params)

        def dtype_f(init, val=None):
            return IMEXEXP_VectorOfVectors(init, val, self.vector_type, self.size)

        self.dtype_f = dtype_f

        self.define_splittings()
        self.parabolic.define_solver()

        self.constant_lambda_and_phi = False

    def rho_nonstiff(self, y, t, fy=None):
        return self.rho_nonstiff_cte

    def define_splittings(self):
        # Here we define different splittings of the rhs into stiff, nonstiff and exponential terms

        if self.splitting == "fast_slow":
            # this is a splitting to be used in multirate explicit stabilized methods. We use it for the mES schemes as mRKC.
            # define nonstiff
            self.im_f_nonstiff = self.ionic_model.f_nonstiff
            self.im_nonstiff_args = self.ionic_model.f_nonstiff_args
            self.im_nonstiff_indeces = self.ionic_model.f_nonstiff_indeces
            # define stiff
            self.im_f_stiff = self.ionic_model.f_stiff
            self.im_stiff_args = self.ionic_model.f_stiff_args
            self.im_stiff_indeces = self.ionic_model.f_stiff_indeces
            # define exp
            self.im_lmbda_exp = None
            self.im_lmbda_yinf_exp = None
            self.im_exp_args = []
            self.im_exp_indeces = []

            self.rho_nonstiff_cte = self.ionic_model.rho_f_nonstiff()

        elif self.splitting == "fast_slow_exponential":
            # this is the standard splitting used in Rush-Larsen methods. We use it for the IMEX-RL and emRKC schemes.
            # define nonstiff.
            self.im_f_nonstiff = self.ionic_model.f_expl
            self.im_nonstiff_args = self.ionic_model.f_expl_args
            self.im_nonstiff_indeces = self.ionic_model.f_expl_indeces
            # define stiff
            self.im_f_stiff = (
                None  # no stiff part coming from ionic model (everything stiff is in the exponential part)
            )
            self.im_stiff_args = []
            self.im_stiff_indeces = []
            # define exp
            self.im_lmbda_exp = self.ionic_model.lmbda_exp
            self.im_lmbda_yinf_exp = self.ionic_model.lmbda_yinf_exp
            self.im_exp_args = self.ionic_model.f_exp_args
            self.im_exp_indeces = self.ionic_model.f_exp_indeces

            self.rho_nonstiff_cte = self.ionic_model.rho_f_expl()

        else:
            raise Exception("Unknown splitting.")

        self.rhs_stiff_args = self.im_stiff_args
        self.rhs_stiff_indeces = self.im_stiff_indeces
        if 0 not in self.rhs_stiff_args:
            self.rhs_stiff_args = [0] + self.rhs_stiff_args
        if 0 not in self.rhs_stiff_indeces:
            self.rhs_stiff_indeces = [0] + self.rhs_stiff_indeces

        self.rhs_nonstiff_args = self.im_nonstiff_args
        self.rhs_nonstiff_indeces = self.im_nonstiff_indeces
        if 0 not in self.rhs_nonstiff_indeces:
            self.rhs_nonstiff_indeces = [0] + self.rhs_nonstiff_indeces

        self.rhs_exp_args = self.im_exp_args
        self.rhs_exp_indeces = self.im_exp_indeces
        self.im_non_exp_indeces = [i for i in range(self.size) if i not in self.im_exp_indeces]

        self.one = self.dtype_u(init=self.init, val=1.0)

        self.lmbda = self.dtype_u(init=self.init, val=0.0)
        self.yinf = self.dtype_u(init=self.init, val=0.0)

    def eval_lmbda_yinf_exp(self, u, lmbda, yinf):
        self.im_lmbda_yinf_exp(u.np_list, lmbda.np_list, yinf.np_list)

    def eval_lmbda_exp(self, u, lmbda):
        self.im_lmbda_exp(u.np_list, lmbda.np_list)

    def solve_system(self, rhs, factor, u0, t, u_sol=None):
        if u_sol is None:
            u_sol = self.dtype_u(init=self.init, val=0.0)

        self.parabolic.solve_system(rhs[0], factor, u0[0], t, u_sol[0])

        if rhs is not u_sol:
            for i in range(1, self.size):
                u_sol[i].copy(rhs[i])

        return u_sol

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None, zero_untouched_indeces=True):
        if fh is None:
            fh = self.dtype_f(init=self.init, val=0.0)

        u.ghostUpdate(addv="insert", mode="forward")

        if eval_expl:  # slow
            fh.expl = self.eval_f_nonstiff(u, t, fh.expl, zero_untouched_indeces)

        if eval_impl:  # fast
            fh.impl = self.eval_f_stiff(u, t, fh.impl, zero_untouched_indeces)

        if eval_exp:  # exponential
            fh.exp = self.eval_f_exp(u, t, fh.exp, zero_untouched_indeces)

        return fh

    def eval_f_nonstiff(self, u, t, fh_nonstiff, zero_untouched_indeces=True):
        # eval ionic model nonstiff terms
        self.eval_expr(self.im_f_nonstiff, u, fh_nonstiff, self.im_nonstiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_nonstiff_indeces:
            fh_nonstiff[0].zero()

        # apply stimulus
        if self.apply_stimulus:
            fh_nonstiff.val_list[0] += self.Istim(t)

        return fh_nonstiff

    def eval_f_stiff(self, u, t, fh_stiff, zero_untouched_indeces=True):
        # eval ionic model stiff terms
        self.eval_expr(self.im_f_stiff, u, fh_stiff, self.im_stiff_indeces, zero_untouched_indeces)

        if not zero_untouched_indeces and 0 not in self.im_stiff_indeces:
            fh_stiff[0].zero()

        # eval diffusion
        self.parabolic.add_disc_laplacian(u[0], fh_stiff[0])

        return fh_stiff

    def eval_f_exp(self, u, t, fh_exp, zero_untouched_indeces=True):
        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            fh_exp.np_array(i)[:] = self.lmbda.np_array(i) * (u.np_array(i) - self.yinf.np_array(i))

        if zero_untouched_indeces:
            fh_exp.zero_sub(self.im_non_exp_indeces)

        return fh_exp

    def eval_phi_f_exp(self, u, factor, t, phi_f_exp=None, zero_untouched_indeces=True):
        if phi_f_exp is None:
            phi_f_exp = self.dtype_u(init=self.init, val=0.0)

        self.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        for i in self.im_exp_indeces:
            phi_f_exp.np_array(i)[:] = (
                (np.exp(factor * self.lmbda.np_array(i)) - 1.0) / (factor) * (u.np_array(i) - self.yinf.np_array(i))
            )

        if zero_untouched_indeces:
            phi_f_exp.zero_sub(self.im_non_exp_indeces)

        return phi_f_exp
