import numpy as np
import dolfinx
from dolfinx import fem, mesh, geometry, io
from dolfinx.fem import FunctionSpace, VectorFunctionSpace
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import adios4dolfinx
import basix
import h5py
from tqdm import tqdm
from pathlib import Path
import json


def print_MPI(message):
    if MPI.COMM_WORLD.rank == 0:
        print(message)


class fibrosis:
    def __init__(self, **params):

        self.params = params
        self.fibers_folder = params["fibers_folder"]
        self.fibrosis_folder = params["fibrosis_folder"]
        self.K = params["K"]
        self.percents = params["percents"]

        np.random.seed(params["seed"])

        if "cuboid" not in params["domain_name"] and "cube" not in params["domain_name"]:
            # we import a mesh
            self.import_mesh = True
            self.dim = 3
        else:  # we generate a rectangular domain on the fly
            self.import_mesh = False
            if "cuboid" in params["domain_name"]:
                if "small" in params["domain_name"]:
                    self.dom_size = [[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]]
                    self.n_elems = 25 * 2 ** params["refinement"]
                else:
                    self.dom_size = [[0.0, 0.0, 0.0], [20.0, 7.0, 3.0]]
                    self.n_elems = 100 * 2 ** params["refinement"]
                self.dim = int(params["domain_name"][7])
            elif "cube" in params["domain_name"]:
                self.dom_size = [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]
                self.n_elems = 250 * 2 ** params["refinement"]
                self.dim = int(params["domain_name"][5])

        self.domain_name = Path(params["domain_name"])
        self.mesh_refinement = Path("ref_" + str(params["refinement"]))
        self.in_folder = self.fibers_folder / self.domain_name / self.mesh_refinement
        self.out_folder = self.fibrosis_folder / self.domain_name / self.mesh_refinement

        self.define_domain()

        self.V = FunctionSpace(self.domain, ("CG", 1))
        self.V_fiber = VectorFunctionSpace(self.domain, ("CG", 1))

        self.define_fibers()

        self.diff_f0 = 1.0
        self.diff_s0 = 1e-2
        self.diff_n0 = 1e-2
        self.rho_list = self.params["rho_list"]

        self.xdmf = io.XDMFFile(self.domain.comm, self.out_folder / Path("fibrosis.xdmf"), "w")
        self.xdmf.write_mesh(self.domain)

    def __del__(self):
        self.xdmf.close()

    def define_domain(self):
        if self.import_mesh:
            with io.XDMFFile(MPI.COMM_WORLD, self.in_folder / Path("fibers.xdmf"), "r") as xdmf:
                self.domain = xdmf.read_mesh(name="mesh", xpath="Xdmf/Domain/Grid")
        else:
            n_elems = self.n_elems
            dom_size = self.dom_size
            d = np.asarray(dom_size[1]) - np.asarray(dom_size[0])
            max_d = np.max(d)
            n = [n_elems] * self.dim
            for i in range(len(n)):
                n[i] = int(np.ceil(n[i] * d[i] / max_d))

            if self.dim == 1:
                self.domain = mesh.create_interval(
                    comm=MPI.COMM_WORLD, nx=n_elems, points=[dom_size[0][0], dom_size[1][0]]
                )
            elif self.dim == 2:
                self.domain = mesh.create_rectangle(
                    comm=MPI.COMM_WORLD,
                    n=n,
                    cell_type=mesh.CellType.triangle,
                    points=[dom_size[0][: self.dim], dom_size[1][: self.dim]],
                )
            elif self.dim == 3:
                self.domain = mesh.create_box(
                    comm=MPI.COMM_WORLD,
                    n=n,
                    cell_type=mesh.CellType.tetrahedron,
                    points=[dom_size[0][: self.dim], dom_size[1][: self.dim]],
                )
            else:
                raise Exception(f"need dim=1,2,3 to instantiate problem, got dim={self.dim}")

            # with io.XDMFFile(self.domain.comm, self.out_folder / Path("fibers.xdmf"), "w") as xdmf:
            #     xdmf.write_mesh(self.domain)

    def define_fibers(self):
        if self.import_mesh:
            fixed_fibers_path = self.in_folder / Path("fibers_fixed_f0")
            if fixed_fibers_path.is_dir():
                print_MPI("Importing fixed fibers...")
                self.f0 = self.import_fixed_fiber(self.in_folder / Path("fibers_fixed_f0"))
                self.s0 = self.import_fixed_fiber(self.in_folder / Path("fibers_fixed_s0"))
                self.n0 = self.import_fixed_fiber(self.in_folder / Path("fibers_fixed_n0"))
            else:
                print_MPI("Importing fibers...")
                self.f0 = self.read_mesh_data(self.in_folder / Path("fibers.h5"), self.domain, "f0")
                self.s0 = self.read_mesh_data(self.in_folder / Path("fibers.h5"), self.domain, "s0")
                self.n0 = self.read_mesh_data(self.in_folder / Path("fibers.h5"), self.domain, "n0")
        else:
            print_MPI("Defining fibers for cuboid or cube...")
            e1 = np.array([[1.0], [0.0], [0.0]])
            e2 = np.array([[0.0], [1.0], [0.0]])
            e3 = np.array([[0.0], [0.0], [1.0]])

            def cte_dom(x, cte):
                b = np.full(x.shape, cte)
                return b[: self.dim, :]

            self.f0, self.s0, self.n0 = (
                fem.Function(self.V_fiber),
                fem.Function(self.V_fiber),
                fem.Function(self.V_fiber),
            )
            self.f0.name = "f0"
            self.s0.name = "s0"
            self.n0.name = "n0"
            self.f0.interpolate(lambda x: cte_dom(x, e1))
            self.s0.interpolate(lambda x: cte_dom(x, e2))
            self.n0.interpolate(lambda x: cte_dom(x, e3))

        # xdmf = io.XDMFFile(self.domain.comm, self.in_folder / Path("fibers_f0.xdmf"), "w")
        # xdmf.write_mesh(self.domain)
        # xdmf.write_function(self.f0)
        # xdmf.close()
        # exit()

    def import_fixed_fiber(self, input_folder):
        domain_r = adios4dolfinx.read_mesh(self.domain.comm, input_folder, "BP4", dolfinx.mesh.GhostMode.shared_facet)
        el = ufl.VectorElement("CG", self.domain.ufl_cell(), 1)
        V_r = fem.FunctionSpace(domain_r, el)
        fib_r = fem.Function(V_r)
        adios4dolfinx.read_function(fib_r, input_folder, "BP4")
        fib = fem.Function(self.V_fiber)
        fib.interpolate(
            fib_r,
            nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                fib.function_space.mesh._cpp_object, fib.function_space.element, fib_r.function_space.mesh._cpp_object
            ),
        )
        fib.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return fib

    def read_mesh_data(self, file_path: Path, mesh: dolfinx.mesh.Mesh, data_path: str):
        # see https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/48
        assert file_path.is_file(), f"File {file_path} does not exist"
        infile = h5py.File(file_path, "r", driver="mpio", comm=mesh.comm)
        num_nodes_global = mesh.geometry.index_map().size_global
        assert data_path in infile.keys(), f"Data {data_path} does not exist"
        dataset = infile[data_path]
        shape = dataset.shape
        assert shape[0] == num_nodes_global, f"Got data of shape {shape}, expected {num_nodes_global, shape[1]}"
        # dtype = dataset.dtype
        # Read data locally on each process
        local_input_range = adios4dolfinx.utils.compute_local_range(mesh.comm, num_nodes_global)
        local_input_data = dataset[local_input_range[0] : local_input_range[1]]

        # Create appropriate function space (based on coordinate map)
        assert len(mesh.geometry.cmaps) == 1, "Mixed cell-type meshes not supported"
        element = basix.ufl.element(
            basix.ElementFamily.P,
            mesh.topology.cell_name(),
            mesh.geometry.cmaps[0].degree,
            mesh.geometry.cmaps[0].variant,
            shape=(shape[1],),
            gdim=mesh.geometry.dim,
        )

        # Assumption: Same doflayout for geometry and function space, cannot test in python
        V = dolfinx.fem.FunctionSpace(mesh, element)
        uh = dolfinx.fem.Function(V, name=data_path)
        # Assume that mesh is first order for now
        assert mesh.geometry.cmaps[0].degree == 1, "Only linear meshes supported"
        x_dofmap = mesh.geometry.dofmap
        igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)
        global_geom_input = igi[x_dofmap]
        global_geom_owner = adios4dolfinx.utils.index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
        for i in range(shape[1]):
            arr_i = adios4dolfinx.comm_helpers.send_dofs_and_recv_values(
                global_geom_input.reshape(-1),
                global_geom_owner,
                mesh.comm,
                local_input_data[:, i],
                local_input_range[0],
            )
            dof_pos = x_dofmap.reshape(-1) * shape[1] + i
            uh.x.array[dof_pos] = arr_i
        infile.close()
        return uh

    def define_variational_forms(self):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        self.mass = u * v * ufl.dx
        self.diff = (
            ufl.dot(
                self.diff_f0 * self.f0 * ufl.dot(self.f0, ufl.grad(u))
                + self.diff_s0 * self.s0 * ufl.dot(self.s0, ufl.grad(u))
                + self.diff_n0 * self.n0 * ufl.dot(self.n0, ufl.grad(u)),
                ufl.grad(v),
            )
            * ufl.dx
            + self.kappa**2 * u * v * ufl.dx
        )

        self.mass_form = fem.form(self.mass)
        self.diff_form = fem.form(self.diff)

    def assemble_vec_mat(self):
        from dolfinx.fem.petsc import assemble_matrix

        self.diff_mat = fem.petsc.assemble_matrix(self.diff_form)
        self.diff_mat.assemble()

        one = fem.Function(self.V)
        one.interpolate(lambda x: 1.0 + 0.0 * x[0])
        mass_lumped = ufl.action(self.mass, one)
        self.ml = fem.Function(self.V)
        with self.ml.vector.localForm() as m_loc:
            m_loc.set(0)
        fem.petsc.assemble_vector(self.ml.vector, fem.form(mass_lumped))
        self.ml.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.ml.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.diff_mat)

    def generate_fibrosis_field(self):
        write_all_rho = self.params["write_all_rho"]
        self.u = fem.Function(self.V)
        self.u.x.array[:] *= 0.0
        k = 0
        for rho in self.rho_list:
            self.nu = 2.0 * self.K - self.dim / 2.0
            self.rho = rho
            self.kappa = 2 * np.sqrt(self.nu) / self.rho
            self.define_variational_forms()
            self.assemble_vec_mat()
            u_rho = self.generate_one_field()
            self.u.x.array[:] += u_rho.x.array
            if write_all_rho:
                self.xdmf.write_function(u_rho, k)
            k += 1

        self.u.x.array[:] /= len(self.rho_list)
        self.normalize(self.u)

        if write_all_rho:
            self.xdmf.write_function(self.u, k)
        else:
            self.xdmf.write_function(self.u)

        adios4dolfinx.write_mesh(self.domain, self.out_folder / Path("fibrosis.bp"), engine="BP4")
        adios4dolfinx.write_function(self.u, self.out_folder / Path("fibrosis.bp"), engine="BP4")

    def generate_one_field(self):
        basis = np.sqrt(1.0 / self.ml.x.array)

        u = [fem.Function(self.V) for _ in range(0, self.K + 1)]
        u[0].x.array[:] = basis * np.random.normal(0.0, 1.0, basis.shape[0])
        u[0].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.normalize(u[0])

        rhs = fem.Function(self.V)
        pbar = tqdm(total=self.K)
        for k in range(1, self.K + 1):
            rhs.vector.pointwiseMult(self.ml.vector, u[k - 1].vector)
            self.solver.solve(rhs.vector, u[k].vector)
            u[k].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            self.normalize(u[k])
            pbar.update(1)
        pbar.close()

        return u[-1]

    def normalize(self, u):
        ind, u_min = u.vector.max()
        ind, u_max = u.vector.min()
        u.x.array[:] = (u.x.array - u_min) / (u_max - u_min)

    def compute_quantiles(self):
        tol = 1e-3
        max_it = 100
        if self.params["write_quantile_fun"]:
            self.xdmf = io.XDMFFile(self.domain.comm, self.out_folder / Path("fibrosis_quant_fun.xdmf"), "w")
            self.xdmf.write_mesh(self.domain)
        one = fem.Constant(self.domain, 1.0)
        dom_vol = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM)
        perc = self.percents
        quantiles = []
        pi = 0
        for p in perc:
            pi += 1
            q_found = False
            qb = [0.0, 1.0]
            it = 0
            while not q_found and it < max_it:
                it += 1
                q = np.average(qb)
                fq = ufl.conditional(ufl.lt(self.u, q), 1.0, 0.0)
                p_vol = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(fq * ufl.dx)), op=MPI.SUM) / dom_vol
                print_MPI(f"Iter {it} for P = {p}, q = {q}, p_vol = {p_vol}")
                if abs(p_vol - p) < tol:
                    print_MPI(f"Found: for P = {p}, quantile = {q}")
                    quantiles.append(q)
                    q_found = True
                    fqi = fem.Function(self.V)
                    fqi.interpolate(fem.Expression(fq, self.V.element.interpolation_points()))
                    if self.params["write_quantile_fun"]:
                        self.xdmf.write_function(fqi, float(pi))
                elif p_vol < p:
                    qb[0] = q
                else:
                    qb[1] = q

        if self.params["write_quantile_fun"]:
            self.xdmf.close()

        res = {"perc": perc, "quantiles": quantiles}
        with open(self.out_folder / Path("quantiles.json"), "w") as outfile:
            json.dump(res, outfile)


def main():
    """
    Generate fibrosis field for any of the meshes found in ./meshes/mesh. Note that the corresponding fibers must be available in ./fibers.
    Other possible domain_names are: "cuboid_nD_small", "cuboid_nD", "cube_nD", with n=1,2,3. In these cases a mesh is generated on the fly.

    Run as:
    docker exec -ti -w /src/meshes_fibers_fibrosis emRKC mpirun -n 12 python3 generate_fibrosis_field.py

    Or on daint:
    . spack/share/spack/setup-env.sh && module load daint-gpu spack-config && spack env activate fenicsx_new
    cd $SCRATCH/pySDC_and_Stabilized_in_FeNICSx
    srun --ntasks=12 --ntasks-per-node=3 --pty -C gpu -A s1074 --time=00:30:00 python utils/generate_fibrosis_field.py
    """

    params = dict()
    params["fibers_folder"] = Path("./fibers")
    params["fibrosis_folder"] = Path("./fibrosis")
    params["domain_name"] = "idealized_LV"
    params["refinement"] = 0
    params["K"] = 2
    params["rho_list"] = [4.0, 20.0]
    # the percents for which we compute the percentiles (usually only the median)
    params["percents"] = [0.5]

    # we usually generate more fields, for different rho values in rho_list and then average them
    # to write all generated fields set write_all_rho to True. The averaged field is written as the last one
    # set it to False to write only the averaged field, which is the one used in simulations
    params["write_all_rho"] = False

    # write the inidcator function indicating where the random field is below or above a certain percentile
    params["write_quantile_fun"] = False

    # seed for random number generator.
    params["seed"] = 1989

    fibr = fibrosis(**params)
    fibr.generate_fibrosis_field()
    fibr.compute_quantiles()


if __name__ == "__main__":
    main()
