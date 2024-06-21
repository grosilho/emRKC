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


class fix_fibers:
    def __init__(self, **params):
        self.params = params
        self.fibers_folder = params["fibers_folder"]

        self.domain_name = Path(params["domain_name"])
        self.mesh_refinement = Path("ref_" + str(params["refinement"]))
        self.inout_folder = self.fibers_folder / self.domain_name / self.mesh_refinement

        print_MPI("Importing domain...")
        self.define_domain()

        self.V = FunctionSpace(self.domain, ("CG", 1))
        self.V_fiber = VectorFunctionSpace(self.domain, ("CG", 1))

        self.V_DG0 = FunctionSpace(self.domain, ("DG", 0))
        self.V_DG0_fiber = VectorFunctionSpace(self.domain, ("DG", 0))

        print_MPI("Importing fibers...")
        self.define_fibers()

    def define_domain(self):
        with io.XDMFFile(MPI.COMM_WORLD, self.inout_folder / Path("fibers.xdmf"), "r") as xdmf:
            self.domain = xdmf.read_mesh(name="mesh", xpath="Xdmf/Domain/Grid")

    def read_mesh_data(self, file_path: Path, mesh: dolfinx.mesh.Mesh, data_path: str):
        # see https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/48
        assert file_path.is_file(), f"File {file_path} does not exist"
        infile = h5py.File(file_path, "r", driver="mpio", comm=mesh.comm)
        num_nodes_global = mesh.geometry.index_map().size_global
        assert data_path in infile.keys(), f"Data {data_path} does not exist"
        dataset = infile[data_path]
        shape = dataset.shape
        assert shape[0] == num_nodes_global, f"Got data of shape {shape}, expected {num_nodes_global, shape[1]}"
        dtype = dataset.dtype
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
            uh.x.array[dof_pos] = np.nan_to_num(arr_i)
        infile.close()

        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return uh

    def define_fibers(self):
        self.f0 = self.read_mesh_data(self.inout_folder / Path("fibers.h5"), self.domain, "f0")
        self.s0 = self.read_mesh_data(self.inout_folder / Path("fibers.h5"), self.domain, "s0")
        self.n0 = self.read_mesh_data(self.inout_folder / Path("fibers.h5"), self.domain, "n0")

    def fix_fibers(self):
        print_MPI("Building mass matrices...")
        from dolfinx.fem.petsc import assemble_matrix

        u = ufl.TrialFunction(self.V_fiber)
        v = ufl.TestFunction(self.V_fiber)
        mass = ufl.dot(u, v) * ufl.dx
        mass_form = fem.form(mass)
        self.M = fem.petsc.assemble_matrix(mass_form)
        self.M.assemble()

        u0 = ufl.TrialFunction(self.V_DG0_fiber)
        v0 = ufl.TestFunction(self.V_DG0_fiber)
        mass0 = ufl.dot(u0, v0) * ufl.dx
        mass0_form = fem.form(mass0)
        self.M0 = fem.petsc.assemble_matrix(mass0_form)
        self.M0.assemble()

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.M)
        self.solver.setType(PETSc.KSP.Type.BCGS)
        pc = self.solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        self.solver0 = PETSc.KSP().create(self.domain.comm)
        self.solver0.setOperators(self.M0)
        self.solver0.setType(PETSc.KSP.Type.BCGS)
        pc0 = self.solver0.getPC()
        pc0.setType(PETSc.PC.Type.HYPRE)
        pc0.setHYPREType("boomeramg")

        print_MPI("Fixing fiber f0...")
        self.f0 = self.fix_fiber(self.f0)
        print_MPI("Fixing fiber s0...")
        self.s0 = self.fix_fiber(self.s0)
        print_MPI("Fixing fiber n0...")
        self.n0 = self.fix_fiber(self.n0)

    def fix_fiber(self, fib):
        v_DG = ufl.TestFunction(self.V_DG0_fiber)
        rhs0 = ufl.dot(fib, v_DG) * ufl.dx
        rhs0_form = fem.form(rhs0)
        F0 = fem.petsc.create_vector(rhs0_form)
        fem.petsc.assemble_vector(F0, rhs0_form)
        F0.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        F0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        fib0 = fem.Function(self.V_DG0_fiber)
        self.solver0.solve(F0, fib0.vector)
        fib0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        v = ufl.TestFunction(self.V_fiber)
        rhs = ufl.dot(fib0, v) * ufl.dx
        rhs_form = fem.form(rhs)
        F = fem.petsc.create_vector(rhs_form)
        fem.petsc.assemble_vector(F, rhs_form)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        fib_from_DG = fem.Function(self.V_fiber)
        self.solver.solve(F, fib_from_DG.vector)
        fib_from_DG.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        fib_new = fem.Function(self.V_fiber)
        fib_new = fib.copy()
        zero_ind = np.where(np.isclose(fib_new.x.array, 0.0))[0]
        fib_new.x.array[zero_ind] = fib_from_DG.x.array[zero_ind]

        fib_new_normalized = fem.Function(self.V_fiber)
        fib_new_normalized.interpolate(
            fem.Expression(fib_new / ufl.sqrt(ufl.dot(fib_new, fib_new)), self.V_fiber.element.interpolation_points())
        )

        return fib_new_normalized

    def write_fibers(self):
        print_MPI("Writing fibers...")
        self.f0.name = "f0"
        self.s0.name = "s0"
        self.n0.name = "n0"

        adios4dolfinx.write_mesh(self.domain, self.inout_folder / Path("fibers_fixed_f0"), "BP4")
        adios4dolfinx.write_function(self.f0, self.inout_folder / Path("fibers_fixed_f0"), "BP4")

        adios4dolfinx.write_mesh(self.domain, self.inout_folder / Path("fibers_fixed_s0"), "BP4")
        adios4dolfinx.write_function(self.s0, self.inout_folder / Path("fibers_fixed_s0"), "BP4")

        adios4dolfinx.write_mesh(self.domain, self.inout_folder / Path("fibers_fixed_n0"), "BP4")
        adios4dolfinx.write_function(self.n0, self.inout_folder / Path("fibers_fixed_n0"), "BP4")

        # check
        # domain_r = adios4dolfinx.read_mesh(self.domain.comm, self.inout_folder / Path("fibers_fixed_f0"), "BP4", dolfinx.mesh.GhostMode.shared_facet)
        # el = ufl.VectorElement('CG', self.domain.ufl_cell(), 1)
        # V_r = dolfinx.fem.FunctionSpace(domain_r, el)
        # f0_r = dolfinx.fem.Function(V_r)
        # adios4dolfinx.read_function(f0_r, self.inout_folder / Path("fibers_fixed_f0"), "BP4")
        # f0_new = fem.Function(self.V_fiber)
        # f0_new.interpolate(f0_r,nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        #                                                                         f0_new.function_space.mesh._cpp_object,
        #                                                                         f0_new.function_space.element,
        #                                                                         f0_r.function_space.mesh._cpp_object))
        # f0_new.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # error_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(f0_new - self.f0,f0_new - self.f0) * ufl.dx)), op=MPI.SUM))
        # sol_norm_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(self.f0,self.f0) * ufl.dx)), op=MPI.SUM))
        # rel_error_L2 = error_L2/sol_norm_L2
        # if self.domain.comm.rank == 0:
        #     print(f"L2-errors: {error_L2}")
        #     print(f"Relative L2-errors: {rel_error_L2}")


def main():
    # Run with docker as:
    # docker exec -ti -w /src/meshes_fibers_fibrosis emRKC mpirun -n 12 python3 fix_fibers.py

    # Or on daint:
    # . spack/share/spack/setup-env.sh && module load daint-gpu spack-config && spack env activate fenicsx_new
    # cd $SCRATCH/pySDC_and_Stabilized_in_FeNICSx
    # srun --ntasks=12 --ntasks-per-node=3 --pty -C gpu -A s1074 --time=00:30:00 python utils/fix_fibers.py

    params = dict()
    params["fibers_folder"] = Path("./fibers")
    params["domain_name"] = "idealized_LV"
    params["refinement"] = 0

    fix = fix_fibers(**params)
    fix.fix_fibers()
    fix.write_fibers()


if __name__ == "__main__":
    main()
