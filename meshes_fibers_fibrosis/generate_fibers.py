import os
from pathlib import Path


"""
For a single run you can simply use/adapt these commands
    export DOMAIN=idealized_LV
    export REF=ref_0
    mkdir -p fibers/$DOMAIN/$REF
    mpirun -n 12 ./lifex_fiber_generation-1.4.0-x86_64.AppImage -f meshes/prm/$DOMAIN\_$REF.prm -o fibers/$DOMAIN/$REF | tee fibers/$DOMAIN/$REF/lifex_fiber_output.log

If you want to generate fibers for all the available meshes then just run/adapt this script.
"""


mesh_folder = Path("./meshes/mesh")
prm_folder = Path("meshes/prm")
fibers_folder = Path("fibers")
n_proc = 12


# Search for all files in the mesh_folder
mesh_files = [Path(f) for f in os.listdir(mesh_folder) if os.path.isfile(os.path.join(mesh_folder, f))]
# order files by size
mesh_files.sort(key=lambda x: os.path.getsize(mesh_folder / x))

max_size = 150  # MB
# remove files bigger than max_size
mesh_files = [f for f in mesh_files if os.path.getsize(mesh_folder / f) / 1024 / 1024 < max_size]

for mesh_file in mesh_files:
    mesh_name = str(mesh_file.stem)
    domain_name, level = mesh_name.split("_ref_")
    refinement = "ref_" + level
    output_folder = fibers_folder / Path(domain_name) / Path(refinement)
    output_file = output_folder / Path("fibers")
    # create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if os.path.isfile(output_file.with_suffix(".h5")) and os.path.isfile(output_file.with_suffix(".xdmf")):
        print("Fibers already generated for " + mesh_name)
        continue

    command = (
        "mpirun -n "
        + str(n_proc)
        + " ./lifex_fiber_generation-1.4.0-x86_64.AppImage"
        + " -f "
        + str(prm_folder / mesh_file.with_suffix(".prm"))
        + " -o "
        + str(output_folder)
        + " | tee "
        + str(output_folder / Path("lifex_fiber_output.log"))
    )

    print(command)
    # execute command and wait for it to finish
    os.system(command)
