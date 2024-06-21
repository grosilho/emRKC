# Content
These folders contain the necessary files for performing FEM simulations on unstructured grids. The mesh is readen from the fibers file, so no separate mesh files are needed. On simple domains with structured grids (the cube and cuboids), fibers are not necessary since the code generates meshes and fibers on the fly. For the DCT spatial discretization they are not needed neither.

## Fibers
The `./fibers` folder contains the fibers generated with the _lifex-fiber_ package [[1]](#1). We used the meshes already provided in the lifex_fiber_generation_examples.zip archive found here [[1]](#1). However, we refined some meshes and removed other ones. 

If you use _lifex-fiber_ on your mesh, be aware that for some meshes the _lifex-fiber_'s generated fibers might contain a few NaNs. In this case run the `fix_fibers.py` script on those fibers before performing the simulation. Basicaly, the script looks for NaNs and replaces them with an interpolation of neighboring values.

## Fibrosis
The `./fibrosis` folder contains the fibrosis fields for the meshes and fibers in `./fibers` and as well for structured simple domains (such as cube and cuboid), for which fibers are not needed. New fibrosis fields can be generated for other _lifex-fiber_'s fibers or strucutred domains with the `generate_fibrosis.py` script. The algorithm employed in the script is taken from [[2]](#2). If using _lifex-fiber_'s fibers  and the script crashes or you see some numeric errors, try to run the `fix_fibers.py` script on those fibers before executing `generate_fibrosis.py`.

# References
<a id="1">[1]</a> Africa, P. C., Piersanti, R., & Fedele, M. (2022). lifex-fiber: an open tool for myofibers generation in cardiac computational models (1.4.0). Zenodo. [https://doi.org/10.5281/zenodo.7622070](https://doi.org/10.5281/zenodo.7622070)

<a id="2">[2]</a> Pezzuto, S., Quaglino, A., & Potse, M. (2019). On Sampling Spatially-Correlated Random Fields for Complex Geometries. In Y. Coudière, V. Ozenne, E. Vigmond, & N. Zemzemi (A c. Di), Functional Imaging and Modeling of the Heart (Vol. 11504, pp. 103–111). Springer International Publishing. [https://doi.org/10.1007/978-3-030-21949-9_12](https://doi.org/10.1007/978-3-030-21949-9_12)

