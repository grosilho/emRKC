# emRKC
A Python implementation of the exponential multirate Runge-Kutta-Chebychev (emRKC) method with applications to the monodomain model in cardiac electrophysiology. This code has been developed and used to produce the numerical results in

> Rosilho De Souza, G. R., Grote, M. J., Pezzuto, S., & Krause, R. (2024). Explicit stabilized multirate methods for the monodomain model in cardiac electrophysiology. ESAIM: Mathematical Modelling and Numerical Analysis (in press). [http://arxiv.org/abs/2401.01745](http://arxiv.org/abs/2401.01745)


## The numerical scheme
The emRKC method combines multirate explicit stabilized methods and exponential integration. It is a first-order accurate fully explicit method without step size restrictions. In numerical experiments it has shown higher performance than the popular and efficient IMEX Rush-Larsen method[^imexrl].

### Target problem
emRKC solves ODEs of the type
$$y'=f_F(t,y)+f_S(t,y)+f_E(t,y), \quad y(0)=y_0,$$
where $f_F$ is a stiff but inexpensive term ($F$ for fast), $f_S$ is a nonstiff but expensive to evaluate term ($S$ for slow), and
$$f_E(t,y)=\Lambda(t,y)(y-y_\infty(t,y)),$$
is a very stiff term where $\Lambda$ is a matrix and $y_\infty$ a general nonlinear term. For efficiency reasons, $\Lambda$ should be diagonal. 

A particular and important application case reducing to such ODE is the semi-discrete monodomain equation, where $f_F$ represents the spatially discretized Laplacian, $f_E$ the ionic model's gating variables and $f_S$ all remaining terms.

### Method
emRKC is an extension of the multirate Runge-Kutta-Chebychev (mRKC) method[^mrkc] [^mrkccode] where the terms $f_F$ and $f_S$ are integrated exactly as in mRKC and the $f_E$ term is integrated exponentially. Hence, as mRKC, also emRKK solves a modified equation.

**The modified equation.**
More specifically, emRKC can be seen as a Runge-Kutta-Chebyshev (RKC) method applied to the modified equation
$$y_\eta'=f_\eta(t,y_\eta),\qquad y_\eta(0)=y_0,$$
with $f_\eta$ an approximation to $f_F+f_S+f_E$. The advantage is that the modified equation is such that the stiffness of $f_\eta$ depends on the slow terms $f_S$ only, and therefore solving  $y_\eta'=f_\eta(t,y_\eta)$ is significantly cheaper than $y'=f_F(t,y)+f_S(t,y)+f_E(t,y)$. Evaluating $f_\eta$ has a similar cost as $f_F+f_S+f_E$ and requires solving an auxiliary problem.

**The auxiliary problem.**
The averaged right-hand side $f_\eta(t,y_\eta)$ is defined by computing
$$y_E=y_\eta+\eta\varphi(\eta\Lambda(t,y_\eta))f_E(t,y_\eta),$$
where $\varphi(z)=(e^z-1)/z$, then solving an auxiliary problem
$$u'=f_F(u)+f_S(y_E) \quad t\in (0,\eta), \quad u(0)=y_E$$
and finally evaluating
$$f_\eta(y_\eta)=\frac{1}{\eta}(u(\eta)-y_\eta).$$
Note that $y_E$ is computed with an exponential Euler step and, in practice, the auxiliary problem is solved using an RKC method. The value of $\eta>0$ depends on the stiffness of $f_S$ and in general satisfies $\eta\ll\Delta t$, with $\Delta t$ the step size used to solve the modified equation $y_\eta'=f_\eta(t,y_\eta)$.
Since the expensive term $f_S$ is frozen, solving the auxiliary problem is comparable to evaluating $f_F+f_S+f_E$.

## The code
The code implements the emRKC, mRKC and the baseline IMEX Rush-Larsen method. As spatial discretization we offer finite differences evaluated with discrete cosine transforms (DCT) and the finite element method (FEM). For FEM we employ the [FEniCSx](https://fenicsproject.org/) package. 

### Using the DCT discretization
For runnning simulations using the DCT spatial discretization a simple conda or venv environment is enough. The environment.yml/requirements.txt files are provided in `/etc`. After setting up the environment, the C++ ionic models must be compiled. On linux do:
```shell
cd problem_classes/ionicmodels/cpp
c++ -O3 -Wall -shared -std=c++11 -fPIC -fvisibility=hidden $(python3 -m pybind11 --includes) bindings_definitions.cpp -o ionicmodels$(python3-config --extension-suffix)
```
On MacOS use the compilation command provided in `problem_classes/ionicmodels/cpp/compilation_commands.txt`.

Then, for running a simulation execute:
```shell
python3 Solve_Monodomain.py --space_disc DCT --enable_output --o_freq 10 --end_time 10 --domain_name cuboid_2D_very_small --pre_refinements 2 --order 4
```
For other command line options and their explanation, see
```shell
python3 Solve_Monodomain.py --help
```
The solution can be visualized with the `visualization/plot_DCT_sol.py` script.

### Using the FEM discretization
For the FEM spatial discretization _FEniCSx_ is needed. However, due to specific code for loading the fibers generated by _life^x-fiber_[^lifex], a specific version of _FEniCSx_ and _adios4dolfinx_ are needed. Therefore, we suggest to work in a Docker container.

Pull the Docker image:
```shell
docker pull rosilho/emrkc_dolfinx
docker tag rosilho/emrkc_dolfinx emrkc
```

From the root folder of the emRKC repository, build the ionic models for the docker containers
```shell
docker run --rm -ti -v "$(pwd)":/src -w /src/problem_classes/ionicmodels/cpp emrkc bash
c++ -O3 -Wall -shared -std=c++11 -fPIC -fvisibility=hidden $(python3 -m pybind11 --includes) bindings_definitions.cpp -o ionicmodels$(python3-config --extension-suffix)
exit
```
Then run a simulation
```shell
docker run --rm -ti -v "$(pwd)":/src emrkc mpirun -n 4 python3 Solve_Monodomain.py --space_disc FEM --o_freq 1 --end_time 10 --domain_name 03_fastl_LA --order 1 --fibrosis
```
Results are visualized with Paraview.

---

[^mrkc]: Abdulle, A., Grote, M. J., & Rosilho de Souza, G. (2022). Explicit stabilized multirate method for stiff differential equations. Mathematics of Computation, 91(338), 2681–2714. [https://doi.org/10.1090/mcom/3753](https://doi.org/10.1090/mcom/3753)

[^mrkccode]: Rosilho De Souza, G. (2022). mRKC: a multirate Runge—Kutta—Chebyshev code. [https://github.com/grosilho/mRKC](https://github.com/grosilho/mRKC)

[^imexrl]: Lindner, L. P., Gerach, T., Jahnke, T., Loewe, A., Weiss, D., & Wieners, C. (2023). Efficient time splitting schemes for the monodomain equation in cardiac electrophysiology. International Journal for Numerical Methods in Biomedical Engineering, 39(2), e3666. [https://doi.org/10.1002/cnm.3666](https://doi.org/10.1002/cnm.3666)

[^lifex]: Africa, P. C., Piersanti, R., & Fedele, M. (2022). lifex-fiber: an open tool for myofibers generation in cardiac computational models (1.4.0). Zenodo. [https://doi.org/10.5281/zenodo.7622070](https://doi.org/10.5281/zenodo.7622070)