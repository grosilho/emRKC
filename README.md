# emRKC
Exponential multirate Runge-Kutta-Chebychev methods applied to the monodomain model

Code refactoring in progress....

Pull image
docker pull rosilho/emrkc_dolfinx

From the root folder of emRKC repository, reate docker image:
docker create --name emRKC -ti -v "$(pwd)":/src rosilho/emrkc_dolfinx

Start, run stop the container:
docker start emRKC
docker exec -ti -w /src/run_scripts emRKC mpirun -n 1 python3 Solve_Monodomain.py
docker stop emRKC




