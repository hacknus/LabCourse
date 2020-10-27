import numpy as np

#material constant
a = 23*1e-7   #  23*1e-6 is Thermal diffusivity of iron [m^2/s]

#initial conditions
u0=10
std=0.05

#boundary condition
const_dirichlet = 1
const_neumann = 0.1
const_cauchy_a = 1  
const_cauchy_b = 0.001      #darf nicht 0 sein
const_cauchy_c = 0.1

#heat source
heating=10             #in [K] over hole integration time     
rotations=0
flatness_time = 0.1    #zero is sharp, 1 is normal distributed


