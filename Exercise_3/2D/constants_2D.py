#material
a = 23*1e-7   #  23*1e-6 is Thermal diffusivity of iron [m^2/s]
r = 0.25
#initial
u0 = 0

#boundary
const_dirichlet_2D=1
const_dirichlet_2D_up = 1
const_dirichlet_2D_down=-1

#source
power=100
rel_x=0.1
rel_y=0.1