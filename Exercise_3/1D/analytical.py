import numpy as np



def analytical_solution_dirichlet(t,x,u0,a,std):
    return u0*std/np.sqrt(std**2+4*a*t)*np.e**((x**2/(4*a*t))*(std**2/(std**2+4*a*t)-1))

def analytical_dirichlet(delta_z,delta_t,number_z,number_t,u0,a,std):
    grid_z = np.array(delta_z*np.arange(number_z))
    grid_t = np.array(delta_t*np.arange(number_t))
    grid = np.zeros((number_t,number_z))
    for i in np.arange(number_t):
        if i*delta_t < 0.01:
            grid[i]=0
        else:
            grid[i]=analytical_solution_dirichlet(grid_t[i],grid_z-0.5*(number_z-1)*delta_z,u0,a,std)
    grid = np.transpose(grid)
    return grid
