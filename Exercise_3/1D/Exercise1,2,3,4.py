import numpy as np
import math 
import matplotlib.pyplot as plt
from  initial import initial_normal, initial_const
from analytical import analytical_dirichlet
from boundary import dirichlet, neumann, cauchy, periodic
from source import no_source, moving, fix

def plot(T,name):
    plt.imshow(T,cmap="jet",aspect='auto')   
    z_ticks_stepsize=0.1
    plt.yticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    t_ticks_stepsize = 60
    plt.xticks(np.arange((max_t)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange((max_t/t_ticks_stepsize)))
    plt.xlabel('time [min]')
    plt.ylabel('position [m]')
    plt.colorbar().set_label('temperature [K]')
    plt.title(name)
    plt.show() 

    
def run(number_t,delta_t,max_t,number_z,delta_z,max_z,A,C,initial,source,name):        
    T=np.zeros((number_t,number_z))
    T[0]=initial    
    #Solving the linear equation sistem for each time step
    for i in np.arange(number_t-1):
        T[i+1]=np.dot(A,T[i])+C+source[i]
    T=np.transpose(T)
    plot(T,name)
  

    

#material constant
a = 23*1e-7   #  23*1e-6 is Thermal diffusivity of iron [m^2/s]

#resolution
max_t = 60*5       #[s]
max_z = 0.21         # [m]     
number_z=1000

delta_z = max_z/number_z
delta_t = delta_z**2*2/a #stability 
number_t = int(math.floor(max_t/delta_t)) #due to stability


#settings:
#initial conditions
u0=10
std=0.05

#boundary condition
const_dirichlet = 1
const_neumann = 10
const_cauchy_a = 1 
const_cauchy_b = 0.1      #darf nicht 0 sein a*C+b*dC/dx +c, so eingestellt wie neumann 
const_cauchy_c = 1

#heat source
f0=1             #in [K] over hole integration time     
w0 = 2*np.pi/max_t*5
tau = max_t/3    #zero is sharp, 1 is normal distributed

#boundary = dirichlet  #cauchy, neumann, dirichlet, periodic
#initial = const     #normal, const
#source = moving   #fix,moving , no


#exercise 1
run(number_t,delta_t,max_t,number_z,delta_z,max_z,dirichlet.A(number_z,delta_z),dirichlet.C(number_z,delta_z,const_dirichlet) ,initial_normal(number_z,delta_z,u0,std),no_source(number_t,number_z,delta_t,delta_z),'Ex. 1: Dirichlet')
plot(analytical_dirichlet(delta_z,delta_t,number_z,number_t,u0,a,std),'Ex. 1: analyticaly Dirichlet')

#exercise 2
a = 23*1e-6 #is Thermal diffusivity of iron [m^2/s]
delta_z = np.sqrt(2*delta_t*a) #stability 
number_z = int(math.floor(max_z/delta_z)) #due to stability

run(number_t,delta_t,max_t,number_z,delta_z,max_z,dirichlet.A(number_z,delta_z),dirichlet.C(number_z,delta_z,const_dirichlet) ,initial_normal(number_z,delta_z,u0,std),no_source(number_t,number_z,delta_t,delta_z),'Ex. 2: Dirichlet')
run(number_t,delta_t,max_t,number_z,delta_z,max_z,neumann.A(number_z,delta_z),neumann.C(number_z,delta_z,const_neumann) ,initial_normal(number_z,delta_z,u0,std),no_source(number_t,number_z,delta_t,delta_z),'Ex. 2: Neumann')
run(number_t,delta_t,max_t,number_z,delta_z,max_z,cauchy.A(number_z,delta_z,const_cauchy_a,const_cauchy_b),cauchy.C(number_z,delta_z,const_cauchy_c,const_cauchy_b) ,initial_normal(number_z,delta_z,u0,std),no_source(number_t,number_z,delta_t,delta_z),'Ex. 2: Cauchy')
run(number_t,delta_t,max_t,number_z,delta_z,max_z,periodic.A(number_z,delta_z),periodic.C(number_z,delta_z) ,initial_normal(number_z,delta_z,u0,std),no_source(number_t,number_z,delta_t,delta_z),'Ex. 2: Periodic')

#exercise 3
u0=0
run(number_t,delta_t,max_t,number_z,delta_z,max_z,dirichlet.A(number_z,delta_z),dirichlet.C(number_z,delta_z,const_dirichlet) ,initial_normal(number_z,delta_z,u0,std),fix(number_t,number_z,delta_t,delta_z,w0,f0,tau),'Ex. 3: Source fix in space')

#exercise 4
run(number_t,delta_t,max_t,number_z,delta_z,max_z,dirichlet.A(number_z,delta_z),dirichlet.C(number_z,delta_z,const_dirichlet) ,initial_normal(number_z,delta_z,u0,std),moving(number_t,number_z,delta_t,delta_z,w0,f0,tau),'Ex. 4: Source moving')