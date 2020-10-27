import numpy as np
import math 
import matplotlib.pyplot as plt


def plot(T,title):
    plt.imshow(T,cmap="jet",aspect='auto')   
    z_ticks_stepsize=0.1
    plt.yticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    plt.xticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),-np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    plt.xlabel('position [m]')
    plt.ylabel('position [m]')
    plt.colorbar().set_label('temperature [K]')
    plt.title(title)
    plt.show()  



def periodic(number_t,number_z,T,R): 
    #define variables to make the code clearer
    r=0.25
    for i in np.arange(number_t-1):
        for j in np.arange(number_z):
            for k in np.arange(number_z):
                T[j][k]=r*(T[j][k-1]+T[j][(k+1)%number_z]+T[j-1][k]+T[(j+1)%number_z][k])+(1-4*r)*T[j][k]+R[j][k]
    plot(T,'Periodic')

def dirichlet(number_t,number_z,T,R,const):
    r=0.25
    T[0]=np.ones(number_z)*const
    T[number_z-1]=np.ones(number_z)*const
    T[:,0]=np.ones(number_z)*const
    T[:,number_z-1]=np.ones(number_z)*const
    for i in np.arange(number_t-1):
        for j in np.arange(1,number_z-1):
            for k in np.arange(1,number_z-1):
                T[j][k]=r*(T[j][k-1]+T[j][(k+1)%number_z]+T[j-1][k]+T[(j+1)%number_z][k])+(1-4*r)*T[j][k]+R[j][k]
    plot(T,'Dirichlet')

def kochtopf(number_t,number_z,T,R,const_up,const_down):
    r=0.25
    T[0]=np.ones(number_z)*const_up
    T[number_z-1]=np.ones(number_z)*const_down
    for i in np.arange(number_t-1):
        for j in np.arange(1,number_z-1):
            for k in np.arange(number_z):
                T[j][k]=r*(T[j][k-1]+T[j][(k+1)%number_z]+T[j-1][k]+T[(j+1)%number_z][k])+(1-4*r)*T[j][k]+R[j][k]
    plot(T,'Kochtopf')

def initial(number_z,delta_z,u0,std,c1,boundary):
    name=boundary
    T=np.zeros((number_z,number_z))
    for i in np.arange(number_z):
        for j in np.arange(number_z):
            x=i*delta_z-number_z*delta_z*0.5
            y=j*delta_z-number_z*delta_z*0.5
            T[i][j]=u0*np.e**(-(x**2+y**2)/std**2)+c1
    plot(T,'Initial {}'.format(name))
    return T

def source(number_z,delta_z,f0,tau,c2,boundary):
    name=boundary
    T=np.zeros((number_z,number_z))
    for i in np.arange(number_z):
        for j in np.arange(number_z):
            x=i*delta_z-number_z*delta_z*0.5
            y=j*delta_z-number_z*delta_z*0.5
            T[i][j]=f0*np.e**(-(x**2+y**2)/tau**2)+c2
    plot(T,'Source {}'.format(name))
    return T


#materia
a = 23*1e-7   #  23*1e-6 is Thermal diffusivity of iron [m^2/s]
r = 0.25

#settings
number_t = 100
max_t = 60     #[s]
max_z = 0.20         # [m]    

delta_t = max_t/number_t
delta_z = np.sqrt(delta_t*a/r) #stability 
number_z = int(math.floor(max_z/delta_z)) #due to stability crit

#boundary
const_dirichlet=1 #dirichlet
platte= 200 +273 #kochtopf
deckel= 30 +273

#initial 
std=max_z/2
u0=1
c1=0

#source
tau = max_z/3
f0=0
c2=0
  
dirichlet(number_t,number_z,initial(number_z,delta_z,u0,std,c1,'Dirichlet'),source(number_z,delta_z,f0,tau,c2,'Dirichlet'),const_dirichlet)
periodic(number_t,number_z,initial(number_z,delta_z,u0,std,c1,'Periodic'),source(number_z,delta_z,f0,tau,c2,'Periodic'))
kochtopf(number_t,number_z,initial(number_z,delta_z,0,std,273+30,'Kochtopf'),source(number_z,delta_z,0,tau,c2,'Kochtopf'),deckel,platte)