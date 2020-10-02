import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.integrate as integrate
import math
from numpy.linalg import inv
from zentit2 import zenit, zenit_value


max_t=7  #[d]
number_t=7*1000
delta_t=max_t*24*3600/number_t #[s]

max_z=15   #[m]
number_z=1071
delta_z=max_z/number_z #[m]

max_z_show=5  #plot sequenz depth
max_t_show=max_t #plot sequenz time
min_t_show=0
scale_z=20

S_0=1367
sigma=5.67*1e-8
T_res=10+273.15 #
latitude=0 

a=0.3                #granit
epsilon=0.45
c=890              
rho=2750
k=2.9
K=k/(c*rho)
r=K*delta_t/(delta_z)**2


############################################################ Matrix for trapezoid
A_trap=np.diagflat(np.full(number_z, -1+r, dtype=np.float))+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),-1)
A_trap[0]=np.full(number_z,1)
A_trap[number_z-1][number_z-1]=-1
A_trap[number_z-2][number_z-1]=0

B_trap=np.diagflat(np.full(number_z, -1-r, dtype=np.float))+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),-1)
B_trap[0]=np.full(number_z,1)
B_trap[number_z-1][number_z-1]=-1
B_trap[number_z-2][number_z-1]=0

C_trap=np.zeros(number_z)
C_trap[number_z-2]=r*T_res
C_trap[number_z-1]=-r*T_res
###########################################################
A_trap_iso=np.zeros((number_z,number_z))
for i in np.arange(1,number_z):
    A_trap_iso[i]=A_trap[i]
A_trap_iso[0][0]=1

B_trap_iso=np.zeros((number_z,number_z))
for i in np.arange(1,number_z):
    B_trap_iso[i]=B_trap[i]
B_trap_iso[0][0]=1

C_trap_iso=C_trap
##############################################################
A_leap=np.diagflat(np.full(number_z, 1, dtype=np.float))+np.diagflat(np.full(number_z-1, 0, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 0, dtype=np.float),-1)
A_leap[0]=np.full(number_z,1)
A_leap[number_z-2][number_z-1]=0


B_leap=np.diagflat(np.full(number_z, 1+2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, -r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -r, dtype=np.float),-1)
B_leap[0]=np.full(number_z,1)
B_leap[number_z-1][number_z-1]=1
B_leap[number_z-2][number_z-1]=0


C_leap=np.zeros(number_z)
C_leap[number_z-2]=-r*T_res
C_leap[number_z-1]=r*T_res
####################################################################
A_leap_iso=np.zeros((number_z,number_z))
for i in np.arange(1,number_z):
    A_leap_iso[i]=A_leap[i]
A_leap_iso[0][0]=1

B_leap_iso=np.zeros((number_z,number_z))
for i in np.arange(1,number_z):
    B_leap_iso[i]=B_leap[i]
B_leap_iso[0][0]=1

C_leap_iso=C_leap
#############################################################
def result(A,B,C):
    T=np.zeros((number_t,number_z))
    T[0][:]=T_res
    invB=np.linalg.inv(B)
    for i in np.arange(number_t-1):
        C[0]=-delta_t/(c*rho*delta_z)*(-epsilon*sigma*T[i][0]**4+(1-a)*S_0*np.cos(zenit_value(i*delta_t/(3600*24),latitude)/360*2*np.pi))
        T[i+1][:]=np.dot(invB,np.dot(A,T[i][:])-C)
    return T
#####################################################################
max_z_show_steps=int(max_z_show/max_z*number_z)
max_t_show_steps=int(max_t_show/max_t*number_t)
min_t_show_steps=int(min_t_show/max_t*number_t)


R0=result(A_trap,B_trap,C_trap).transpose()
R=np.zeros((max_z_show_steps*scale_z,max_t_show_steps-min_t_show_steps))
for j in np.arange(max_z_show_steps):
    for l in np.arange(scale_z):
        for i in np.arange(max_t_show_steps-min_t_show_steps):
            R[j*scale_z+l][i]=R0[j][i+min_t_show_steps]
            
        
plt.matshow(R)
plt.title('Trapezoidal')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z*scale_z,np.arange(max_z_show+1))
plt.xticks(np.arange(max_t_show-min_t_show+1)*number_t/max_t,np.arange(min_t_show,max_t_show+1))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('Temperature [K]')
plt.show()        
T_trap=R0[0]

R0=result(A_trap_iso,B_trap_iso,C_trap_iso).transpose()
R=np.zeros((max_z_show_steps*scale_z,max_t_show_steps-min_t_show_steps))
for j in np.arange(max_z_show_steps):
    for l in np.arange(scale_z):
        for i in np.arange(max_t_show_steps-min_t_show_steps):
            R[j*scale_z+l][i]=R0[j][i+min_t_show_steps]
plt.matshow(R)
plt.title('Trapezoidal Iso')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z*scale_z,np.arange(max_z_show+1))
plt.xticks(np.arange(max_t_show-min_t_show+1)*number_t/max_t,np.arange(min_t_show,max_t_show+1))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('Temperature [K]')
plt.show()        
T_trap_iso=R0[0]

R0=result(A_leap,B_leap,C_leap).transpose()
R=np.zeros((max_z_show_steps*scale_z,max_t_show_steps-min_t_show_steps))
for j in np.arange(max_z_show_steps):
    for l in np.arange(scale_z):
        for i in np.arange(max_t_show_steps-min_t_show_steps):
            R[j*scale_z+l][i]=R0[j][i+min_t_show_steps]
plt.matshow(R)
plt.title('Leap-Frog')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z*scale_z,np.arange(max_z_show+1))
plt.xticks(np.arange(max_t_show-min_t_show+1)*number_t/max_t,np.arange(min_t_show,max_t_show+1))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('Temperature [K]')
plt.show()        
T_leap=R0[0]

R0=result(A_leap_iso,B_leap_iso,C_leap_iso).transpose()
R=np.zeros((max_z_show_steps*scale_z,max_t_show_steps-min_t_show_steps))
for j in np.arange(max_z_show_steps):
    for l in np.arange(scale_z):
        for i in np.arange(max_t_show_steps-min_t_show_steps):
            R[j*scale_z+l][i]=R0[j][i+min_t_show_steps]
plt.matshow(R)
plt.title('Leap-Frog Iso')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z*scale_z,np.arange(max_z_show+1))
plt.xticks(np.arange(max_t_show-min_t_show+1)*number_t/max_t,np.arange(min_t_show,max_t_show+1))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('Temperature [K]')
plt.show()        
T_leap_iso=R0[0]


plt.plot(T_leap, label='Leap-Frog')
plt.plot(T_trap_iso, label='Isolated')
#plt.plot(T_trap, label='Trapezoidal')
plt.title('Surface Temperature $T_{0}$')
plt.xticks(np.arange(max_t+1)*number_t/max_t,np.arange(max_t+1))
plt.xlabel('time [d]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()
