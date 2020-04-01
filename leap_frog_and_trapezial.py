import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.integrate as integrate
import math
from numpy.linalg import inv
from zentit2 import zenit, zenit_value


max_t=10    #[d]
number_t=10*24
delta_t=max_t*24*3600/number_t #[s]

max_z=10   #[m]
number_z=1000
delta_z=max_z/number_z #[m]


a=0.3
S_0=1367
epsilon=0.6
sigma=5.67*1e-8
T_res=10+273.15

c=890               #granit
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
print(A_leap_iso)
print(A_leap)
print(B_leap_iso)
print(B_leap)
###################################################################3
A_leap_iso=A_leap
A_leap_iso[0]=A_trap_iso[0]

B_leap_iso=B_leap
B_leap_iso[0]=B_trap_iso[0]

C_leap_iso=C_trap_iso
#############################################################
def result(A,B,C):
    T=np.zeros((number_t,number_z))
    T[0][:]=T_res
    invB=np.linalg.inv(B)
    for i in np.arange(number_t-1):
        C[0]=-delta_t/(c*rho*delta_z)*(-epsilon*sigma*T[i][0]**4+(1-a)*S_0*np.cos(zenit_value(i*delta_t/(3600*24),69)/360*2*np.pi))
        T[i+1][:]=np.dot(invB,np.dot(A,T[i][:])-C)
    return T
#####################################################################
figure(figsize=(9, 9))
plt.matshow(np.delete(result(A_trap_iso,B_trap_iso,C_trap_iso).transpose(),number_z-1,0))
plt.colorbar()
plt.show()
plt.plot(result(A_leap_iso,B_leap_iso,C_leap_iso).transpose()[0])