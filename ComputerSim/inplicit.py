import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.integrate as integrate
import math
from numpy.linalg import inv
from zentit2 import zenit, zenit_value

number_z=1000
number_t=24*365

delta_z=1/100
delta_t=3*60*60

c=890               #granit
rho=2750
k=2.9
K=k/(c*rho)
a=0.3
S_0=1367
epsilon=0.6
sigma=5.67*1e-8

T_res=10+273.15
#distance_m=delta_m*number_m          
#distance_t=delta_t*number_t        


##############################################################################3


A=np.zeros((number_z,number_z))
B=np.zeros((number_z,number_z))
for i in np.arange(1,number_z):
        A[0][i]=-1
        B[0][i]=1
        A[i][i]=-1+K*delta_t/delta_z**2
        B[i][i]=1+K*delta_t/delta_z**2
        A[i][i-1]=-1/2*K*delta_t/delta_z**2
        B[i][i-1]=-1/2*K*delta_t/delta_z**2
        if i<number_z-2: 
            A[i][i+1]=-1/2*K*delta_t/delta_z**2
            B[i][i+1]=-1/2*K*delta_t/delta_z**2
A[0][0]=-1
#B[0][0]=1
A[number_z-1][number_z-1]=-1
B[number_z-1][number_z-1]=1
#invB=np.linalg.inv(B)


C=np.zeros(number_z)
C[number_z-2]=k/(c*rho)*delta_t/delta_z**2*T_res
C[number_z-1]=-k/(c*rho)*delta_t/delta_z**2*T_res

T=np.zeros((number_t,number_z))
T[0][:]=T_res

omega=zenit(number_t,delta_t)
for i in np.arange(number_t-1):

    B[0][0]=1+epsilon*sigma*delta_t*4*T[i][0]**3/(c*rho*delta_z*2)
    C[0]=delta_t/(c*rho*delta_z*2)*(2*epsilon*sigma*T[i][0]**4+(1-a)*S_0*(np.cos(omega[i+1]/360*2*np.pi)+np.cos(omega[i]/360*2*np.pi)))
#    C[0]=delta_t/(c*rho*delta_z)*(-epsilon*sigma*T[i][0]**4+(1-a)*S_0*np.cos(omega[i]/360*2*np.pi))
    invB=np.linalg.inv(B)
    T[i+1][:]=np.dot(invB,C-np.dot(A,T[i][:]))


T_plot=np.zeros((number_z-1,number_t))
for i in np.arange(number_z-1):
    T_plot[:][i]=T.transpose()[:][i]
figure(figsize=(9, 9))
plt.matshow(T_plot)
#plt.yticks((0,100,200,300,400,500),(0,1,2,3,4,5))
#plt.xticks(np.arange(number_t*delta_t),np.arange(number_t*delta_t))
#plt.yticks(size=20)
#plt.xticks(size=20)
plt.colorbar()
plt.show()
