import numpy as np
from constants_2D import const_dirichlet_2D, const_dirichlet_2D_up, const_dirichlet_2D_down,r 

class dirichlet:
    def A(number_z):
        A=np.zeros((number_z**2,number_z**2))
        for i in np.arange(number_z,number_z**2-number_z):
            if i%number_z!=0 and i%number_z!=number_z-1:
                A[i][i-number_z]=r
                A[i][i+number_z]=r
                A[i][i-1]=r
                A[i][i+1]=r
                A[i][i]=1-4*r
        return A
    def C(number_z):
        C=np.zeros(number_z**2)
        for i in np.arange(number_z):
            C[i]= const_dirichlet_2D
            C[i-number_z]=const_dirichlet_2D
        for i in np.arange(number_z,number_z**2-number_z):
            if i%number_z==0 or i%number_z==number_z-1:
                C[i]=const_dirichlet_2D
        return C 
class periodic_horizontal:
    def A(number_z):
        A=np.zeros((number_z**2,number_z**2))
        for i in np.arange(number_z**2):
            if i%number_z!=0 and i%number_z!=number_z-1:
                A[i][i-number_z]=r
                A[i][(i+number_z)%(number_z**2)]=r
                A[i][i-1]=r
                A[i][(i+1)%(number_z**2)]=r
                A[i][i]=1-4*r
        return A
    def C(number_z):
        C=np.zeros(number_z**2)
        for i in np.arange(number_z**2):
            if i%number_z==0:
                C[i]=const_dirichlet_2D_up
            if i%number_z==number_z-1:
                C[i]= const_dirichlet_2D_down
        return C
        