import numpy as np




class dirichlet:
    def A(number_z,delta_z):
        r=0.5
        A=np.diagflat(np.full(number_z, 1-2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, r, dtype=np.float),-1)
        A[0][0]=0
        A[0][1]=0
        A[-1][-1]=0
        A[-1][-2]=0
        return A
    
    def C(number_z,delta_z,const_dirichlet):
        C=np.zeros(number_z)
        C[0]=const_dirichlet
        C[-1]=const_dirichlet
        return C
    

class neumann:
    def A(number_z,delta_z):
        r=0.5
        A=np.diagflat(np.full(number_z, 1-2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, r, dtype=np.float),-1)
        A[0][0]=1-r
        A[-1][-1]=1-r
        return A
    
    def C(number_z,delta_z,const_neumann):
        C = np.zeros(number_z, dtype=np.float)
        C[0] = 0.5*delta_z*const_neumann
        C[-1]= 0.5*delta_z*const_neumann
        return C
    
class cauchy:
    def A(number_z,delta_z,const_cauchy_a,const_cauchy_b):
        r=0.5
        A=np.diagflat(np.full(number_z, 1-2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, r, dtype=np.float),-1)
        A[0][0]=(1-r*(const_cauchy_a/const_cauchy_b*delta_z+1))
        A[-1][-1]=(1-r*(const_cauchy_a/const_cauchy_b*delta_z+1))
        return A
    
    def C(number_z,delta_z,const_cauchy_c,const_cauchy_b):
        C = np.zeros(number_z, dtype=np.float)
        C[0] = 0.5*delta_z*const_cauchy_c/const_cauchy_b
        C[-1]= 0.5*delta_z*const_cauchy_c/const_cauchy_b
        return C

class periodic:
    def A(number_z,delta_z):
        r=0.5
        A=np.diagflat(np.full(number_z, 1-2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, r, dtype=np.float),-1)
        A[0][-1]=r
        A[-1][0]=r
        return A
    
    def C(number_z,delta_z):
        C=np.zeros(number_z)
        return C