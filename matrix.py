import numpy as np

#Creating the matrices of the trapezoidal method. 
#Boundary conditions Take into account the heat dissipation of the uppermost cell.
class Trapezoidal:
    def A(number_z, r):
        A_trap=np.diagflat(np.full(number_z, -1+r, dtype=np.float))+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),-1)
        A_trap[0]=np.full(number_z,1)
        A_trap[number_z-1][number_z-1]=-1
        A_trap[number_z-2][number_z-1]=0
        return A_trap
        
    def B(number_z, r):
        B_trap=np.diagflat(np.full(number_z, -1-r, dtype=np.float))+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),-1)
        B_trap[0]=np.full(number_z,1)
        B_trap[number_z-1][number_z-1]=-1
        B_trap[number_z-2][number_z-1]=0
        return B_trap
        
    def C(number_z, r, T_res):
        C_trap=np.zeros(number_z)
        C_trap[number_z-2]=r*T_res
        C_trap[number_z-1]=-r*T_res
        return C_trap
    
#Creating the matrices of the trapezoidal method. 
#Boundary conditions doesn't Take into account the heat dissipation of the uppermost cell.
class Trapezoidal_Iso:
    def A(number_z, r):
        A_trap_iso=np.diagflat(np.full(number_z, -1+r, dtype=np.float))+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -1/2*r, dtype=np.float),-1)
        A_trap_iso[0][0]=1
        A_trap_iso[0][1]=0
        A_trap_iso[number_z-1][number_z-1]=-1
        A_trap_iso[number_z-2][number_z-1]=0
        return A_trap_iso
    
    def B(number_z, r):
        B_trap_iso=np.diagflat(np.full(number_z, -1-r, dtype=np.float))+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 1/2*r, dtype=np.float),-1)
        B_trap_iso[0][0]=1
        B_trap_iso[0][1]=0
        B_trap_iso[number_z-1][number_z-1]=-1
        B_trap_iso[number_z-2][number_z-1]=0
        return B_trap_iso
    def C(number_z, r, T_res):
        C_trap_iso=np.zeros(number_z)
        C_trap_iso[number_z-2]=r*T_res
        C_trap_iso[number_z-1]=-r*T_res
        return C_trap_iso

#Creating the matrices of the leap-frog method. 
#Boundary conditions Take into account the heat dissipation of the uppermost cell.   
class Leap_Frog:
    def A(number_z, r):
        A_leap=np.diagflat(np.full(number_z, 1, dtype=np.float))+np.diagflat(np.full(number_z-1, 0, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 0, dtype=np.float),-1)
        A_leap[0]=np.full(number_z,1)
        A_leap[number_z-2][number_z-1]=0
        return A_leap

    def B(number_z, r): 
        B_leap=np.diagflat(np.full(number_z, 1+2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, -r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -r, dtype=np.float),-1)
        B_leap[0]=np.full(number_z,1)
        B_leap[number_z-1][number_z-1]=1
        B_leap[number_z-2][number_z-1]=0
        return B_leap

    def C(number_z, r, T_res):
        C_leap=np.zeros(number_z)
        C_leap[number_z-2]=-r*T_res
        C_leap[number_z-1]=r*T_res
        return C_leap

#Creating the matrices of the leap-frog method.   
#Boundary conditions doesn't Take into account the heat dissipation of the uppermost cell.
class Leap_Frog_Iso:
    def A(number_z, r):
        A_leap_iso=np.diagflat(np.full(number_z, 1, dtype=np.float))+np.diagflat(np.full(number_z-1, 0, dtype=np.float),1)+np.diagflat(np.full(number_z-1, 0, dtype=np.float),-1)
        A_leap_iso[0][0]=1
        A_leap_iso[0][1]=0
        A_leap_iso[number_z-2][number_z-1]=0
        return A_leap_iso

    def B(number_z, r): 
        B_leap_iso=np.diagflat(np.full(number_z, 1+2*r, dtype=np.float))+np.diagflat(np.full(number_z-1, -r, dtype=np.float),1)+np.diagflat(np.full(number_z-1, -r, dtype=np.float),-1)
        B_leap_iso[0][0]=1
        B_leap_iso[0][1]=0
        B_leap_iso[number_z-1][number_z-1]=1
        B_leap_iso[number_z-2][number_z-1]=0
        return B_leap_iso

    def C(number_z, r,T_res):
        C_leap_iso=np.zeros(number_z)
        C_leap_iso[number_z-2]=-r*T_res
        C_leap_iso[number_z-1]=r*T_res
        return C_leap_iso
