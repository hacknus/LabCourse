import numpy as np       


def invert(B):
    
    
    #these gausformations makes B tridiagonal
    number_z=np.size(B[0])
    B_inv=np.diagflat(np.full(number_z, 1, dtype=np.float))
    if B[0][1]==1:
        if B[1][2]<0:
            for i in np.arange(1,number_z):
                B[0]=B[0]-B[i]
                B_inv[0]=B_inv[0]-B_inv[i]
        if B[1][2]>0:
            for i in np.arange(1,number_z):
                B[0]=B[0]+B[i]
                B_inv[0]=B_inv[0]+B_inv[i]
    
    #making B[0][0]=1
    B_inv[0]=B_inv[0]/B[0][0]
    B[0]=B[0]/B[0][0]
    
    #making B[i][i-1]=0  and B[i][i]  
    for i in np.arange(number_z-1):
        B_inv[i+1]=(B_inv[i+1]-B[i+1][i]*B_inv[i])/(B[i+1][i+1]-B[i+1][i]*B[i][i+1])
        B[i+1]=(B[i+1]-B[i+1][i]*B[i])/(B[i+1][i+1]-B[i+1][i]*B[i][i+1])
    
    #making B[i][i+1]=0
    for i in np.arange(number_z-1,0,-1):
        B_inv[i-1]=B_inv[i-1]-B[i-1][i]*B_inv[i]
        B[i-1]=B[i-1]-B[i-1][i]*B[i]
        
    return B_inv 

