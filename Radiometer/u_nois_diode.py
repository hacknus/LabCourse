import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import csv


OS = platform.system()

for i in [16,17,18,19]:
    K_and_C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i))
    nois_and_load = pd.read_csv(r'{}GHZ\noise_load_after.csv'.format(i))
    hot_load = pd.read_csv(r'{}GHZ\hot_load_after.csv'.format(i))
    
    K = K_and_C.K
    K_err = K_and_C.K_err
    
    C= K_and_C.C
    C_err = K_and_C.C_err
    
    U_nois_and_load = nois_and_load.v_val[0]
    U_nois_and_load_err = nois_and_load.v_std[0]

    
    U_load = hot_load.v_val[0]
    U_load_err = hot_load.v_std[0]
    
    U_0 = -C/K
    U_0_err = np.sqrt((C/K**2*K_err)**2 + (1/K*C_err)**2)
        
    U_diode = (U_nois_and_load - U_load + U_0)[0]
    
    U_diode_err =(np.sqrt(U_nois_and_load_err**2 + U_load_err**2 + U_0_err**2))[0] 
    
    with open('{}GHZ\diode_only.csv'.format(i), 'w', newline='') as csvfile:
        fieldnames = ['v_val', 'v_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'v_val':U_diode , 'v_std': U_diode_err})
            
    