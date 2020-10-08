import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

def T(U,K,C):
    return K*U+C


T_hot_load = np.zeros(4)
T_nois = np.zeros(4)
frequency= np.array([16, 17, 18, 19])+3.75+0.55
for i in np.arange(4):
    hot_load_after = pd.read_csv(r'{}GHZ\hot_load_after.csv'.format(i+16))
    nois = pd.read_csv(r'{}GHZ\noise_load_after.csv'.format(i+16))
    U_nois = nois.v_val 
    U_hot_load = hot_load_after.v_val
    K = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).K
    C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).C
    T_nois[i] = T(U_nois,K,C)
    T_hot_load[i] = T(U_hot_load,K,C)
    T_diode = T_nois - T_hot_load
plt.plot(frequency,(T_diode-273.15),'bo')
plt.xticks(frequency)
plt.xlabel('frequency [GHz]')
plt.ylabel('T [$C°$]')
plt.show