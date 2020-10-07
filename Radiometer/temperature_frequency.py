import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

def T(U,K,c):
    return K*U+C

rel_thickness = np.array(1 / np.cos((90 - np.array(pd.read_csv(r'{}GHZ\angles.csv'.format(16)).ele_val)) / 180 * np.pi)).round(1)
for elevation, c in zip([0,3,6,9], ['red', 'blue', 'green', 'orange']):  
    T_freq = np.zeros(4)
    frequency= np.array([16, 17, 18, 19])+3.75+0.55
    for i in np.arange(4):
        angels = pd.read_csv(r'{}GHZ\angles.csv'.format(i+16))
        U_freq = angels.v_val[elevation]
        K = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).K
        C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).C
        T_freq[i] = T(U_freq,K,C)
    plt.plot(frequency,T_freq,'bo',color=c,label= rel_thickness[elevation])
plt.legend(title='thickness')
plt.xticks(frequency)
plt.xlabel('frequency [GHz]')
plt.ylabel('T [$K°$]')
plt.show