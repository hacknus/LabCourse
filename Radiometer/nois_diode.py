import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

def T(U,K,c):
    return K*U+C


T_freq = np.zeros(4)
frequency = np.array([16, 17, 18, 19])+3.75+0.55
for i in np.arange(4):
    hot_load_after = pd.read_csv(r'{}GHZ\hot_load_after.csv'.format(i+16))
    nois = pd.read_csv(r'{}GHZ\noise_load_after.csv'.format(i+16))
    U_freq = nois.v_val - hot_load_after.v_val
    print(U_freq)
    K = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).K
    print(K)
    C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16)).C
    print(C)
    print()
    T_freq[i] = T(U_freq,K,C)
plt.plot(frequency,T_freq,'bo')
plt.xticks(frequency)
plt.xlabel('frequency [GHz]')
plt.ylabel('T [$K°$]')
plt.show()