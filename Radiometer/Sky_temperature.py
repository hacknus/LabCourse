import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

def T(U,K,c):
    return K*U+C

OS = platform.system()
for freq, c in zip([16, 17, 18, 19], ['red', 'blue', 'green', 'orange']):
    if OS == "Windows":
        K_and_C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(freq))  
        angles = pd.read_csv(r'{}GHZ\angles.csv'.format(freq))
    else:
        K_and_C = pd.read_csv('{}GHZ/K_and_C.csv'.format(freq))
        angles = pd.read_csv(r'{}GHZ/angles.csv'.format(freq))
    
    K=np.array(K_and_C.K)
    C=np.array(K_and_C.C)
    T_sky = T(np.array(angles.v_val),K,C)
    rel_thickness = 1 / np.cos((90 - np.array(angles.ele_val)) / 180 * np.pi)
    plt.plot(rel_thickness,T_sky)
    
plt.xlabel("rel. thickness [-]")
plt.ylabel("T [$C°$]")
plt.legend()
plt.show()    