import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from data_analysis import K,c,rel_thickness
def T(U,K,c):
    return K*U+c

OS = platform.system()

for freq, c in zip([16, 17, 18, 19], ['red', 'blue', 'green', 'orange']):
    if OS == "Windows":
        angles = pd.read_csv(r'{}GHZ\angles.csv'.format(freq))  
    else:
        angles = pd.read_csv('{}GHZ/angles.csv'.format(freq))
    print(angles.v_val)
    K=1
    c=2
    T_sky = T(angles.v_val,K,c)
    
    plt.plot(rel_thickness,T_sky)
plt.xlabel("rel. thickness [-]")
plt.ylabel("T [$C°$]")
plt.legend()
plt.show()    