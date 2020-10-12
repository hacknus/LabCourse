import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from main import mat_temp

frequency= np.array([16, 17, 18, 19])+3.75+0.55
T_diode = np.zeros(4)
T_diode_err = np.zeros(4)
for i in [16, 17, 18, 19]:
    df = pd.read_csv(r'{}GHZ\diode_only.csv'.format(i))
    params = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i))
    T_diode[i-16], T_diode_err[i-16] = mat_temp(df, params)
    print(T_diode[i-16])
plt.errorbar(frequency, T_diode, yerr=T_diode_err, color='black', fmt='.', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
plt.plot(frequency,T_diode,'.',color='black')
plt.xticks(frequency)
plt.ylabel('T [C°]')
plt.xlabel('frequency [GHz]')
#plt.legend()
plt.tight_layout()
plt.savefig("nois_diode.pdf")
plt.show()
