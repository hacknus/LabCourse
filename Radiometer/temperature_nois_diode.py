import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from main import mat_temp

frequency= np.array([16, 17, 18, 19])+3.75+0.55
for i in [16, 17, 18, 19]:
    df = pd.read_csv(r'{}GHZ\diode_only.csv'.format(i))
    params = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i))
    T_diode, T_diode_err = mat_temp(df, params)
    print(T_diode)
