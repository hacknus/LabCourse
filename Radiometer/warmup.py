import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def mat_temp(df, params):
    K = np.array(params.K)[0]
    K_err = np.array(params.K_err)[0]
    C = np.array(params.C)[0]
    C_err = np.array(params.C_err)[0]
    T = C + K*np.array(df.VDetector) - 273.15
    T_err = (C_err)**2 + (np.array(df.VDetector) * K_err)**2 + (K * np.array(0.05))**2
    return T, np.sqrt(T_err)

def linear(x, m, b):
    return x*m + b

def func(t,B,C,t0):
    return 2/np.sqrt(B*t - B*t0) + C


if __name__ == "__main__":
    header = ["date","time","TLoad","TAmp","TPlate","T4","VDetector","Elevation","V3","V4","Sky","Hot","Amp","Noise","LOMHz"]
    filename = "data/dataset{:03d}.csv".format(1)
    df = pd.read_csv(filename, header=0)
    t = np.arange(len(df))/2
    #plt.plot(t, df.VDetector, color="black")
    #plt.xlabel(r'$t$ [s]')
    #plt.ylabel(r'$U_{Detector}$ [V]')
    #plt.savefig("warmup.pdf")
    #plt.show()

    params_df = pd.read_csv("18GHZ/K_and_C.csv")
    T, Terr = mat_temp(df, params_df)
    print(Terr, T)

    b = T[-1]
    T = np.diff(np.array(T))

    plt.plot(t[1:], T + b, color="black")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$T$ [C]')
    plt.savefig("warmup_temperature.pdf")
    plt.show()


    dT_T = []
    t = []
    m = 40
    sample = range(3000, 6000, m)
    for k in range(5, m):
        temp = np.zeros(len(sample))
        for i, j in enumerate(sample):
            temp[i] = sum(T[j:j+k])
        temp /= k
        dT_T.append(temp.std() / (temp.mean() + b))
        t.append(0.1*k)
    plt.scatter(t, dT_T, color="black", s=4, marker="+")
    popt, pcov = curve_fit(func, t, dT_T, p0=[100e6,0,0])
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    plt.plot(t, func(np.array(t), *popt), color="red", ls="--")
    plt.xlabel(r"$t_{int}$ [s]")
    plt.ylabel(r"$\Delta T/T$ [-]")
    print(f"bandwidth: {popt[0]/1e6:.2f} +/- {np.sqrt(pcov[0][0])/1e6:.2f} MHz")
    print(f"dT/T: {1/np.sqrt(0.1*popt[0]/1e6):.2f} +/- {1/np.sqrt(0.1*np.sqrt(pcov[0][0])/1e6):.2f}")
    print(f"dT/T: {1/np.sqrt(10*popt[0]/1e6):.2f} +/- {1/np.sqrt(10*np.sqrt(pcov[0][0])/1e6):.2f}")
    plt.tight_layout()
    plt.savefig("bandwidth.pdf")
    plt.show()

