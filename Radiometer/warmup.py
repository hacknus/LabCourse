import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mat_temp(df, params):
    K = np.array(params.K)[0]
    K_err = np.array(params.K_err)[0]
    C = np.array(params.C)[0]
    C_err = np.array(params.C_err)[0]
    T = C + K*np.array(df.VDetector) - 273.15
    T_err = (C_err)**2 + (np.array(df.VDetector) * K_err)**2 + (K * np.array(0.05))**2
    return T, np.sqrt(T_err)


if __name__ == "__main__":
    header = ["date","time","TLoad","TAmp","TPlate","T4","VDetector","Elevation","V3","V4","Sky","Hot","Amp","Noise","LOMHz"]
    filename = "data/dataset{:03d}.csv".format(1)
    df = pd.read_csv(filename, header=0)
    t = np.arange(len(df))/2
    plt.plot(t, df.VDetector, color="black")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$U_{Detector}$ [V]')
    plt.savefig("warmup.pdf")
    plt.show()

    params_df = pd.read_csv("16GHZ/K_and_C.csv")
    T, Terr = mat_temp(df, params_df)
    plt.plot(t, T, color="black")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$T$ [C]')
    plt.savefig("warmup_temperature.pdf")
    plt.show()