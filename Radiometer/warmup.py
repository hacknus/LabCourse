import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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