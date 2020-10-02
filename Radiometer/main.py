import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_file(filename):
    return pd.read_csv(filename, header=0)


def plot():
    for directory in ["16GHZ", "17GHZ", "18GHZ", "19GHZ"]:
        df = read_file(directory + "/angles.csv")
        plt.plot(df.ele_val,df.v_val,label=directory)
    plt.legend()
    plt.xlabel("elevation [°]")
    plt.ylabel("detector voltage [V]")
    plt.show()

if __name__ == "__main__":
    plot()