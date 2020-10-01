import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def split_dataset(filename):
    parse_dates = ['time']
    measurement = 0
    last_i = 0
    df = pd.read_csv(filename, parse_dates=parse_dates)
    for i in df.index[:-1]:
        t0 = df.time[i].to_datetime64().astype(int) / 10**9
        t1 = df.time[i+1].to_datetime64().astype(int) / 10**9
        if t1-t0 > 0.6:
            # new measurement detected
            measurement += 1
            new_df = df.iloc[last_i:i]
            new_df.to_csv("data/dataset{:03d}.csv".format(measurement))
            last_i = i + 1

def read_dataset(i=0):
    filename = "data/dataset{:03d}.csv".format(i)




if __name__ == "__main__":
    split_dataset("data_Raw.csv")