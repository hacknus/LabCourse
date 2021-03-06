import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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


def create_files(df, out_folder, out_file, save = True):
    V_val = np.array(df.VDetector).mean()
    V_std = np.array(df.VDetector).std()
    T_load_val = np.array(df.TLoad).mean()
    T_load_std = np.array(df.TLoad).std()
    T_amp_val = np.array(df.TAmp).mean()
    T_amp_std = np.array(df.TAmp).std()
    ele_val = np.array(df.Elevation).mean()
    ele_std = np.array(df.Elevation).std()
    d = {
        "v_val": [V_val],
        "v_std": [V_std],
        "T_load_val": [T_load_val],
        "T_load_std": [T_load_std],
        "T_amp_val": [T_amp_val],
        "T_amp_std": [T_amp_std],
        "ele_val": [ele_val],
        "ele_std": [ele_std]
    }
    dfnew = pd.DataFrame(data=d)
    if not save:
        return dfnew
    dfnew.to_csv(out_folder + "/" + out_file)


def gather_dataset(indices, folder):
    """
    gets  dataset and averages the measurements for each frequency
    saves hot_before.csv
    saves diode_before.csv
    saves hot_after.csv
    saves diode_after.csv
    saves elevation.csv
    saves hand.csv
    saves hand2.csv
    saves blackbody.csv
    saves acrylic.csv
    saves bluefoam.csv
    saves cellphone.csv
    :param indices, folder:
    :return:
    """

    header = ["date","time","TLoad","TAmp","TPlate","T4","VDetector","Elevation","V3","V4","Sky","Hot","Amp","Noise","LOMHz"]
    Path(folder).mkdir(parents=True, exist_ok=True)
    # hot load
    filename = "data/dataset{:03d}.csv".format(indices[0])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "hot_load_before.csv")

    if np.mean(df.Hot) != 1 and np.mean(df.Noise) != 0:
        print("error in hot load file")

    # noise load
    filename = "data/dataset{:03d}.csv".format(indices[1])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "noise_load_after.csv")

    if np.mean(df.Noise) != 1:
        print("error in hot noise file")

    for i in range(2, 12):
        filename = "data/dataset{:03d}.csv".format(indices[i])
        if i == 2:
            df = pd.read_csv(filename, header=0)
            if np.mean(df.Sky) != 1:
                print("error in angles file")
            df = create_files(df, folder, "angles.csv",False)
        else:
            new_df = pd.read_csv(filename, header=0)
            if np.mean(new_df.Sky) != 1:
                print("error in angles file")
            new_df = create_files(new_df, folder, "angles.csv", False)
            df = df.append(new_df,ignore_index = True)
    df.to_csv(folder + "/angles.csv")
    #plt.plot(df.ele_val,df.v_val)
    #plt.show()



    # hot load
    filename = "data/dataset{:03d}.csv".format(indices[12])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "hot_load_after.csv")

    if np.mean(df.Hot) != 1 and np.mean(df.Noise) != 0:
        print("error in hot Load file")

    # noise load
    filename = "data/dataset{:03d}.csv".format(indices[13])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "noise_load_after.csv")

    if np.mean(df.Noise) != 1:
        print("error in hot noise file")

    # hand
    filename = "data/dataset{:03d}.csv".format(indices[14])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "hand.csv")

    if np.mean(df.Sky) != 1:
        print("error in hand file")

    # hand2
    filename = "data/dataset{:03d}.csv".format(indices[15])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "hand2.csv")

    if np.mean(df.Sky) != 1:
        print("error in hand 2 file")

    # blackbody
    filename = "data/dataset{:03d}.csv".format(indices[16])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "blackbody.csv")

    if np.mean(df.Sky) != 1:
        print("error in blackbody file")

    # acrylic
    filename = "data/dataset{:03d}.csv".format(indices[17])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "acrylic.csv")

    if np.mean(df.Sky) != 1:
        print("error in acrylic file")

    # blue foam
    filename = "data/dataset{:03d}.csv".format(indices[18])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "bluefoam.csv")

    if np.mean(df.Sky) != 1:
        print("error in blue foam file")

    # cellphone
    filename = "data/dataset{:03d}.csv".format(indices[19])
    df = pd.read_csv(filename, header=0)
    create_files(df, folder, "cellphone.csv")

    if np.mean(df.Sky) != 1:
        print("error in cellphone file")

if __name__ == "__main__":
    #split_dataset("data_raw.csv")
    FREQ1 = [2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    gather_dataset(FREQ1, "16GHZ")
    FREQ2 = [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
    gather_dataset(FREQ2, "17GHZ")
    FREQ3 = [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
    gather_dataset(FREQ3, "18GHZ")
    FREQ4 = [65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]
    gather_dataset(FREQ4, "19GHZ")