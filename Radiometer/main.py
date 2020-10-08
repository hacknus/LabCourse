import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_file(filename):
    return pd.read_csv(filename, header=0)


def plot_angles():
    for directory in ["16GHZ", "17GHZ", "18GHZ", "19GHZ"]:
        df = read_file(directory + "/angles.csv")
        plt.plot(df.ele_val,df.v_val,label=directory)
    plt.legend()
    plt.xlabel("elevation [°]")
    plt.ylabel("detector voltage [V]")
    plt.savefig("4freq_angles.pdf")
    plt.show()


def mat_temp(df, params):
    K = np.array(params.K)[0]
    K_err = np.array(params.K_err)[0]
    C = np.array(params.C)[0]
    C_err = np.array(params.C_err)[0]
    T = C + K*np.array(df.v_val)[0] - 273.15
    T_err = (C_err)**2 + (np.array(df.v_val)[0] * K_err)**2 + (K * np.array(df.v_std)[0])**2
    return T, np.sqrt(T_err)

def get_materials():
    for directory in ["16GHZ", "17GHZ", "18GHZ", "19GHZ"]:
        params_df = read_file(directory + "/K_and_C.csv")
        print(directory)
        hand_df = read_file(directory + "/hand.csv")
        Thand, Thanderr = mat_temp(hand_df, params_df)
        print("T_hand = {:.6f} +/- {:.6f}".format(Thand, Thanderr))
        hand2_df = read_file(directory + "/hand2.csv")
        Thand2, Thand2err = mat_temp(hand2_df, params_df)
        print("T_hand2 = {:.6f} +/- {:.6f}".format(Thand2, Thand2err))
        bb_df = read_file(directory + "/blackbody.csv")
        Tbb, Tbberr = mat_temp(bb_df, params_df)
        print("T_bb = {:.6f} +/- {:.6f}".format(Tbb, Tbberr))
        bluefoam_df = read_file(directory + "/bluefoam.csv")
        Tbf, Tbferr = mat_temp(bluefoam_df, params_df)
        print("T_bluefoam = {:.6f} +/- {:.6f}".format(Tbf, Tbferr))
        cellphone_df = read_file(directory + "/cellphone.csv")
        Tcp, Tcperr = mat_temp(cellphone_df, params_df)
        print("T_cellphone = {:.6f} +/- {:.6f}".format(Tcp, Tcperr))
        acrylic_df = read_file(directory + "/acrylic.csv")
        Tac, Tacerr = mat_temp(acrylic_df, params_df)
        print("T_acrylic = {:.6f} +/- {:.6f}".format(Tac, Tacerr))


        hl_df = read_file(directory + "/hot_load_after.csv")
        ns_df = read_file(directory + "/noise_load_after.csv")
        Tn, Tnerr = mat_temp(ns_df, params_df)
        T, Terr = mat_temp(hl_df, params_df)
        print("T_noise = {:.6f} +/- {:.6f}".format(Tn, Tnerr))
        print("T_load = {:.6f} +/- {:.6f}".format(T, Terr))

        d = {
            "Material" : ["hand", "2 hands", "blackbody", "blue foam", "cellphone", "acrylic", "noise", "load"],
            "T" : [Thand, Thand2, Tbb, Tbf, Tcp, Tac, Tn, T],
            "Terr": [Thanderr, Thand2err, Tbberr, Tbferr, Tcperr, Tacerr, Tnerr, Terr]
             }
        df = pd.DataFrame(data=d)
        df.to_csv(f"{directory}data.csv",index=0)


if __name__ == "__main__":
    get_materials()