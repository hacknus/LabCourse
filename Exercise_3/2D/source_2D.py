import numpy as np
from constants_2D import power,rel_x,rel_y


def source(number_z):
    S = np.zeros(number_z**2)
    S[int(rel_x*(number_z-1)*number_z)+int((number_z-1)*rel_y)]=power
    S[int((rel_x*(number_z-1)+1)*number_z)+int((number_z-1)*rel_y)]
    return S
