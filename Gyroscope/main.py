import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df01 = pd.read_csv("0_1.txt",names=["v","f","std"])
df1 = pd.read_csv("1.txt",names=["v","f","std"])

plt.scatter(np.abs(df1.v),df1.f,label="1ms")
plt.scatter(np.abs(df01.v),df01.f,label="0.1 ms")
plt.xlabel("v")
plt.ylabel("f")
plt.legend()
plt.show()