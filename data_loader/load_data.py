from scipy.io import loadmat
import numpy as np

m = loadmat("C:/Users/shertheus/Downloads/sims/sim1.mat")
a = m["ts"]
a = np.mean(a, axis=1)
a = np.mean(a, axis=2)
print(a)
