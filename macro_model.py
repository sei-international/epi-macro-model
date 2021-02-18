# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:10:09 2021

@author: Eric
"""
import numpy as np
import matplotlib.pyplot as plt
from io_model import IO_model

x = IO_model(r'io_config.yaml')
GDP = np.zeros(20)
GDP[0] = x.get_value_added().sum()
GDP_gr = np.zeros(20)
GDP_gr[0] = x.gamma
X = np.zeros(20)
X[0] = x.X.sum()
X_gr = np.zeros(20)
X_gr[0] = x.gamma
F = np.zeros(20)
F[0] = x.F.sum()
F_gr = np.zeros(20)
F_gr[0] = x.gamma
u_ave = np.zeros(20)
u_ave[0] = 1
for t in range(1, 20):
    x.update()
    GDP[t] = x.get_value_added().sum()
    GDP_gr[t] = GDP[t]/GDP[t-1] - 1
    X[t] = x.X.sum()
    X_gr[t] = X[t]/X[t-1] - 1
    F[t] = x.F.sum()
    F_gr[t] = F[t]/F[t-1] - 1
    u_ave[t] = x.Y.sum()/x.Ypot.sum()
   
plt.plot(GDP)
plt.show()

plt.plot(GDP_gr)
plt.show()

plt.plot(X_gr)
plt.show()

plt.plot(F_gr)
plt.show()

plt.plot(u_ave)
plt.show()
