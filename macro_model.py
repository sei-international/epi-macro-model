# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:10:09 2021

@author: Eric
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io_model import IO_model

x = IO_model(r'io_config.yaml')
nyears = 20
nsteps = nyears * x.timesteps_per_year

VA = pd.DataFrame(columns = x.sectors, index = range(0,nsteps))
VA.loc[0] = x.get_value_added()
GDP = np.zeros(nsteps)
GDP[0] = x.get_value_added().sum()
GDP_gr = np.zeros(nsteps)
GDP_gr[0] = x.gamma_ann
X = np.zeros(nsteps)
X[0] = x.X.sum()
X_gr = np.zeros(nsteps)
X_gr[0] = x.gamma_ann
F = np.zeros(nsteps)
F[0] = x.F.sum()
F_gr = np.zeros(nsteps)
F_gr[0] = x.gamma_ann
I = np.zeros(nsteps)
I[0] = x.I
I_gr = np.zeros(nsteps)
I_gr[0] = x.gamma_ann
u_ave = np.zeros(nsteps)
u_ave[0] = 1
for t in range(1, nsteps):
    x.update()
    GDP[t] = x.get_value_added().sum()
    GDP_gr[t] = (GDP[t]/GDP[t-1])**x.timesteps_per_year - 1
    X[t] = x.X.sum()
    X_gr[t] = (X[t]/X[t-1])**x.timesteps_per_year - 1
    F[t] = x.F.sum()
    F_gr[t] = (F[t]/F[t-1])**x.timesteps_per_year - 1
    u_ave[t] = x.Y.sum()/x.Ypot.sum()
    I[t] = x.I
    I_gr[t] = (I[t]/I[t-1])**x.timesteps_per_year - 1
    VA.loc[t] = x.get_value_added()

plt.plot(GDP)
plt.show()

plt.plot(X_gr)
plt.plot(GDP_gr)
plt.plot(F_gr)
plt.plot(I_gr)
plt.legend(['X','GDP','F','I'])
plt.show()

plt.plot(u_ave)
plt.show()

VA_perc = VA.divide(VA.sum(1), 0)
VA_perc.plot.area()
