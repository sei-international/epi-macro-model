# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:10:09 2021

@author: Eric
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from io_model import IO_model
from common import Window, get_datetime, timesteps_between_dates, get_datetime_array

x = IO_model(r'io_config.yaml')

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

start_datetime = get_datetime(common_params['time']['start date'])
nsteps = timesteps_between_dates(common_params['time']['start date'], common_params['time']['end date'], x.days_per_timestep)
macro_datetime_array = get_datetime_array(common_params['time']['start date'], common_params['time']['end date'], x.days_per_timestep)

soc_dist_windows = []
for window in common_params['social distance']:
    if window['apply']:
        ts_start = timesteps_between_dates(common_params['time']['start date'], window['start date'], x.days_per_timestep)
        ts_end = timesteps_between_dates(common_params['time']['start date'], window['end date'], x.days_per_timestep)
        soc_dist_windows.append(Window(ts_start, ts_end,
                                       round(window['ramp up for']/x.days_per_timestep),
                                       round(window['ramp down for']/x.days_per_timestep),
                                       window['effectiveness']))

# Global GDP
global_GDP_points = common_params['global-GDP-trajectory']
timesteps_array = np.array(range(0,nsteps))
global_GDP_ts = np.empty(len(global_GDP_points))
global_GDP_rate = np.empty(len(global_GDP_points))
for i in range(0,len(global_GDP_points)):
    global_GDP_ts[i] = timesteps_between_dates(common_params['time']['start date'], global_GDP_points[i][0], x.days_per_timestep)
    global_GDP_rate[i] = global_GDP_points[i][1]
global_GDP_gr = (1 + np.interp(timesteps_array, global_GDP_ts, global_GDP_rate))**(1/x.timesteps_per_year) - 1

util = pd.DataFrame(columns = x.sectors, index = range(0,nsteps))
util.loc[0] = 1
VA = pd.DataFrame(columns = x.sectors, index = range(0,nsteps))
VA.loc[0] = x.get_value_added()
GDP = np.zeros(nsteps)
GDP[0] = x.get_value_added().sum()
GDP_gr = np.zeros(nsteps)
GDP_gr[0] = x.gamma_ann
GDP_ref = np.zeros(nsteps)
GDP_ref[0] = GDP[0]
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
    PHA_social_distancing = 0
    for w in soc_dist_windows:
        PHA_social_distancing += w.window(t)
    
    if macro_datetime_array[t] < epi_datetime_array[0]:
        hosp_index = 1
    else:
        hosp_index = hospitalization_index[epi_datetime_array.index(macro_datetime_array[t])]
        
    # Advance one timestep
    x.update(global_GDP_gr[t-1], hosp_index, PHA_social_distancing)
    
    GDP[t] = x.get_value_added().sum()
    GDP_gr[t] = (GDP[t]/GDP[t-1])**x.timesteps_per_year - 1
    GDP_ref[t] = (1 + x.gamma) * GDP_ref[t-1]
    X[t] = x.X.sum()
    X_gr[t] = (X[t]/X[t-1])**x.timesteps_per_year - 1
    F[t] = x.F.sum()
    F_gr[t] = (F[t]/F[t-1])**x.timesteps_per_year - 1
    u_ave[t] = x.Y.sum()/x.Ypot.sum()
    I[t] = x.I
    I_gr[t] = (I[t]/I[t-1])**x.timesteps_per_year - 1
    VA.loc[t] = x.get_value_added()
    util.loc[t] = x.u

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

# plt.plot(np.cumprod(1 + global_GDP_gr))
# plt.show()

units_conversion = x.monetary_units['scale'] * x.timesteps_per_year

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * GDP)
plt.plot(macro_datetime_array, units_conversion * GDP_ref)
ax.set_ylim([0,None])
plt.title('GDP')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * X)
ax.set_ylim([0,None])
plt.title('Exports')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * F)
ax.set_ylim([0,None])
plt.title('Final Demand, Excluding Investment')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * I)
ax.set_ylim([0,None])
plt.title('Investment')
plt.ylabel(x.monetary_units['currency'])
plt.show()


# ax = plt.gca()
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)
# ax.set_ylim([-5,5])
# plt.plot(macro_datetime_array, 100 * X_gr)
# plt.plot(macro_datetime_array, 100 * GDP_gr)
# plt.plot(macro_datetime_array, 100 * F_gr)
# plt.plot(macro_datetime_array, 100 * I_gr)
# plt.legend(['X','GDP','F','I'])
# plt.ylabel('%/year')
# plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.set_ylim([0.5,1.1])
plt.plot(macro_datetime_array, u_ave)
plt.title('Capacity Utilization')
plt.show()

VA.plot()

VA_perc = VA.divide(VA.sum(1), 0)
VA_perc.plot.area()

util.plot()
