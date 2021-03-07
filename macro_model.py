# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:10:09 2021

@author: Eric
"""
from numpy import array as np_array, empty as np_empty, zeros as np_zeros, interp as np_interp
from pandas import DataFrame
import yaml
from io_model import IO_model
from common import Window, timesteps_between_dates, get_datetime_array

def macroeconomic_model(epi_datetime_array, hospitalization_index):
    x = IO_model(r'io_config.yaml')
    
    with open(r'common_params.yaml') as file:
        common_params = yaml.full_load(file)
    
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
    travel_ban_windows = []
    for window in common_params['international travel restrictions']:
        if window['apply'] and window['ban']:
            ts_start = timesteps_between_dates(common_params['time']['start date'], window['start date'], x.days_per_timestep)
            ts_end = timesteps_between_dates(common_params['time']['start date'], window['end date'], x.days_per_timestep)
            travel_ban_windows.append(Window(ts_start, ts_end,
                                           round(window['ramp up for']/x.days_per_timestep),
                                           round(window['ramp down for']/x.days_per_timestep),
                                           window['effectiveness']))
    
    # Global GDP
    global_GDP_points = common_params['global GDP trajectory']
    global_GDP_gr_trend = (1 + global_GDP_points[len(global_GDP_points) - 1][1])**(1/x.timesteps_per_year) - 1
    timesteps_array = np_array(range(0,nsteps))
    global_GDP_ts = np_empty(len(global_GDP_points))
    global_GDP_rate = np_empty(len(global_GDP_points))
    for i in range(0,len(global_GDP_points)):
        global_GDP_ts[i] = timesteps_between_dates(common_params['time']['start date'], global_GDP_points[i][0], x.days_per_timestep)
        global_GDP_rate[i] = global_GDP_points[i][1]
    global_GDP_gr = (1 + np_interp(timesteps_array, global_GDP_ts, global_GDP_rate))**(1/x.timesteps_per_year) - 1
    
    util = DataFrame(columns = x.sectors, index = range(0,nsteps))
    util.loc[0] = 1
    VA = DataFrame(columns = x.sectors, index = range(0,nsteps))
    VA.loc[0] = x.get_value_added()
    GDP = np_zeros(nsteps)
    GDP[0] = x.get_value_added().sum()
    GDP_gr = np_zeros(nsteps)
    GDP_gr[0] = x.gamma_ann
    GDP_ref = np_zeros(nsteps)
    GDP_ref[0] = GDP[0]
    X = np_zeros(nsteps)
    X[0] = x.X.sum()
    X_gr = np_zeros(nsteps)
    X_gr[0] = x.gamma_ann
    F = np_zeros(nsteps)
    F[0] = x.F.sum()
    F_gr = np_zeros(nsteps)
    F_gr[0] = x.gamma_ann
    I = np_zeros(nsteps)
    I[0] = x.I
    I_gr = np_zeros(nsteps)
    I_gr[0] = x.gamma_ann
    u_ave = np_zeros(nsteps)
    u_ave[0] = 1
    for t in range(1, nsteps):
        PHA_soc_demand_reduction_mult = 0
        for w in soc_dist_windows:
            PHA_soc_demand_reduction_mult += w.window(t)
        PHA_trav_demand_reduction_mult = 0
        for w in travel_ban_windows:
            PHA_trav_demand_reduction_mult += w.window(t)
        
        if macro_datetime_array[t] < epi_datetime_array[0]:
            hosp_index = 1
        else:
            hosp_index = hospitalization_index[epi_datetime_array.index(macro_datetime_array[t])]
            
        # Advance one timestep
        x.update(global_GDP_gr[t-1] - global_GDP_gr_trend, hosp_index, PHA_soc_demand_reduction_mult, PHA_trav_demand_reduction_mult)
        
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

    return macro_datetime_array, VA