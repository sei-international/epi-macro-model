# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:25:57 2021

@author: Eric
"""

import numpy as np
import numpy.linalg as la
from pandas import Series, DataFrame, read_csv
import yaml

class IO_model:
    def __init__(self, io_params_file: str):
    
        with open(io_params_file) as file:
            io_params = yaml.full_load(file)
        
        csv_data = read_csv(io_params['input-file']['name'],
                               sep = io_params['input-file']['delimiter'],
                               quotechar = io_params['input-file']['quote-character'],
                               index_col = 0)
        
        self.days_per_timestep = io_params['days-per-time-step']
        self.timesteps_per_year = round(365/io_params['days-per-time-step'])
        self.t = 0 # Initialize timestep counter

        nsector = io_params['sectors']['count']
        interind_data = csv_data.iloc[:nsector,:nsector]/self.timesteps_per_year
        sector_names_data = list(interind_data)
        findem_data = csv_data.iloc[:nsector,nsector:]/self.timesteps_per_year
        wages_data = csv_data.loc[io_params['wages']][sector_names_data].transpose()/self.timesteps_per_year
        self.monetary_units = io_params['monetary-units']

        non_tradeables = io_params['sectors']['non-tradeables']
        tradeables = io_params['sectors']['tradeables']
        sector_aggr = {}
        sector_aggr.update(non_tradeables)
        sector_aggr.update(tradeables)
        self.sectors = list(sector_aggr.keys())
        self.sectors_non_tradeable = list(non_tradeables.keys())
        self.sectors_tradeable = list(tradeables.keys())
        self.interind = DataFrame(index = self.sectors, columns = self.sectors)
        self.findem = DataFrame(index = self.sectors, columns = list(findem_data))
        self.wages = DataFrame(index = self.sectors, columns = [io_params['wages']])

        # For later use, create an identity matrix
        self.ident = np.identity(len(self.sectors))
        
        for s in self.sectors:
            subs = sector_aggr[s]
            self.findem.loc[s] = findem_data.loc[subs].sum()
            self.wages.loc[s] = wages_data[subs].sum()
            for s2 in self.sectors:
                subs2 = sector_aggr[s2]
                # The way indexing works, have to transpose the indices
                self.interind[s2][s] = interind_data.loc[subs][subs2].sum().sum()
                self.interind[s][s2] = interind_data.loc[subs2][subs].sum().sum()
        
        self.G = self.findem[io_params['final-demand']['government']]
        self.H = self.findem[io_params['final-demand']['household']]
        self.H0 = Series(data = 0.0, index = self.H.index)
        if 'min-hh-dom-share' in io_params['sectors']:
            min_hh_dom_shares = io_params['sectors']['min-hh-dom-share']
            for s in min_hh_dom_shares:
                self.H0[s] = min_hh_dom_shares[s] * self.H[s]

        # Ensure non-tradeables have zero X & M
        self.X = self.findem[io_params['final-demand']['exports']]
        self.M = abs(self.findem[io_params['final-demand']['imports']])
        self.X[non_tradeables] = 0.0
        self.M[non_tradeables] = 0.0
        
        self.global_GDP_elast_of_X = io_params['sectors']['global-GDP-elasticity-of-exports']
        
        tot_intermed_dmd = self.interind.sum(1)
        inv_expend = self.findem[io_params['final-demand']['investment']]
        dom_findem = self.H + self.G + inv_expend
        self.Y = tot_intermed_dmd + dom_findem + self.X - self.M

        # Domestic production as share of total domestic final demand: assume G domestically supplied
        self.dom_share = 1 - np.divide(self.M, tot_intermed_dmd + dom_findem - self.G)
        self.m_A = 1 - self.dom_share
        # Correct household import shares for minimum domestic sourcing
        self.m_H = 1 - self.dom_share
        for s in min_hh_dom_shares:
            self.m_H[s] = (1 - self.dom_share[s])/(1 - min_hh_dom_shares[s])
            
        self.F = self.desired_final_demand()

        # Technical matrix & Leontief matrix
        self.A = np.divide(self.interind, self.Y)
        # Leontief matrix is corrected by the import propensity for intermediate goods
        self.Leontief = (self.ident - np.matmul((self.ident - np.diag(self.m_A)), self.A)).astype('float')
        self.Leontief = DataFrame(la.inv(self.Leontief), index = self.sectors, columns = self.sectors)
        
        
        self.I = sum(inv_expend)/self.timesteps_per_year
        theta = np.divide(inv_expend, self.I)
        self.theta_dom = theta * self.dom_share
        self.theta_imp = theta - self.theta_dom
        
        self.W = self.wages[io_params['wages']]      
        
        # Capital depreciation (annual rate)
        self.delta_ann = np.divide(1.0, Series(io_params['sectors']['typical-lifetime'], index = sector_aggr))
        # Correct for time steps per year
        self.delta = (1 + self.delta_ann)**(1/self.timesteps_per_year) - 1
        
        # Get target growth rate (annual rate)
        self.gamma_ann = io_params['target-growth-rate']
        # Correct for time steps per year
        self.gamma = (1 + self.gamma_ann)**(1/self.timesteps_per_year) - 1
        self.Ygr = Series(data = self.gamma, index = self.sectors)
        self.Wgr = Series(data = self.gamma, index = self.sectors)
        
        # Initialize utilization and related variables
        self.u = Series(data = io_params['sectors']['initial-utilization'], index = self.sectors)
        epsilon = self.gamma * io_params['calib']['threshold-util']/(1 - io_params['calib']['threshold-util'])
        self.phi = 1 - epsilon
        self.Ypot = np.divide(self.Y, self.u)
        
        # Create calibrated capital productivities
        g = self.get_gross_inv_rate()
        tot_intermed_expend = self.interind.sum(0)
        profshare = np.divide(self.Y - tot_intermed_expend - self.W, self.Y)
        denom = np.multiply(profshare, np.multiply(g, self.Ypot)).sum()
        payback_period = (1 + self.gamma) * self.I/denom
        self.capprod = np.divide(1.0, payback_period * profshare)
        
        #-----------------------------------------------
        # Public health parameters
        #-----------------------------------------------
        self.soc_dist_sens = io_params['public-health-response']['social-distance-sensitivity']
        self.trav_ban_sens = io_params['public-health-response']['travel-ban-sensitivity']
        self.hosp_sens = io_params['hospitalization-sensitivity']
        self.max_util = 1.05 # Max utilization exceedance
        
    def get_gross_inv_rate(self):
        num = self.u * (1 + self.gamma)
        den = 1 - self.phi * (1 - self.u)
        # Don't allow gross investment to fall below replacement rate
        return np.maximum(0, num/den - 1) + self.delta
    
    def get_value_added(self):
        return np.multiply(self.Y, 1 - self.A.sum(0))
    
    def desired_final_demand(self):
        return self.H - np.multiply(self.m_H, self.H - self.H0) + self.G + self.X
        
    def update_desired_final_demand(self, delta_global_GDP_gr, hospitalization_index, soc_distance, trav_ban):
        self.H0 *= (1 + self.gamma) # Assume this "baseline" level follows expected growth
        self.H *= (1 + self.Wgr)
        self.G *= (1 + self.gamma)
        for s in self.sectors_tradeable:
            self.X[s] *= 1 + self.gamma + delta_global_GDP_gr * self.global_GDP_elast_of_X[s]
        self.F = self.desired_final_demand()
        # Now correct for social distancing and travel bans
        for s in self.soc_dist_sens:
            self.F *= 1 - soc_distance * self.soc_dist_sens[s]
        for s in self.trav_ban_sens:
            self.F *= 1 - trav_ban * self.trav_ban_sens[s]
        # And hospitalization sensitivity
        for s in self.hosp_sens:
            self.F *= 1 + (hospitalization_index - 1) * self.hosp_sens[s]
        
    def update_utilization(self, hospitalization_index):
        g = self.get_gross_inv_rate()
        self.I = np.multiply(np.divide(g, self.capprod), self.Ypot).sum()
        dom_supply = np.dot(self.Leontief, self.F + self.theta_dom * self.I)
        # Calculate next-period Ypot, but don't update self.Ypot yet
        Ystar = np.multiply(1 + g - self.delta, self.Ypot)
        ustar = np.divide(dom_supply, Ystar)
        self.u = np.minimum(1, ustar)
        # Correct for sectors affected by hospitalization, which might exceed one
        for s in self.hosp_sens:
            effective_util = 1 + (hospitalization_index - 1) * self.hosp_sens[s]
            self.u[s] = min(max(1, min(self.max_util, effective_util)), ustar[s])
        # A matrix used in subsequent calculations: Keep here to keep expressions manageable
        B = self.ident - np.matmul(self.ident - np.diag(self.m_A), self.A)
        deltaF = np.dot(B, np.multiply(ustar - self.u, Ystar))
        self.F -= deltaF
        # Update Ypot and Y
        self.Ypot = Ystar
        Yprev = self.Y
        self.Y = np.multiply(self.u, self.Ypot)
        self.Ygr = np.divide(self.Y, Yprev) - 1
        
    def update_wages(self):
        Wprev = self.W.sum()
        self.W = np.multiply(self.W, 1 + self.Ygr)
        self.Wgr = self.W.sum()/Wprev - 1
        
    def update(self, delta_global_GDP_gr, hospitalization_index = 1.0, soc_distance = 0.0, trav_ban = 0.0):
        # These must occur in this order:
        self.update_desired_final_demand(delta_global_GDP_gr, hospitalization_index, soc_distance, trav_ban)
        self.update_utilization(hospitalization_index)
        self.update_wages()
        self.t += 1
        