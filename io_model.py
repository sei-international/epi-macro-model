# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:25:57 2021

@author: Eric
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
import yaml

class IO_model:
    def __init__(self, io_params_file: str):
    
        with open(io_params_file) as file:
            io_params = yaml.full_load(file)
        
        csv_data = pd.read_csv(io_params['input-file']['name'],
                               sep = io_params['input-file']['delimiter'],
                               quotechar = io_params['input-file']['quote-character'],
                               index_col = 0)
        
        self.days_per_timestep = io_params['days-per-time-step']
        self.timesteps_per_year = round(365/io_params['days-per-time-step'])
        self.t = 0 # Initialize timestep counter

        nsector = io_params['sectors']['count']
        interind_data = csv_data.iloc[:nsector,:nsector]
        sector_names_data = list(interind_data)
        findem_data = csv_data.iloc[:nsector,nsector:]
        wages_data = csv_data.loc[io_params['wages']][sector_names_data].transpose()

        non_tradeables = io_params['sectors']['non-tradeables']
        tradeables = io_params['sectors']['tradeables']
        sector_aggr = {}
        sector_aggr.update(non_tradeables)
        sector_aggr.update(tradeables)
        self.sectors = list(sector_aggr.keys())
        self.sectors_non_tradeable = list(non_tradeables.keys())
        self.sectors_tradeable = list(tradeables.keys())
        self.interind = pd.DataFrame(index = self.sectors, columns = self.sectors)
        self.findem = pd.DataFrame(index = self.sectors, columns = list(findem_data))
        self.wages = pd.DataFrame(index = self.sectors, columns = [io_params['wages']])

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
        
        self.G = self.findem[io_params['final-demand']['government']]/self.timesteps_per_year
        self.H = self.findem[io_params['final-demand']['household']]/self.timesteps_per_year
        self.H0 = pd.Series(data = 0.0, index = self.H.index)
        min_hh_dom_shares = io_params['sectors']['min-hh-dom-share']
        for s in min_hh_dom_shares:
            self.H0[s] = min_hh_dom_shares[s] * self.H[s]

        # Ensure non-tradeables have zero X & M
        self.X = self.findem[io_params['final-demand']['exports']]/self.timesteps_per_year
        self.M = abs(self.findem[io_params['final-demand']['imports']])/self.timesteps_per_year
        self.X[non_tradeables] = 0.0
        self.M[non_tradeables] = 0.0
        
        tot_intermed_dmd = self.interind.sum(1)
        inv_expend = self.findem[io_params['final-demand']['investment']]
        dom_findem = self.H + self.G + inv_expend
        self.output = tot_intermed_dmd + dom_findem + self.X - self.M
        self.Y = self.output

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
        self.Leontief = (self.ident - np.matmul((self.ident - np.diag(self.m_A)), self.A)).convert_dtypes()
        self.Leontief = pd.DataFrame(la.inv(self.Leontief), index = self.sectors, columns = self.sectors)
        
        
        self.I = sum(inv_expend)/self.timesteps_per_year
        theta = np.divide(inv_expend, self.I)
        self.theta_dom = theta * self.dom_share
        self.theta_imp = theta - self.theta_dom
        
        self.W = self.wages[io_params['wages']]      
        
        # Capital depreciation (annual rate)
        self.delta_ann = np.divide(1.0, pd.Series(io_params['sectors']['typical-lifetime'], index = sector_aggr))
        # Correct for time steps per year
        self.delta = (1 + self.delta_ann)**(1/self.timesteps_per_year) - 1
        
        # Get target growth rate (annual rate)
        self.gamma_ann = io_params['target-growth-rate']
        # Correct for time steps per year
        self.gamma = (1 + self.gamma_ann)**(1/self.timesteps_per_year) - 1
        self.Ygr = pd.Series(data = self.gamma, index = self.sectors)
        self.Wgr = pd.Series(data = self.gamma, index = self.sectors)
        
        # Initialize utilization and related variables
        self.u = pd.Series(data = io_params['sectors']['initial-utilization'], index = self.sectors)
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
        
    def get_gross_inv_rate(self):
        num = self.u * (1 + self.gamma)
        den = 1 - self.phi * (1 - self.u)
        # Don't allow gross investment to fall below replacement rate
        return np.maximum(0, num/den - 1) + self.delta
    
    def get_value_added(self):
        return self.Y - np.matmul(self.A, self.Y)
    
    def desired_final_demand(self):
        return self.H - np.multiply(self.m_H, self.H - self.H0) + self.G + self.X
        
    def update_desired_final_demand(self):
        # TODO: Replace this stub, where everything grows at the constant target rate
        self.H0 *= (1 + self.gamma) # Assume this "baseline" level follows expected growth
        self.H *= (1 + self.Wgr)
        self.G *= (1 + self.gamma)
        year = self.t/self.timesteps_per_year
        if year < 7 or year > 12:
            Xwindow = 1
        elif year < 8:
            Xwindow = 1 - 0.8 * (year - 7)
        elif year > 8:
            Xwindow = 1 - 0.8 * (12 - year)/4
        else:
            Xwindow = 0.2
        self.X *= (1 + self.gamma * Xwindow)
        self.F = self.desired_final_demand()
        
    def update_utilization(self):
        g = self.get_gross_inv_rate()
        self.I = np.multiply(np.divide(g, self.capprod), self.Ypot).sum()
        dom_supply = np.dot(self.Leontief, self.F + self.theta_dom * self.I)
        # Calculate next-period Ypot, but don't update self.Ypot yet
        Ystar = np.multiply(1 + g - self.delta, self.Ypot)
        ustar = np.divide(dom_supply, Ystar)
        self.u = np.minimum(1, ustar)
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
        
    def update(self):
        # These must occur in this order:
        self.update_desired_final_demand()
        self.update_utilization()
        self.update_wages()
        self.t += 1
        