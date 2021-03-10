# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:25:57 2021

@author: Eric
"""

from numpy import identity as np_identity, divide as np_divide, dot as np_dot, linalg as la, \
    diag as np_diag, matmul as np_matmul, multiply as np_multiply, maximum as np_maximum, minimum as np_minimum
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
        
        self.gov_aut_frac = io_params['govt-expend-autonomous-fraction']

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
        self.ident = np_identity(len(self.sectors))
        
        # Initial and maximum utilizations default to 1.0
        self.u = Series(data = 1.00, index = self.sectors)
        u_init_is_specified = 'initial-utilization' in io_params['sectors']
        if u_init_is_specified:
            specified_u_init = io_params['sectors']['initial-utilization']
        self.u_max = Series(data = 1.00, index = self.sectors)
        u_max_is_specified = 'max-utilization' in io_params['sectors']
        if u_max_is_specified:
            specified_u_max = io_params['sectors']['max-utilization']
        for s in self.sectors:
            subs = sector_aggr[s]
            self.findem.loc[s] = findem_data.loc[subs].sum()
            self.wages.loc[s] = wages_data[subs].sum()
            if u_init_is_specified:
                self.u[s] = specified_u_init[s]
            if u_max_is_specified:
                self.u_max[s] = specified_u_max[s]
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
        self.dom_share = 1 - np_divide(self.M, tot_intermed_dmd + dom_findem - self.G)
        self.m_A = 1 - self.dom_share
        # Correct household import shares for minimum domestic sourcing
        self.m_H = 1 - self.dom_share
        for s in min_hh_dom_shares:
            self.m_H[s] = (1 - self.dom_share[s])/(1 - min_hh_dom_shares[s])
            
        self.F = self.desired_final_demand()

        # Technical matrix & Leontief matrix
        self.A = np_divide(self.interind, self.Y)
        # Leontief matrix is corrected by the import propensity for intermediate goods
        self.Leontief = (self.ident - np_matmul((self.ident - np_diag(self.m_A)), self.A)).astype('float')
        self.Leontief = DataFrame(la.inv(self.Leontief), index = self.sectors, columns = self.sectors)
        
        
        self.I = sum(inv_expend)/self.timesteps_per_year
        theta = np_divide(inv_expend, self.I)
        self.theta_dom = theta * self.dom_share
        self.theta_imp = theta - self.theta_dom
        
        self.W = self.wages[io_params['wages']]      
        
        # Capital depreciation (annual rate)
        self.delta_ann = np_divide(1.0, Series(io_params['sectors']['typical-lifetime'], index = sector_aggr))
        # Correct for time steps per year
        self.delta = (1 + self.delta_ann)**(1/self.timesteps_per_year) - 1
        
        # Get target growth rate (annual rate)
        self.gamma_ann = io_params['target-growth-rate']
        # Correct for time steps per year
        self.gamma = (1 + self.gamma_ann)**(1/self.timesteps_per_year) - 1
        self.Ygr = Series(data = self.gamma, index = self.sectors)
        self.Wgr = Series(data = self.gamma, index = self.sectors)
        self.GDPgr = self.gamma
        self.GDPgr_smoothed = self.gamma
        
        # Initialize potential output and 
        epsilon = self.gamma * io_params['calib']['threshold-util']/(1 - io_params['calib']['threshold-util'])
        self.phi = max(0, 1 - epsilon)
        self.Ypot = np_divide(self.Y, self.u)
        
        # Create calibrated capital productivities
        g = self.get_gross_inv_rate()
        tot_intermed_expend = self.interind.sum(0)
        profshare = np_divide(self.Y - tot_intermed_expend - self.W, self.Y)
        denom = np_multiply(profshare, np_multiply(g, self.Ypot)).sum()
        payback_period = (1 + self.gamma) * self.I/denom
        self.capprod = np_divide(1.0, payback_period * profshare)
        
        #-----------------------------------------------
        # Public health parameters
        #-----------------------------------------------
        self.soc_dist_sens = io_params['public-health-response']['social-distance-sensitivity']
        self.trav_ban_sens = io_params['public-health-response']['travel-ban-sensitivity']
        self.hosp_sens = io_params['hospitalization-sensitivity']
        
    def get_gross_inv_rate(self):
        num = self.u * (1 + self.gamma)
        den = 1 - self.phi * (1 - self.u)
        # Don't allow gross investment to fall below replacement rate
        return np_maximum(0, num/den - 1) + self.delta
    
    def get_value_added(self):
        return np_multiply(self.Y, 1 - self.A.sum(0))
    
    def desired_final_demand(self):
        return self.H - np_multiply(self.m_H, self.H - self.H0) + self.G + self.X
        
    def update_desired_final_demand(self, delta_global_GDP_gr, hospitalization_index, soc_distance, trav_ban):
        self.H0 *= (1 + self.gamma) # Assume this "baseline" level follows expected growth
        self.H = (1 + self.Wgr) * (self.H - self.H0) + self.H0
        self.G *= self.gov_aut_frac * (1 + self.gamma) + (1 - self.gov_aut_frac) * (1 + self.GDPgr_smoothed)
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
        
    def update_utilization(self):
        g = self.get_gross_inv_rate()
        self.I = np_multiply(np_divide(g, self.capprod), self.Ypot).sum()
        dom_supply = np_dot(self.Leontief, self.F + self.theta_dom * self.I)
        # Calculate next-period Ypot, but don't update self.Ypot yet
        Ystar = np_multiply(1 + g - self.delta, self.Ypot)
        ustar = np_divide(dom_supply, Ystar)
        self.u = np_minimum(self.u_max, ustar)
        B = self.ident - np_matmul(self.ident - np_diag(self.m_A), self.A)
        deltaF = np_dot(B, np_multiply(ustar - self.u, Ystar))
        self.F -= deltaF
        # Update Ypot and Y
        self.Ypot = Ystar
        Yprev = self.Y
        self.Y = np_multiply(self.u, self.Ypot)
        self.Ygr = np_divide(self.Y, Yprev) - 1
        
    def update_wages(self):
        Wprev = self.W.sum()
        self.W = np_multiply(self.W, 1 + self.Ygr)
        self.Wgr = self.W.sum()/Wprev - 1
        
    def update(self, delta_global_GDP_gr, hospitalization_index = 1.0, soc_distance = 0.0, trav_ban = 0.0):
        # These must occur in this order:
        VAprev = self.get_value_added().sum()
        self.update_desired_final_demand(delta_global_GDP_gr, hospitalization_index, soc_distance, trav_ban)
        self.update_utilization()
        self.update_wages()
        self.GDPgr = self.get_value_added().sum()/VAprev - 1
        self.GDPgr_smoothed = self.GDPgr_smoothed + (1/self.timesteps_per_year) * (self.GDPgr - self.GDPgr_smoothed)
        self.t += 1
        