import math
import numpy as np
from scipy import stats as st
import yaml

class Window:
    def __init__(self, start, end, ramp_up, ramp_down):
        self.start = start
        self.end = end
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
    
    # Ramp to a maximum value of 1.0 and then back down
    # Ramp time can be zero
    def window(self, time):
        if time < self.start or time > self.end:
            return 0
        elif time >= self.start + self.ramp_up and time <= self.end - self.ramp_down:
            return 1
        elif time < self.start + self.ramp_up:
            return (time - self.start)/self.ramp_up
        else:
            return (self.end - time)/self.ramp_down


class SEIR_matrix:
    def __init__(self, config_file, initial_values, geog):
        
        self.eps = 1.0e-9
        
        with open(r'seir_params.yaml') as file:
            seir_params = yaml.full_load(file)
        
        self.R0 = seir_params['R0']
        self.k = seir_params['k factor']
        self.population_at_risk_frac = seir_params['population at risk fraction']
        self.case_fatality_rate = seir_params['case fatality rate']['not at risk']
        self.case_fatality_rate_at_risk = seir_params['case fatality rate']['at risk']

        self.coeff_of_variation_i= seir_params['statistical-model']['coeff of variation of infected where spreading']
        self.invisible_fraction = seir_params['unobserved fraction of cases']
        self.fraction_of_visible_requiring_hospitalization = seir_params['fraction of observed cases requiring hospitalization']['not at risk']
        self.fraction_of_visible_requiring_hospitalization_at_risk = seir_params['fraction of observed cases requiring hospitalization']['at risk']
    
        self.rd_I = np.array(seir_params['matrix-params']['prob recover or death given infected']['not at risk'])
        self.infective_time_period = len(self.rd_I)
        self.rd_I_r = np.array(seir_params['matrix-params']['prob recover or death given infected']['at risk'])
        self.infected_E = np.array(seir_params['matrix-params']['prob infected given exposed'])
        self.exposed_time_period = len(self.infected_E)
        
        self.overflow_hospitalized_mortality_rate_factor = seir_params['case fatality rate']['overflow hospitalized mortality rate factor']

        #-------------------------------------------------------------------
        # Calculated parameters based on imported parameters
        #-------------------------------------------------------------------
        # Mean infectious period based on the matrix model
        self.mean_infectious_period = 0
        P = 1
        for i in range(1,self.infective_time_period):
            self.mean_infectious_period += (i + 1) * P * self.rd_I[i - 1]
            P *= 1 - self.rd_I[i - 1]
            
        self.baseline_hospitalized_mortality_rate = self.case_fatality_rate / self.fraction_of_visible_requiring_hospitalization
        self.baseline_hospitalized_mortality_rate_at_risk = self.case_fatality_rate_at_risk / self.fraction_of_visible_requiring_hospitalization_at_risk
        
        self.base_individual_exposure_rate = self.R0/self.mean_infectious_period

        #-------------------------------------------------------------------
        # State variables
        #-------------------------------------------------------------------
        # Total population and population at risk
        self.N = initial_values['total population']
        initial_infected = initial_values['infected fraction'] * self.N
        self.Itot = initial_infected
        self.Itot_prev = initial_infected
        self.N_prev = self.N

        # Exposed
        self.E = np.zeros(self.exposed_time_period + 1)
        # Infected, either not at risk (nr) or at risk (r)
        self.I_nr = np.zeros(self.infective_time_period + 1)
        self.I_r = np.zeros(self.infective_time_period + 1)
        
        self.E[1] = initial_values['exposed population']
        self.I_nr[1] = (1 - self.population_at_risk_frac) * initial_infected
        self.I_r[1] = self.population_at_risk_frac * initial_infected
        
        # Recovered: Assume none at initial time step
        self.R = 0
        self.recovered_pool = 0

        # Susceptible population
        self.S = self.N - self.Itot - self.R - initial_infected
        self.S_prev = self.S
        
        self.new_deaths = 0
        
        self.comm_spread_frac = initial_values['population with community spread']
        self.p_move = geog['movement probability']
        self.n_loc = geog['number of localities']

    def p_spread(self, num_inf, pub_health_factor):
        kn = self.k * num_inf
        # product of k & n (as well as R0 and public health factor) must be non-negative
        if kn < 0:
            return math.nan
        # Add small value to ensure no divide-by-zero error
        return 1 - (1/(1 + pub_health_factor * self.R0/(kn+self.eps)))**kn
    
    def mortality_rate(self, infected_fraction, bed_occupancy_fraction, beds_per_1000, at_risk):
            hospital_p_i_threshold = ((1 - bed_occupancy_fraction) * beds_per_1000 / 1000) / self.fraction_of_visible_requiring_hospitalization
            hospital_z_i = (1 / self.coeff_of_variation_i) * (hospital_p_i_threshold/(infected_fraction + self.eps) - 1)

            mean_exceedance_per_infected_fraction = self.coeff_of_variation_i * (-hospital_z_i * (1 - st.norm.cdf(hospital_z_i)) + math.exp(-0.5 * hospital_z_i ** 2) / math.sqrt(2 * math.pi))

            if at_risk:
                baseline_hospitalized_mortality_rate_to_use = self.baseline_hospitalized_mortality_rate_at_risk
            else:
                baseline_hospitalized_mortality_rate_to_use = self.baseline_hospitalized_mortality_rate
            
            return (1 - self.invisible_fraction) * self.fraction_of_visible_requiring_hospitalization * baseline_hospitalized_mortality_rate_to_use * ((1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * self.overflow_hospitalized_mortality_rate_factor)

    def social_exposure_rate(self, pub_health_factor):
        # Add eps to avoid divide by zero error if comm_spread_frac initialized to zero
        adj_base_individual_exposure_rate = self.base_individual_exposure_rate/(self.comm_spread_frac + self.eps)
        p_i = self.Itot_prev/self.N_prev
        cv_corr = self.coeff_of_variation_i**2 * self.Itot_prev/self.S_prev
        clust_corr = (1 - self.comm_spread_frac) * self.N_prev/self.S_prev
        return pub_health_factor * adj_base_individual_exposure_rate * p_i * (1 - cv_corr - clust_corr)

    def update(self, pub_health_factor, bed_occupancy_fraction, beds_per_1000):
        RD_nr = self.I_nr[self.infective_time_period]
        RD_r = self.I_r[self.infective_time_period]
        infected_fraction = sum(self.I_r + self.I_nr)/self.N

        for j in range(self.infective_time_period,1,-1):
            self.I_nr[j] = (1 - self.rd_I[j-1]) * self.I_nr[j-1]
            self.I_r[j] = (1 - self.rd_I_r[j-1]) * self.I_r[j-1]
            RD_nr = RD_nr + self.rd_I[j-1]*self.I_nr[j-1]
            RD_r = RD_r + self.rd_I_r[j-1]*self.I_r[j-1]
        self.I_nr[1] = (1 - self.population_at_risk_frac) * self.E[self.exposed_time_period]
        self.I_r[1] = self.population_at_risk_frac * self.E[self.exposed_time_period]
        for j in range(self.exposed_time_period, 1, -1):
            new_infected = self.infected_E[j-1] * self.E[j - 1]
            self.E[j] = self.E[j - 1] - new_infected
            self.I_nr[1] = self.I_nr[1] + (1 - self.population_at_risk_frac) * new_infected
            self.I_r[1] = self.I_r[1] + self.population_at_risk_frac * new_infected
        self.Itot_prev = self.Itot
        self.Itot = np.sum(self.I_nr) + np.sum(self.I_r)
        self.S_prev = self.S
        self.E[1] = self.social_exposure_rate(pub_health_factor) * self.S
        self.S -= self.E[1]
        self.N_prev = self.N
        new_deaths_nr = self.mortality_rate(infected_fraction, bed_occupancy_fraction, beds_per_1000, False) * RD_nr
        new_deaths_r = self.mortality_rate(infected_fraction, bed_occupancy_fraction, beds_per_1000, True) * RD_r
        self.new_deaths = new_deaths_nr + new_deaths_r
        self.N -= self.new_deaths
        self.recovered_pool = RD_nr + RD_r - self.new_deaths
        self.R += self.recovered_pool
        ni_addl = self.p_move * self.Itot/self.n_loc
        self.comm_spread_frac += (1 - self.comm_spread_frac) * self.p_spread(ni_addl, pub_health_factor)
        