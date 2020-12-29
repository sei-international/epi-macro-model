import numpy as np
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
    def __init__(self, config_file):
        
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

        # Calculate the mean infectious period based on the matrix model
        self.mean_infectious_period = 0
        P = 1
        for i in range(1,self.infective_time_period):
            self.mean_infectious_period += (i + 1) * P * self.rd_I[i - 1]
            P *= 1 - self.rd_I[i - 1]

    def p_spread(self, num_inf, pub_health_factor):
        kn = self.k * num_inf
        return 1 - (1/(1 + pub_health_factor * self.R0/kn))^kn



