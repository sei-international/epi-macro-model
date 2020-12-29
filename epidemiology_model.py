import numpy as np
import matplotlib.pyplot as plt
import yaml
from epidemiology_functions import Window, SEIR_matrix

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

# Load epidemiological model object
epi = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'])

start_time = common_params['time']['start']
end_time = common_params['time']['end']
time_step = common_params['time']['step']

beds_per_1000 = common_params['beds per 1000']
normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
max_reduction_in_normal_bed_occupancy= common_params['bed occupancy']['max reduction']

avoid_elective_operations= common_params['avoid elective operations']

isolate_cases_window = Window(common_params['isolate cases']['start at'],
                              common_params['isolate cases']['end at'],
                              common_params['isolate cases']['ramp up for'],
                              common_params['isolate cases']['ramp down for'])

social_distancing_window = Window(common_params['social distance']['start at'],
                                  common_params['social distance']['end at'],
                                  common_params['social distance']['ramp up for'],
                                  common_params['social distance']['ramp down for'])

# Initialize values for indicator graphs
deaths = 0
cumulative_cases = 0
susceptible_over_time = np.zeros(end_time - start_time + 1)
susceptible_over_time[0] = epi.S
exposed_over_time = np.zeros(end_time - start_time + 1)
exposed_over_time[0] = epi.E[1]
infective_over_time = np.zeros(end_time - start_time + 1)
infective_over_time[0] = epi.I_nr[1] + epi.I_r[1]
deaths_over_time = np.zeros(end_time - start_time + 1)
deaths_over_time[0] = 0
recovered_over_time = np.zeros(end_time - start_time + 1)
recovered_over_time[0] = 0
initial_population = epi.N
comm_spread_frac_over_time = np.zeros(end_time - start_time + 1)
comm_spread_frac_over_time[0] = epi.comm_spread_frac

for i in range(start_time, end_time, time_step):

	# Public health measures
    if common_params['social distance']['apply']:
        PHA_social_distancing = common_params['social distance']['effectiveness'] * social_distancing_window.window(i)
    else:
        PHA_social_distancing = 0
    if common_params['isolate cases']['apply to visible cases']:
        PHA_isolate_visible_cases = (1 - epi.invisible_fraction) * isolate_cases_window.window(i)
    else:
        PHA_isolate_visible_cases = 0
    if common_params['isolate cases']['apply to infectious cases']:
        PHA_isolate_infectious_cases = common_params['isolate cases']['fraction of infectious cases identified'] * isolate_cases_window.window(i)
    else:
        PHA_isolate_infectious_cases = 0
    PHA_isolate_cases = max(PHA_isolate_visible_cases, PHA_isolate_infectious_cases)
    public_health_adjustment = (1 - PHA_social_distancing) * (1 - PHA_isolate_cases)
    
    # Beds and Mortality
    if avoid_elective_operations:
        bed_occupancy_factor = (1 - social_distancing_window.window(i) * max_reduction_in_normal_bed_occupancy)
    else:
        bed_occupancy_factor = 1
    bed_occupancy_fraction = bed_occupancy_factor * normal_bed_occupancy_fraction
    
    # Run the model for one time step
    epi.update(public_health_adjustment, bed_occupancy_fraction, beds_per_1000)
    
    # Update values for indicator graphs
    deaths += epi.new_deaths
    cumulative_cases_per_1000 = 1000 * cumulative_cases/initial_population
    susceptible_over_time[i] = epi.S
    exposed_over_time[i] = np.sum(epi.E)
    infective_over_time[i] = epi.Itot
    deaths_over_time[i] = deaths
    recovered_over_time[i] = epi.R
    new_visible_cases = (1 - epi.invisible_fraction) * (epi.I_nr[1] + epi.I_r[1])
    cumulative_cases += new_visible_cases
    comm_spread_frac_over_time[i] = epi.comm_spread_frac

plt.stackplot(range(start_time, end_time, time_step),
              susceptible_over_time[start_time:end_time],
              exposed_over_time[start_time:end_time],
              infective_over_time[start_time:end_time],
              recovered_over_time[start_time:end_time],
              deaths_over_time[start_time:end_time],
              labels=['susceptible','exposed','infected','recovered','died'])
plt.legend(loc='lower right')
plt.show()

plt.plot(deaths_over_time[start_time:end_time])
plt.ylabel('cumulative deaths')
plt.show()

plt.plot(comm_spread_frac_over_time[start_time:end_time])
plt.ylabel('community spread')
plt.show()
