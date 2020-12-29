import numpy as np
from scipy import stats as st
import math
import matplotlib.pyplot as plt
import yaml
from epidemiology_functions import Window, SEIR_matrix

# Load epidemiological model object
epi = SEIR_matrix(r'seir_params.yaml')

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

start_time = common_params['time']['start']
end_time = common_params['time']['end']
time_step = common_params['time']['step']

initial_population= common_params['initial']['total population']
initial_exposed = common_params['initial']['exposed population']
initial_beds_per_1000 = common_params['initial']['beds per 1000']
initial_infected_fraction = common_params['initial']['infected fraction']

avoid_elective_operations= common_params['avoid elective operations']

isolate_cases_window = Window(common_params['isolate cases']['start at'],
                              common_params['isolate cases']['end at'],
                              common_params['isolate cases']['ramp up for'],
                              common_params['isolate cases']['ramp down for'])

social_distancing_window = Window(common_params['social distance']['start at'],
                                  common_params['social distance']['end at'],
                                  common_params['social distance']['ramp up for'],
                                  common_params['social distance']['ramp down for'])

max_reduction_in_normal_bed_occupancy= 0.33
normal_bed_occupancy_fraction = 0.64

E = np.zeros(epi.exposed_time_period + 1)
I = np.zeros(epi.infective_time_period + 1)
I_r = np.zeros(epi.infective_time_period + 1)

E[1] = initial_exposed


baseline_hospitalized_mortality_rate = epi.case_fatality_rate / epi.fraction_of_visible_requiring_hospitalization
baseline_hospitalized_mortality_rate_at_risk = epi.case_fatality_rate_at_risk / epi.fraction_of_visible_requiring_hospitalization_at_risk

deaths = 0

cumulative_cases = 0

N = initial_population

N_r = initial_population * epi.population_at_risk_frac

recovered_pool = 0

susceptible_no_risk = initial_population - N_r

susceptible_at_risk = N_r

beds_per_1000 = initial_beds_per_1000

post_peak = 0

lagged_wage_gap = 0

Itot_lagged = 0

N_lagged = initial_population

S_lagged = initial_population

RD = 0

RD_r = 0

infective = 0

exposed_at_risk = initial_exposed * epi.population_at_risk_frac

exposed_no_risk = initial_exposed - (exposed_at_risk)

susceptible_over_time = np.zeros(end_time - start_time + 1)
susceptible_over_time[0] = susceptible_no_risk + susceptible_at_risk
exposed_over_time = np.zeros(end_time - start_time + 1)
exposed_over_time[0] = exposed_at_risk + exposed_no_risk
infective_over_time = np.zeros(end_time - start_time + 1)
infective_over_time[0] = 0
deaths_over_time = np.zeros(end_time - start_time + 1)
deaths_over_time[0] = 0
recovered_over_time = np.zeros(end_time - start_time + 1)
recovered_over_time[0] = 0
infected_fraction = initial_infected_fraction


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
    
    
    
    base_individual_exposure_rate = epi.R0/epi.mean_infectious_period

    individual_exposure = public_health_adjustment * base_individual_exposure_rate

    #Beds and Mortality

    bed_occupancy_fraction = (1 - avoid_elective_operations * social_distancing_window.window(i) * max_reduction_in_normal_bed_occupancy) * normal_bed_occupancy_fraction

    hospital_p_i_threshold = ((1 - bed_occupancy_fraction) * beds_per_1000 / 1000) / epi.fraction_of_visible_requiring_hospitalization

    hospital_z_i = (1 / epi.coeff_of_variation_i) * (hospital_p_i_threshold/(infected_fraction + epi.eps) - 1)

    mean_exceedance_per_infected_fraction = epi.coeff_of_variation_i * (-hospital_z_i * (1 - st.norm.cdf(hospital_z_i)) + math.exp(-0.5 * hospital_z_i ** 2) / math.sqrt(2 * 3.14159))

    mortality_rate = (1 - epi.invisible_fraction) * epi.fraction_of_visible_requiring_hospitalization * baseline_hospitalized_mortality_rate * ((1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * epi.overflow_hospitalized_mortality_rate_factor)

    mortality_rate_at_risk = (1 - epi.invisible_fraction) * epi.fraction_of_visible_requiring_hospitalization * baseline_hospitalized_mortality_rate_at_risk * ((1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * epi.overflow_hospitalized_mortality_rate_factor)

    #Epidemiology Matrix Model

    individual_exposure_rate = public_health_adjustment * base_individual_exposure_rate
    social_exposure_rate = individual_exposure_rate * (Itot_lagged/N_lagged) * (1 - epi.coeff_of_variation_i**2 * Itot_lagged/S_lagged)

    RD = I[epi.infective_time_period]
    RD_r = I_r[epi.infective_time_period]

    for j in range(epi.infective_time_period,1,-1):
        I[j] = (1 - epi.rd_I[j-1]) * I[j-1]
        I_r[j] = (1 - epi.rd_I_r[j-1]) * I[j-1]
        RD = RD + epi.rd_I[j-1]*I[j-1]
        RD_r = RD_r + epi.rd_I_r[j-1]*I_r[j-1]
    I[1] = E[epi.exposed_time_period]
    I_r[1] = E[epi.exposed_time_period] * epi.population_at_risk_frac
    for j in range(epi.exposed_time_period, 1, -1):
        E[j] = (1 - epi.infected_E[j - 1]) * E[j - 1]
        I[1] = I[1] + epi.infected_E[j - 1] * E[j - 1]
        I_r[1] = I[1] + epi.infected_E[j-1] * E[j - 1] * epi.population_at_risk_frac
    E[1] = social_exposure_rate * susceptible_no_risk + social_exposure_rate * susceptible_at_risk
    Itot_lagged = infective
    infective = np.sum(I) + np.sum(I_r)
    S_lagged = susceptible_no_risk + susceptible_at_risk
    susceptible_at_risk -= social_exposure_rate * susceptible_at_risk
    susceptible_no_risk -= social_exposure_rate * susceptible_no_risk
    exposed = np.sum(E)
    N_lagged = N
    N_lagged_r = N_r
    new_deaths = mortality_rate*RD
    new_deaths_r = mortality_rate_at_risk*RD_r
    N -= new_deaths
    N_r -= new_deaths_r
    deaths = deaths + new_deaths + new_deaths_r
    recovered_pool = (1 - mortality_rate) * RD
    new_visible_cases = (1 - epi.invisible_fraction) * I[1] + (1 - epi.invisible_fraction) * I_r[1]
    cumulative_cases = cumulative_cases + new_visible_cases
    cumulative_cases_per_1000 = 1000 * cumulative_cases/initial_population
    susceptible_over_time[i] = susceptible_no_risk + susceptible_at_risk
    exposed_over_time[i] = exposed_no_risk + exposed_at_risk
    infective_over_time[i] = infective
    deaths_over_time[i] = deaths
    recovered_over_time[i] = recovered_pool
    
print('SUSCEPTIBLE\n')
#print(susceptible_over_time)
plt.plot(susceptible_over_time[start_time:end_time] )
plt.ylabel('SUSCEPTIBLE')
plt.show()
print('EXPOSED\n')
#print(exposed_over_time)
plt.plot(exposed_over_time[start_time:end_time])
plt.ylabel('EXPOSED')
plt.show()
print('INFECTIVE\n')
#print(infective_over_time)
plt.plot(infective_over_time[start_time:end_time])
plt.ylabel('INFECTIVE')
plt.show()
print('DEATHS\n')
#print(deaths_over_time)
plt.plot(deaths_over_time[start_time:end_time])
plt.ylabel('DEATHS')
plt.show()
print('RECOVERED\n')
#print(recovered_over_time)
plt.plot(recovered_over_time[start_time:end_time])
plt.ylabel('RECOVERED')
plt.show()