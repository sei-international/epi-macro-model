import numpy as np
from scipy import stats as st
import math
import matplotlib.pyplot as plt
import yaml


def ramp(time,slope,start_time,end_time):
    if(time > start_time):
        if(time <  end_time):
            return slope*(time - start_time)
        else:
            return slope*(end_time - start_time)
    else:
        return 0

def step(time, sheight, stime):
    if(time <  stime):
        return 0
    else:
        return sheight

def XIDZ(a,b,x):
	if b == 0:
		return x
	else:
		return a/b

def INTEG(a,b):
	return b + a

with open(r'input.yaml') as file:
    parameters = yaml.full_load(file)


start_time = parameters['time']['start']
end_time = parameters['time']['end']
time_step = parameters['time']['step']
exposed_time_period = 12
infective_time_period = 32
infected_fraction = 0
social_distancing_ramp_time = 31
isolate_cases_ramp_time = 31
initial_govt_deficit_as_share_of_Gc0= 0.480282
initial_population= 3.27e+08
initial_population_at_risk_percentage = 0.05
baseline_hospital_expenditure_as_share_of_wages= 0.1
unemployment_coverage_fraction= 1
producer_taxes_as_share_of_initial_profits= 0.114203
hospital_covid_cost_recovery_rate= 0.25
avoid_elective_operations= 1
omega_c = 0.48
omega_s = 0.7
omega_h = 0.81
sales_tax_rate_c = 0.056
sales_tax_rate_s = 0.06
prod_tax_rate_c= 0.039
prod_tax_rate_h= 0.019
prod_tax_rate_s= 0.025
Gc0 = 1.622e+06/365
import_fraction = 0.042
govt_payroll_multiplier = 1.24
wage_tax_rate = 0.277
initial_exposed = 5000
economic_impact_of_social_distancing= 0.5
isolate_cases_start = 120
isolate_cases_end=600
isolate_infectious_cases_extent= 0.9
isolate_infectious_cases= 0
isolate_visible_cases= 0
marginal_consumption_fraction_cc= 0.6
social_distancing_start = 120
social_distancing_end=600
social_distancing = 1
social_distancing_extent = 0.75
R0= 2.25
case_fatality_rate= 0.057
case_fatality_rate_at_risk = 0.07
mean_infectious_period=16
capital_productivity = 0.357
default_fraction = 0.1
investment_utilization_response = 0.025
max_reduction_in_normal_bed_occupancy= 0.33
normal_bed_occupancy_fraction = 0.64
normal_utilization = 0.85
overflow_hospitalized_mortality_rate_factor = 2
participation_rate = 0.63
tech_coeff_ac = 0.428
tech_coeff_ah = 0.335
tech_coeff_as= 0.395
coeff_of_variation_i= 0.3
working_age_fraction = 0.63
invisible_fraction = 0.87
fraction_of_visible_requiring_hospitalization = 0.38
fraction_of_visible_requiring_hospitalization_at_risk = 0.6
max_investment_rate_boost = 0.25
base_net_investment_rate = 0.02
initial_beds_per_1000 = 2.4
endogenize_bed_expansion= 0
depreciation_rate = 0.05
initial_average_wage_share= 0.639197
rd_I = np.zeros(infective_time_period)
rd_I_r = np.zeros(infective_time_period)
rd_I = parameters['rd_I']
rd_I_r = parameters['rd_I_r']
infected_E = np.zeros(exposed_time_period)
infected_E = parameters['infected_E']
I = np.zeros(infective_time_period + 1)
E = np.zeros(exposed_time_period + 1)
I_r = np.zeros(infective_time_period + 1)
E[1] = initial_exposed
GDP0 = 1.72665e+07/365
coeff_of_variation_i= 0.3


initial_wage_bill= initial_average_wage_share * GDP0

Xh0 = baseline_hospital_expenditure_as_share_of_wages * (1 - wage_tax_rate) * initial_wage_bill

baseline_hospitalized_mortality_rate = case_fatality_rate / fraction_of_visible_requiring_hospitalization
baseline_hospitalized_mortality_rate_at_risk = case_fatality_rate_at_risk / fraction_of_visible_requiring_hospitalization_at_risk

deaths = 0

cumulative_cases = 0

N = initial_population

N_r = initial_population * initial_population_at_risk_percentage

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

exposed_at_risk = initial_exposed * initial_population_at_risk_percentage

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


for i in range(start_time, end_time):

	#Epidemiology controls
    social_distancing_window = ramp(i, 1 / social_distancing_ramp_time, social_distancing_start,social_distancing_start + social_distancing_ramp_time) * (1 - step(i, 1, social_distancing_end))

    isolate_cases_window = ramp(i, 1 / isolate_cases_ramp_time, isolate_cases_start,isolate_cases_start + isolate_cases_ramp_time) * (1 - step(i, 1, isolate_cases_end))

    PHA_social_distancing = social_distancing_window * social_distancing * social_distancing_extent

    PHA_isolate_cases = isolate_cases_window * max(isolate_visible_cases * (1 - invisible_fraction), isolate_infectious_cases * isolate_infectious_cases_extent)

    public_health_adjustment = (1 - PHA_social_distancing) * (1 - PHA_isolate_cases)

    base_individual_exposure_rate = R0/mean_infectious_period

    individual_exposure = public_health_adjustment * base_individual_exposure_rate

    deaths_per_1000 = 1000 * deaths/initial_population

    #Beds and Mortality

    bed_occupancy_fraction = (1 - avoid_elective_operations * social_distancing_window * max_reduction_in_normal_bed_occupancy) * normal_bed_occupancy_fraction

    hospital_p_i_threshold = ((1 - bed_occupancy_fraction) * beds_per_1000 / 1000) / fraction_of_visible_requiring_hospitalization

    hospital_z_i = (1 / coeff_of_variation_i) * (XIDZ(hospital_p_i_threshold, infected_fraction, 1e+09) - 1)

    mean_exceedance_per_infected_fraction = coeff_of_variation_i * (-hospital_z_i * (1 - st.norm.cdf(hospital_z_i)) + math.exp(-0.5 * hospital_z_i ** 2) / math.sqrt(2 * 3.14159))

    mortality_rate = (1 - invisible_fraction) * fraction_of_visible_requiring_hospitalization * baseline_hospitalized_mortality_rate * ((1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * overflow_hospitalized_mortality_rate_factor)

    mortality_rate_at_risk = (1 - invisible_fraction) * fraction_of_visible_requiring_hospitalization * baseline_hospitalized_mortality_rate_at_risk * ((1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * overflow_hospitalized_mortality_rate_factor)

    #Epidemiology Matrix Model

    individual_exposure_rate = public_health_adjustment * base_individual_exposure_rate
    social_exposure_rate = individual_exposure_rate * (Itot_lagged/N_lagged) * (1 - coeff_of_variation_i**2 * Itot_lagged/S_lagged)

    RD = I[32]
    RD_r = I_r[32]

    for j in range(32,1,-1):
        I[j] = (1 - rd_I[j-1]) * I[j-1]
        I_r[j] = (1 - rd_I_r[j-1]) * I[j-1]
        RD = RD + rd_I[j-1]*I[j-1]
        RD_r = RD_r + rd_I_r[j-1]*I_r[j-1]
    I[1] = E[12]
    I_r[1] = E[12] * initial_population_at_risk_percentage
    for j in range(12, 1, -1):
        E[j] = (1 - infected_E[j - 1]) * E[j - 1]
        I[1] = I[1] + infected_E[j - 1] * E[j - 1]
        I_r[1] = I[1] + infected_E[j-1] * E[j - 1] * initial_population_at_risk_percentage
    E[1] = social_exposure_rate * susceptible_no_risk + social_exposure_rate * susceptible_at_risk
    Itot_lagged = infective
    infective = np.sum(I) + np.sum(I_r)
    S_lagged = susceptible_no_risk + susceptible_at_risk
    susceptible_at_risk = INTEG(-social_exposure_rate * susceptible_at_risk, susceptible_at_risk)
    susceptible_no_risk = INTEG(-social_exposure_rate * susceptible_no_risk, susceptible_no_risk)
    exposed = np.sum(E)
    N_lagged = N
    N_lagged_r = N_r
    N = INTEG(-mortality_rate*RD, N)
    N_r = INTEG(-mortality_rate_at_risk*RD_r, N_r)
    new_deaths = mortality_rate*RD
    new_deaths_r = mortality_rate_at_risk*RD_r
    deaths = deaths +  + new_deaths_r
    recovered_pool = (1 - mortality_rate) * RD
    new_visible_cases = (1 - invisible_fraction) * I[1] + (1 - invisible_fraction) * I_r[1]
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
print(exposed_over_time)
plt.plot(exposed_over_time[start_time:end_time])
plt.ylabel('EXPOSED')
plt.show()
print('INFECTIVE\n')
print(infective_over_time)
plt.plot(infective_over_time[start_time:end_time])
plt.ylabel('INFECTIVE')
plt.show()
print('DEATHS\n')
print(deaths_over_time)
plt.plot(deaths_over_time[start_time:end_time])
plt.ylabel('DEATHS')
plt.show()
print('RECOVERED\n')
print(recovered_over_time)
plt.plot(recovered_over_time[start_time:end_time])
plt.ylabel('RECOVERED')
plt.show()