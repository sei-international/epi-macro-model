import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from seir_model import SEIR_matrix
from common import Window, get_datetime, timesteps_between_dates, get_datetime_array

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

# Load epidemiological model object
# TODO: DON'T DO THIS! Should be more robust -- loop over geographies, or have different input files!
epi_urban = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'][0]['number of localities'],'urban')
epi_rural = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'][1]['number of localities'], 'rural')
start_datetime = get_datetime(common_params['time']['start date'])
start_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['COVID start'])
end_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['end date'])
epi_datetime_array = get_datetime_array(common_params['time']['COVID start'], common_params['time']['end date'])

beds_per_1000 = common_params['beds per 1000']
baseline_hosp = common_params['initial']['total population'] * beds_per_1000/1000
normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
max_reduction_in_normal_bed_occupancy= common_params['bed occupancy']['max reduction']
hosp_per_infective_urban = (1 - epi_urban.invisible_fraction) * epi_urban.ave_fraction_of_visible_requiring_hospitalization
hosp_per_infective_rural = (1 - epi_rural.invisible_fraction) * epi_rural.ave_fraction_of_visible_requiring_hospitalization


avoid_elective_operations= common_params['avoid elective operations']

isolate_symptomatic_cases_windows = []
for window in common_params['isolate symptomatic cases']:
    if window['apply']:
        isolate_symptomatic_cases_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                        (get_datetime(window['end date']) - start_datetime).days,
                                                        window['ramp up for'],
                                                        window['ramp down for'],
                                                        (1 - epi_urban.invisible_fraction) * window['fraction of cases isolated']))

test_and_trace_windows = []
for window in common_params['test and trace']:
    if window['apply']:
        test_and_trace_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                             (get_datetime(window['end date']) - start_datetime).days,
                                             window['ramp up for'],
                                             window['ramp down for'],
                                             window['fraction of infectious cases isolated']))

soc_dist_windows = []
for window in common_params['social distance']:
    if window['apply']:
        soc_dist_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                       (get_datetime(window['end date']) - start_datetime).days,
                                       window['ramp up for'],
                                       window['ramp down for'],
                                       window['effectiveness']))

infected_arrivals_urban = (common_params['international travel']['daily arrivals'] * \
                    common_params['international travel']['fraction infected'] * \
                    common_params['international travel']['duration of stay']) + \
                          (common_params['geography'][1]['population'] * \
                    common_params['geography'][1]['external mobility rate'])
infected_arrivals_rural = (common_params['geography'][0]['population'] * \
                    common_params['geography'][0]['external mobility rate'])
internal_movement_per_person_urban = common_params['geography'][0]['internal mobility rate']
internal_movement_per_person_rural = common_params['geography'][1]['internal mobility rate']



# Initialize values for indicator graphs urban
deaths_urban = 0
cumulative_cases_urban = 0
susceptible_over_time_urban = np.zeros(end_time - start_time + 1)
susceptible_over_time_urban[0] = epi_urban.S
exposed_over_time_urban = np.zeros(end_time - start_time + 1)
exposed_over_time_urban[0] = np.sum(epi_urban.E)
infective_over_time_urban = np.zeros(end_time - start_time + 1)
infective_over_time_urban[0] = np.sum(epi_urban.I_nr + epi_urban.I_r)
deaths_over_time_urban = np.zeros(end_time - start_time + 1)
deaths_over_time_urban[0] = 0
new_deaths_over_time_urban = np.zeros(end_time - start_time + 1)
new_deaths_over_time_urban[0] = 0
recovered_over_time_urban = np.zeros(end_time - start_time + 1)
recovered_over_time_urban[0] = 0
initial_population_urban = epi_urban.N
comm_spread_frac_over_time_urban = np.zeros(end_time - start_time + 1)
comm_spread_frac_over_time_urban[0] = epi_urban.comm_spread_frac
mortality_rate_over_time_urban = np.zeros(end_time - start_time + 1)

# Initialize values for indicator graphs rural
deaths_rural = 0
cumulative_cases_rural = 0
susceptible_over_time_rural = np.zeros(end_time - start_time + 1)
susceptible_over_time_rural[0] = epi_urban.S
exposed_over_time_rural = np.zeros(end_time - start_time + 1)
exposed_over_time_rural[0] = np.sum(epi_urban.E)
infective_over_time_rural = np.zeros(end_time - start_time + 1)
infective_over_time_rural[0] = np.sum(epi_urban.I_nr + epi_urban.I_r)
deaths_over_time_rural = np.zeros(end_time - start_time + 1)
deaths_over_time_rural[0] = 0
new_deaths_over_time_rural = np.zeros(end_time - start_time + 1)
new_deaths_over_time_rural[0] = 0
recovered_over_time_rural = np.zeros(end_time - start_time + 1)
recovered_over_time_rural[0] = 0
initial_population_rural = epi_urban.N
comm_spread_frac_over_time_rural = np.zeros(end_time - start_time + 1)
comm_spread_frac_over_time_rural[0] = epi_urban.comm_spread_frac
mortality_rate_over_time_rural = np.zeros(end_time - start_time + 1)
hospitalization_index = np.ones(end_time - start_time + 1)
hospitalization_index[0] = 1

for i in range(start_time, end_time):

	# Public health measures
    PHA_social_distancing = 0
    for w in soc_dist_windows:
        PHA_social_distancing += w.window(i)
    PHA_isolate_visible_cases = 0
    for w in isolate_symptomatic_cases_windows:
        PHA_isolate_visible_cases += w.window(i)
    PHA_isolate_infectious_cases = 0
    for w in test_and_trace_windows:
        PHA_isolate_infectious_cases += w.window(i)
    PHA_isolate_cases = max(PHA_isolate_visible_cases, PHA_isolate_infectious_cases)
    public_health_adjustment = (1 - PHA_social_distancing) * (1 - PHA_isolate_cases)

    # Beds and Mortality
    if avoid_elective_operations:
        bed_occupancy_factor = (1 - PHA_social_distancing * max_reduction_in_normal_bed_occupancy)
    else:
        bed_occupancy_factor = 1
    bed_occupancy_fraction = bed_occupancy_factor * normal_bed_occupancy_fraction

    # Run the model for one time step
    epi_urban.update(infected_arrivals_urban, internal_movement_per_person_urban, public_health_adjustment, bed_occupancy_fraction, beds_per_1000)
    epi_rural.update(infected_arrivals_rural, internal_movement_per_person_rural, public_health_adjustment, bed_occupancy_fraction, beds_per_1000)

    # TODO: DON'T DO THIS! Loop over geographies and allow user to name them
    # Update values for indicator graphs
    new_deaths_over_time_urban[i - start_time] = epi_urban.new_deaths
    deaths_urban += epi_urban.new_deaths
    cumulative_cases_per_1000_urban = 1000 * cumulative_cases_urban/initial_population_urban
    susceptible_over_time_urban[i - start_time] = epi_urban.S
    exposed_over_time_urban[i - start_time] = np.sum(epi_urban.E)
    infective_over_time_urban[i - start_time] = epi_urban.Itot
    deaths_over_time_urban[i - start_time] = deaths_urban
    recovered_over_time_urban[i - start_time] = epi_urban.R
    new_visible_cases_urban = (1 - epi_urban.invisible_fraction) * (epi_urban.I_nr[1] + epi_urban.I_r[1])
    cumulative_cases_urban += new_visible_cases_urban
    comm_spread_frac_over_time_urban[i - start_time] = epi_urban.comm_spread_frac
    mortality_rate_over_time_urban[i - start_time] = epi_urban.curr_mortality_rate

    new_deaths_over_time_rural[i - start_time] = epi_rural.new_deaths
    deaths_rural += epi_rural.new_deaths
    cumulative_cases_per_1000_rural = 1000 * cumulative_cases_rural / initial_population_rural
    susceptible_over_time_rural[i - start_time] = epi_rural.S
    exposed_over_time_rural[i - start_time] = np.sum(epi_rural.E)
    infective_over_time_rural[i - start_time] = epi_rural.Itot
    deaths_over_time_rural[i - start_time] = deaths_rural
    recovered_over_time_rural[i - start_time] = epi_rural.R
    new_visible_cases_rural = (1 - epi_rural.invisible_fraction) * (epi_rural.I_nr[1] + epi_rural.I_r[1])
    cumulative_cases_rural += new_visible_cases_rural
    comm_spread_frac_over_time_rural[i - start_time] = epi_rural.comm_spread_frac
    mortality_rate_over_time_rural[i - start_time] = epi_rural.curr_mortality_rate
    
    excess_hosp = hosp_per_infective_urban * epi_urban.Itot + hosp_per_infective_rural * epi_rural.Itot
    hospitalization_index[i - start_time] = bed_occupancy_factor + excess_hosp/baseline_hosp

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.stackplot(epi_datetime_array,
              susceptible_over_time_urban[0:end_time-start_time],
              exposed_over_time_urban[0:end_time-start_time],
              infective_over_time_urban[0:end_time-start_time],
              recovered_over_time_urban[0:end_time-start_time],
              deaths_over_time_urban[0:end_time-start_time],
              labels=['susceptible','exposed','infected','recovered','died'])
plt.legend(loc='lower right')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, deaths_over_time_urban[0:end_time-start_time])
plt.ylabel('cumulative deaths urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, new_deaths_over_time_urban[0:end_time-start_time])
plt.ylabel('new deaths/day urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, comm_spread_frac_over_time_urban[0:end_time-start_time])
plt.ylabel('community spread urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, mortality_rate_over_time_urban[0:end_time-start_time])
plt.ylabel('mortality rate urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.stackplot(epi_datetime_array,
              susceptible_over_time_rural[0:end_time-start_time],
              exposed_over_time_rural[0:end_time-start_time],
              infective_over_time_rural[0:end_time-start_time],
              recovered_over_time_rural[0:end_time-start_time],
              deaths_over_time_rural[0:end_time-start_time],
              labels=['susceptible', 'exposed', 'infected', 'recovered', 'died'])
plt.legend(loc='lower right')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, deaths_over_time_rural[0:end_time-start_time])
plt.ylabel('cumulative deaths rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, new_deaths_over_time_rural[0:end_time-start_time])
plt.ylabel('new deaths/day rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, comm_spread_frac_over_time_rural[0:end_time-start_time])
plt.ylabel('community spread rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(epi_datetime_array, mortality_rate_over_time_rural[0:end_time-start_time])
plt.ylabel('mortality rate rural')
plt.show()
