import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import yaml
from epidemiology_functions import Window, SEIR_matrix

# d is a dict with keys 'year', 'month', 'day'
def get_datetime(d):
    return dt.date(d['year'],d['month'],d['day'])

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

# Load epidemiological model object
epi = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'])

start_time = 0
start_datetime = get_datetime(common_params['time']['start date'])
end_time = (get_datetime(common_params['time']['end date']) - start_datetime).days
datetime_array = [start_datetime + dt.timedelta(days=x) for x in range(start_time, end_time)]

beds_per_1000 = common_params['beds per 1000']
normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
max_reduction_in_normal_bed_occupancy= common_params['bed occupancy']['max reduction']

avoid_elective_operations= common_params['avoid elective operations']

isolate_symptomatic_cases_windows = []
for window in common_params['isolate symptomatic cases']:
    if window['apply']:
        isolate_symptomatic_cases_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                        (get_datetime(window['end date']) - start_datetime).days,
                                                        window['ramp up for'],
                                                        window['ramp down for'],
                                                        (1 - epi.invisible_fraction) * window['fraction of cases isolated']))

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

infected_arrivals = common_params['international travel']['daily arrivals'] * \
                    common_params['international travel']['fraction infected'] * \
                    common_params['international travel']['duration of stay']

# Initialize values for indicator graphs
deaths = 0
cumulative_cases = 0
susceptible_over_time = np.zeros(end_time - start_time + 1)
susceptible_over_time[0] = epi.S
exposed_over_time = np.zeros(end_time - start_time + 1)
exposed_over_time[0] = np.sum(epi.E)
infective_over_time = np.zeros(end_time - start_time + 1)
infective_over_time[0] = np.sum(epi.I_nr + epi.I_r)
deaths_over_time = np.zeros(end_time - start_time + 1)
deaths_over_time[0] = 0
new_deaths_over_time = np.zeros(end_time - start_time + 1)
new_deaths_over_time[0] = 0
recovered_over_time = np.zeros(end_time - start_time + 1)
recovered_over_time[0] = 0
initial_population = epi.N
comm_spread_frac_over_time = np.zeros(end_time - start_time + 1)
comm_spread_frac_over_time[0] = epi.comm_spread_frac

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
    epi.update(infected_arrivals, public_health_adjustment, bed_occupancy_fraction, beds_per_1000)
    
    # Update values for indicator graphs
    new_deaths_over_time[i] = epi.new_deaths
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

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.stackplot(datetime_array,
              susceptible_over_time[start_time:end_time],
              exposed_over_time[start_time:end_time],
              infective_over_time[start_time:end_time],
              recovered_over_time[start_time:end_time],
              deaths_over_time[start_time:end_time],
              labels=['susceptible','exposed','infected','recovered','died'])
plt.legend(loc='lower right')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, deaths_over_time[start_time:end_time])
plt.ylabel('cumulative deaths')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, new_deaths_over_time[start_time:end_time])
plt.ylabel('new deaths/day')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, comm_spread_frac_over_time[start_time:end_time])
plt.ylabel('community spread')
plt.show()
