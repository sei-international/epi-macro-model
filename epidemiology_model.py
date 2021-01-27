import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import yaml
from seir_model import SEIR_matrix

class Window:
    def __init__(self, start, end, ramp_up, ramp_down, effectiveness = 1.0):
        self.start = start
        self.end = end
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.effectiveness = effectiveness
    
    # Ramp to a maximum value of 1.0 and then back down
    # Ramp time can be zero
    def window(self, time):
        if time < self.start or time > self.end:
            w = 0
        elif time >= self.start + self.ramp_up and time <= self.end - self.ramp_down:
            w = 1
        elif time < self.start + self.ramp_up:
            w = (time - self.start)/self.ramp_up
        else:
            w = (self.end - time)/self.ramp_down
        return self.effectiveness * w

# d is a dict with keys 'year', 'month', 'day'
def get_datetime(d):
    return dt.date(d['year'],d['month'],d['day'])

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

# Load epidemiological model object
epi_urban = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'][0]['number of localities'],'urban')
epi_rural = SEIR_matrix(r'seir_params.yaml', common_params['initial'], common_params['geography'][1]['number of localities'], 'rural')
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

    # Update values for indicator graphs
    new_deaths_over_time_urban[i] = epi_urban.new_deaths
    deaths_urban += epi_urban.new_deaths
    cumulative_cases_per_1000_urban = 1000 * cumulative_cases_urban/initial_population_urban
    susceptible_over_time_urban[i] = epi_urban.S
    exposed_over_time_urban[i] = np.sum(epi_urban.E)
    infective_over_time_urban[i] = epi_urban.Itot
    deaths_over_time_urban[i] = deaths_urban
    recovered_over_time_urban[i] = epi_urban.R
    new_visible_cases_urban = (1 - epi_urban.invisible_fraction) * (epi_urban.I_nr[1] + epi_urban.I_r[1])
    cumulative_cases_urban += new_visible_cases_urban
    comm_spread_frac_over_time_urban[i] = epi_urban.comm_spread_frac
    mortality_rate_over_time_urban[i] = epi_urban.curr_mortality_rate

    new_deaths_over_time_rural[i] = epi_rural.new_deaths
    deaths_rural += epi_rural.new_deaths
    cumulative_cases_per_1000_rural = 1000 * cumulative_cases_rural / initial_population_rural
    susceptible_over_time_rural[i] = epi_rural.S
    exposed_over_time_rural[i] = np.sum(epi_rural.E)
    infective_over_time_rural[i] = epi_rural.Itot
    deaths_over_time_rural[i] = deaths_rural
    recovered_over_time_rural[i] = epi_rural.R
    new_visible_cases_rural = (1 - epi_rural.invisible_fraction) * (epi_rural.I_nr[1] + epi_rural.I_r[1])
    cumulative_cases_rural += new_visible_cases_rural
    comm_spread_frac_over_time_rural[i] = epi_rural.comm_spread_frac
    mortality_rate_over_time_rural[i] = epi_rural.curr_mortality_rate

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.stackplot(datetime_array,
              susceptible_over_time_urban[start_time:end_time],
              exposed_over_time_urban[start_time:end_time],
              infective_over_time_urban[start_time:end_time],
              recovered_over_time_urban[start_time:end_time],
              deaths_over_time_urban[start_time:end_time],
              labels=['susceptible','exposed','infected','recovered','died'])
plt.legend(loc='lower right')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, deaths_over_time_urban[start_time:end_time])
plt.ylabel('cumulative deaths urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, new_deaths_over_time_urban[start_time:end_time])
plt.ylabel('new deaths/day urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, comm_spread_frac_over_time_urban[start_time:end_time])
plt.ylabel('community spread urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, mortality_rate_over_time_urban[start_time:end_time])
plt.ylabel('mortality rate urban')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.stackplot(datetime_array,
              susceptible_over_time_rural[start_time:end_time],
              exposed_over_time_rural[start_time:end_time],
              infective_over_time_rural[start_time:end_time],
              recovered_over_time_rural[start_time:end_time],
              deaths_over_time_rural[start_time:end_time],
              labels=['susceptible', 'exposed', 'infected', 'recovered', 'died'])
plt.legend(loc='lower right')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, deaths_over_time_rural[start_time:end_time])
plt.ylabel('cumulative deaths rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, new_deaths_over_time_rural[start_time:end_time])
plt.ylabel('new deaths/day rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, comm_spread_frac_over_time_rural[start_time:end_time])
plt.ylabel('community spread rural')
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(datetime_array, mortality_rate_over_time_rural[start_time:end_time])
plt.ylabel('mortality rate rural')
plt.show()
