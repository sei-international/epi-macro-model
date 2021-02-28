import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from seir_model import SEIR_matrix
from common import Window, get_datetime, timesteps_between_dates, get_datetime_array, timesteps_over_timedelta_weeks

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

with open(r'regions.yaml') as file:
    regions = yaml.full_load(file)

nregions = len(regions)
epi = []
intl_visitors = []
between_region_mobility_rate = []
between_locality_mobility_rate = []
beds_per_1000 = []
baseline_hosp = []
for rgn in regions:
    beds_per_1000.append(rgn['initial']['beds per 1000'])
    baseline_hosp.append(rgn['initial']['population'] * rgn['initial']['beds per 1000']/1000)
    
    epi.append(SEIR_matrix(r'seir_params.yaml', rgn))
    if 'international travel' in rgn:
        intl_visitors.append(rgn['international travel']['daily arrivals'] * rgn['international travel']['duration of stay'])
    else:
        intl_visitors.append(0.0)
    between_locality_mobility_rate.append(rgn['between locality mobility rate'])
    between_region_mobility_rate.append(rgn['between region mobility rate'])

start_datetime = get_datetime(common_params['time']['COVID start'])
start_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['COVID start'])
end_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['end date'])
epi_datetime_array = get_datetime_array(common_params['time']['COVID start'], common_params['time']['end date'])
ntimesteps = end_time - start_time + 1

# All the epidemiological regional models will give the same values for these parameters
hosp_per_infective = (1 - epi[0].invisible_fraction) * epi[0].ave_fraction_of_visible_requiring_hospitalization
epi_invisible_fraction = epi[0].invisible_fraction

normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
max_reduction_in_normal_bed_occupancy= common_params['bed occupancy']['max reduction']


avoid_elective_operations= common_params['avoid elective operations']

# Global infection rate per person
global_infection_points = common_params['global infection rate']
global_infection_npoints = len(global_infection_points)
global_infection_traj_start = global_infection_points[0][0]
if get_datetime(global_infection_traj_start) > start_datetime:
    global_infection_traj_start = common_params['time']['COVID start']
global_infection_traj_end = global_infection_points[global_infection_npoints - 1][0]
global_infection_traj_timesteps_array = np.array(range(0,timesteps_between_dates(global_infection_traj_start, common_params['time']['end date']) + 1))
global_infection_ts = np.empty(global_infection_npoints)
global_infection_val = np.empty(global_infection_npoints)
for i in range(0,global_infection_npoints):
    global_infection_ts[i] = timesteps_between_dates(global_infection_traj_start, global_infection_points[i][0])
    global_infection_val[i] = global_infection_points[i][1]/1000 # Values are entered per 1000
global_infection_rate = np.interp(global_infection_traj_timesteps_array, global_infection_ts, global_infection_val)
# Trunctate at start as necessary
ntrunc = timesteps_between_dates(global_infection_traj_start, common_params['time']['COVID start'])
global_infection_rate = global_infection_rate[ntrunc:]

# Maximum vaccination rate
vaccination_points = common_params['vaccination']['maximum doses per day']
vaccination_delay = timesteps_over_timedelta_weeks(common_params['vaccination']['time to efficacy'])
vaccination_npoints = len(vaccination_points)
vaccination_start = vaccination_points[0][0]
vaccination_end = vaccination_points[vaccination_npoints - 1][0]
vaccination_timesteps_array = np.array(range(0,timesteps_between_dates(common_params['time']['COVID start'], common_params['time']['end date']) + 1))
vaccination_ts = np.empty(vaccination_npoints)
vaccination_val = np.empty(vaccination_npoints)
for i in range(0,vaccination_npoints):
    vaccination_ts[i] = timesteps_between_dates(common_params['time']['COVID start'], vaccination_points[i][0]) + vaccination_delay
    vaccination_val[i] = vaccination_points[i][1]
vaccination_max_doses = np.interp(vaccination_timesteps_array, vaccination_ts, vaccination_val)

isolate_symptomatic_cases_windows = []
for window in common_params['isolate symptomatic cases']:
    if window['apply']:
        isolate_symptomatic_cases_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                        (get_datetime(window['end date']) - start_datetime).days,
                                                        window['ramp up for'],
                                                        window['ramp down for'],
                                                        (1 - epi_invisible_fraction) * window['fraction of cases isolated']))

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

travel_restrictions_windows = []
for window in common_params['travel restrictions']:
    if window['apply']:
        travel_restrictions_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                       (get_datetime(window['end date']) - start_datetime).days,
                                       window['ramp up for'],
                                       window['ramp down for'],
                                       window['effectiveness']))


# Initialize values for indicator graphs
deaths = np.zeros(nregions)
cumulative_cases = np.zeros(nregions)

deaths_over_time = np.zeros((nregions, ntimesteps))
new_deaths_over_time = np.zeros((nregions, ntimesteps))
recovered_over_time = np.zeros((nregions, ntimesteps))
mortality_rate_over_time = np.zeros((nregions, ntimesteps))
hospitalization_index = np.ones((nregions, ntimesteps))

susceptible_over_time = np.zeros((nregions, ntimesteps))
susceptible_over_time[:,0] = [e.S for e in epi]

exposed_over_time = np.zeros((nregions, ntimesteps))
exposed_over_time[:,0] = [np.sum(e.E) for e in epi]

infective_over_time = np.zeros((nregions, ntimesteps))
infective_over_time[:,0] = [np.sum(e.I_nr + e.I_r) for e in epi]

comm_spread_frac_over_time = np.zeros((nregions, ntimesteps))
comm_spread_frac_over_time[:,0] = [e.comm_spread_frac for e in epi]


for i in range(0, ntimesteps):
    # Public health measures
    PHA_social_distancing = 0
    for w in soc_dist_windows:
        PHA_social_distancing += w.window(i)
    PHA_travel_restrictions = 0
    for w in travel_restrictions_windows:
        PHA_travel_restrictions += w.window(i)
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
    

    # Loop over regions
    for j in range(0, nregions):
        intl_infected_visitors = intl_visitors[j] * global_infection_rate[i] * min(0, 1 - PHA_travel_restrictions)
        dom_infected_visitors = 0
        if nregions > 1:
            for k in range(0, nregions):
                if k != j:
                    dom_infected_visitors += epi[k].Itot_prev * between_region_mobility_rate[k]/(nregions - 1)

        # Run the model for one time step
        epi[j].update(dom_infected_visitors + intl_infected_visitors,
                      between_locality_mobility_rate[j],
                      public_health_adjustment,
                      bed_occupancy_fraction,
                      beds_per_1000[j],
                      vaccination_max_doses[i])

        # Update values for indicator graphs
        new_deaths_over_time[j,i] = epi[j].new_deaths
        deaths[j] += epi[j].new_deaths
        susceptible_over_time[j,i] = epi[j].S
        exposed_over_time[j,i] = np.sum(epi[j].E)
        infective_over_time[j,i] = epi[j].Itot
        deaths_over_time[j,i] = deaths[j]
        recovered_over_time[j,i] = epi[j].R
        cumulative_cases[j] += (1 - epi[j].invisible_fraction) * (epi[j].I_nr[1] + epi[j].I_r[1])
        comm_spread_frac_over_time[j,i] = epi[j].comm_spread_frac
        mortality_rate_over_time[j,i] = epi[j].curr_mortality_rate
        hospitalization_index[j,i] = bed_occupancy_factor + hosp_per_infective * epi[j].Itot/baseline_hosp[j]

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

for j in range(0,nregions):
    info = regions[j]
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.stackplot(epi_datetime_array,
                  susceptible_over_time[j,0:end_time-start_time],
                  exposed_over_time[j,0:end_time-start_time],
                  infective_over_time[j,0:end_time-start_time],
                  recovered_over_time[j,0:end_time-start_time],
                  deaths_over_time[j,0:end_time-start_time],
                  labels=['susceptible','exposed','infected','recovered','died'])
    plt.legend(loc='lower right')
    plt.title(info['name'])
    plt.show()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, deaths_over_time[j,0:end_time-start_time])
    plt.ylabel('cumulative deaths')
    plt.title(info['name'])
    plt.show()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, new_deaths_over_time[j,0:end_time-start_time])
    plt.ylabel('new deaths/day')
    plt.title(info['name'])
    plt.show()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, comm_spread_frac_over_time[j,0:end_time-start_time])
    plt.ylabel('community spread')
    plt.title(info['name'])
    plt.show()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, mortality_rate_over_time[j,0:end_time-start_time])
    plt.ylabel('mortality rate')
    plt.title(info['name'])
    plt.show()
