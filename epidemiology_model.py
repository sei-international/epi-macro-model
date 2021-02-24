import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from seir_model import SEIR_matrix
from common import Window, get_datetime, timesteps_between_dates, get_datetime_array

with open(r'common_params.yaml') as file:
    common_params = yaml.full_load(file)

with open(r'geographies.yaml') as geo_file:
    geographies = yaml.full_load(geo_file)

# Load epidemiological model object
for j in range(0, len(geographies['geography'])):
    info = geographies['geography'][j]
    epi = SEIR_matrix(r'seir_params.yaml', common_params['initial'], info)
    start_datetime = get_datetime(common_params['time']['start date'])
    start_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['COVID start'])
    end_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['end date'])
    epi_datetime_array = get_datetime_array(common_params['time']['COVID start'], common_params['time']['end date'])

    beds_per_1000 = common_params['beds per 1000']
    baseline_hosp = common_params['initial']['total population'] * beds_per_1000/1000
    normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
    max_reduction_in_normal_bed_occupancy= common_params['bed occupancy']['max reduction']
    hosp_per_infective = (1 - epi.invisible_fraction) * epi.ave_fraction_of_visible_requiring_hospitalization


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

    infected_arrivals = (info['international travel']['daily arrivals'] * info['international travel'][
        'fraction infected'] * info['international travel']['duration of stay']) + (
                                    info['population'] * info['external mobility rate'])

    internal_movement_per_person = info['internal mobility rate']

    # Initialize values for indicator graphs urban

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
    mortality_rate_over_time = np.zeros(end_time - start_time + 1)
    hospitalization_index = np.ones(end_time - start_time + 1)
    hospitalization_index[0] = 1

    for i in range(0, end_time - start_time + 1):
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
        epi.update(infected_arrivals, internal_movement_per_person, public_health_adjustment, bed_occupancy_fraction, beds_per_1000)

        # Update values for indicator graphs
        new_deaths_over_time[i] = epi.new_deaths
        deaths += epi.new_deaths
        cumulative_cases_per_1000 = 1000 * cumulative_cases / initial_population
        susceptible_over_time[i] = epi.S
        exposed_over_time[i] = np.sum(epi.E)
        infective_over_time[i] = epi.Itot
        deaths_over_time[i] = deaths
        recovered_over_time[i] = epi.R
        new_visible_cases = (1 - epi.invisible_fraction) * (epi.I_nr[1] + epi.I_r[1])
        cumulative_cases += new_visible_cases
        comm_spread_frac_over_time[i] = epi.comm_spread_frac
        mortality_rate_over_time[i] = epi.curr_mortality_rate

    
        excess_hosp = hosp_per_infective * epi.Itot
        hospitalization_index[i - start_time] = bed_occupancy_factor + excess_hosp/baseline_hosp

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.stackplot(epi_datetime_array,
                  susceptible_over_time[0:end_time-start_time],
                  exposed_over_time[0:end_time-start_time],
                  infective_over_time[0:end_time-start_time],
                  recovered_over_time[0:end_time-start_time],
                  deaths_over_time[0:end_time-start_time],
                  labels=['susceptible','exposed','infected','recovered','died'])
    plt.legend(loc='lower right')
    plt.show()

    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, deaths_over_time[0:end_time-start_time])
    plt.ylabel('cumulative deaths '+ info['name'])
    plt.show()

    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, new_deaths_over_time[0:end_time-start_time])
    plt.ylabel('new deaths/day ' + info['name'])
    plt.show()

    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, comm_spread_frac_over_time[0:end_time-start_time])
    plt.ylabel('community spread ' + info['name'])
    plt.show()

    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.plot(epi_datetime_array, mortality_rate_over_time[0:end_time-start_time])
    plt.ylabel('mortality rate ' + info['name'])
    plt.show()


