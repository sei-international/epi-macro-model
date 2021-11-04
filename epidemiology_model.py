from numpy import array as np_array, zeros as np_zeros, sum as np_sum, empty as np_empty, \
    amax as np_amax, interp as np_interp, ones as np_ones, tile as np_tile
import yaml
from seir_model import SEIR_matrix
from common import Window, get_datetime, timesteps_between_dates, get_datetime_array, timesteps_over_timedelta_weeks
from sys import exit

def epidemiology_model():

    with open(r'common_params.yaml') as file:
        common_params = yaml.full_load(file)

    with open(r'regions.yaml') as file:
        regions = yaml.full_load(file)

    with open(r'seir_params.yaml') as file:
        seir_params_multivar = yaml.full_load(file)


    nvars=len(seir_params_multivar) # (var=1 is baseline model, var=2 is delta variant)
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
        epivar=[]
        for var in seir_params_multivar:
            epivar.append(SEIR_matrix(rgn, var, common_params))
        if 'international travel' in rgn:
            intl_visitors.append(rgn['international travel']['daily arrivals'] * rgn['international travel']['duration of stay'])
        else:
            intl_visitors.append(0.0)
        between_locality_mobility_rate.append(rgn['between locality mobility rate'])
        between_region_mobility_rate.append(rgn['between region mobility rate'])
        epi.append(epivar) # contains objects with following order: [[rgn1/var1, rgn2/var1], [rgn1/var2, rgn2/var2]]


    proportion_total = [e.proportion_global_infected for e in epi[0]]
    test1=np_sum(proportion_total,axis=0)
    if any(test1<0.999) or any(test1>1.001):
        print('Error test1: aborted')
        print('proportions of global infections across variants do not sum to 1')
        exit()

    start_datetime = get_datetime(common_params['time']['COVID start'])
    start_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['COVID start'])
    end_time = timesteps_between_dates(common_params['time']['start date'], common_params['time']['end date'])
    epi_datetime_array = get_datetime_array(common_params['time']['COVID start'], common_params['time']['end date'])
    ntimesteps = end_time - start_time

    # All the epidemiological regional models will give the same values for these parameters
    epi_invisible_fraction = epi[0][0].invisible_fraction_1stinfection

    normal_bed_occupancy_fraction = common_params['bed occupancy']['normal']
    max_reduction_in_normal_bed_occupancy = common_params['bed occupancy']['max reduction']

    if 'vaccinate at risk first' in common_params['vaccination']:
        vaccinate_at_risk = common_params['vaccination']['vaccinate at risk first']
    else:
        vaccinate_at_risk = False
    avoid_elective_operations= common_params['avoid elective operations']

    # Global infection rate per person
    global_infection_points = common_params['global infection rate']
    global_infection_npoints = len(global_infection_points)
    global_infection_traj_start = global_infection_points[0][0]
    if get_datetime(global_infection_traj_start) > start_datetime:
        global_infection_traj_start = common_params['time']['COVID start']
    global_infection_traj_timesteps_array = np_array(range(0,timesteps_between_dates(global_infection_traj_start, common_params['time']['end date']) + 1))
    global_infection_ts = np_empty(global_infection_npoints)
    global_infection_val = np_empty(global_infection_npoints)
    for i in range(0,global_infection_npoints):
        global_infection_ts[i] = timesteps_between_dates(global_infection_traj_start, global_infection_points[i][0])
        global_infection_val[i] = global_infection_points[i][1]/1000 # Values are entered per 1000
    global_infection_rate = np_interp(global_infection_traj_timesteps_array, global_infection_ts, global_infection_val)
    # Trunctate at start as necessary
    ntrunc = timesteps_between_dates(global_infection_traj_start, common_params['time']['COVID start'])
    global_infection_rate = global_infection_rate[ntrunc:]

    # Maximum vaccination rate
    vaccination_points = common_params['vaccination']['maximum doses per day']
    vaccination_delay = timesteps_over_timedelta_weeks(common_params['vaccination']['time to efficacy'])
    vaccination_npoints = len(vaccination_points)
    vaccination_timesteps_array = np_array(range(0,timesteps_between_dates(common_params['time']['COVID start'], common_params['time']['end date']) + 1))
    vaccination_ts = np_empty(vaccination_npoints)
    vaccination_val = np_empty(vaccination_npoints)
    for i in range(0,vaccination_npoints):
        vaccination_ts[i] = timesteps_between_dates(common_params['time']['COVID start'], vaccination_points[i][0]) + vaccination_delay
        vaccination_val[i] = vaccination_points[i][1]
    vaccination_max_doses = np_interp(vaccination_timesteps_array, vaccination_ts, vaccination_val)

    isolate_symptomatic_cases_windows = []
    if 'isolate symptomatic cases' in common_params:
        for window in common_params['isolate symptomatic cases']:
            if window['apply']:
                isolate_symptomatic_cases_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                                (get_datetime(window['end date']) - start_datetime).days,
                                                                window['ramp up for'],
                                                                window['ramp down for'],
                                                                (1 - epi_invisible_fraction) * window['fraction of cases isolated']))

    isolate_at_risk_windows = []
    if 'isolate at risk' in common_params:
        for window in common_params['isolate at risk']:
            if window['apply']:
                isolate_at_risk_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                      (get_datetime(window['end date']) - start_datetime).days,
                                                      window['ramp up for'],
                                                      window['ramp down for'],
                                                      window['fraction of population isolated']))

    test_and_trace_windows = []
    if 'test and trace' in common_params:
        for window in common_params['test and trace']:
            if window['apply']:
                test_and_trace_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                                     (get_datetime(window['end date']) - start_datetime).days,
                                                     window['ramp up for'],
                                                     window['ramp down for'],
                                                     window['fraction of infectious cases isolated']))

    soc_dist_windows = []
    if 'social distance' in common_params:
        for window in common_params['social distance']:
            if window['apply']:
                soc_dist_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                               (get_datetime(window['end date']) - start_datetime).days,
                                               window['ramp up for'],
                                               window['ramp down for'],
                                               window['effectiveness']))

    travel_restrictions_windows = []
    if 'international travel restrictions' in common_params:
        for window in common_params['international travel restrictions']:
            if window['apply']:
                travel_restrictions_windows.append(Window((get_datetime(window['start date']) - start_datetime).days,
                                               (get_datetime(window['end date']) - start_datetime).days,
                                               window['ramp up for'],
                                               window['ramp down for'],
                                               window['effectiveness']))

    # Initialize values for indicator graphs
    Itot_allvars=np_zeros(nregions)
    comm_spread_frac_allvars = np_zeros((nregions, nvars))
    deaths = np_zeros((nregions, nvars))
    deaths_reinf = np_zeros((nregions, nvars))
    cumulative_cases = np_zeros((nregions, nvars))

    deaths_over_time = np_zeros((nregions, ntimesteps, nvars))
    new_deaths_over_time = np_zeros((nregions, ntimesteps, nvars))
    deaths_reinf_over_time = np_zeros((nregions, ntimesteps, nvars))
    recovered_over_time = np_zeros((nregions, ntimesteps, nvars))
    immune_over_time = np_zeros((nregions, ntimesteps, nvars))
    mortality_rate_over_time = np_zeros((nregions, ntimesteps, nvars))

    hospitalization_index_region = np_ones(nregions)
    hospitalization_index = np_ones(ntimesteps)

    infective_over_time = np_zeros((nregions, ntimesteps, nvars))
    reinfective_over_time = np_zeros((nregions, ntimesteps, nvars))

    susceptible_over_time = np_zeros((nregions, ntimesteps, nvars))
    for j in range(0,nregions):
        susceptible_over_time[j,0,:] = [e.S for e in epi[j]]
    # susceptible_over_time = np_zeros((nregions, ntimesteps, nvars))
    # for j in range(0,nregions):
    #     e=epi[j]
    #     for v in range(0, len(e)):
    #         susceptible_over_time[j,0,v] = e[v].S

    exposed_over_time = np_zeros((nregions, ntimesteps, nvars))
    for j in range(0,nregions):
        exposed_over_time[j,0,:] = [np_sum(e.E_nr) + np_sum(e.E_r) for e in epi[j]]

    reexposed_over_time = np_zeros((nregions, ntimesteps, nvars))
    for j in range(0,nregions):
        reexposed_over_time[j,0,:] = [np_sum(e.RE_nr) + np_sum(e.RE_r) for e in epi[j]]    

    comm_spread_frac_over_time = np_zeros((nregions, ntimesteps, nvars))
    for j in range(0,nregions):
        comm_spread_frac_over_time[j,0,:] = [e.comm_spread_frac for e in epi[j]]

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

        PHA_isolate_at_risk = 0
        for w in isolate_at_risk_windows:
            PHA_isolate_at_risk += w.window(i)

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

        #Community spread
        for j in range(0, nregions):
            comm_spread_frac_allvars[j,:] = [e.comm_spread_frac for e in epi[j]]

        # Loop of variants
        for v in range(0,nvars):

            # Loop over regions
            for j in range(0, nregions):
                intl_infected_visitors = intl_visitors[j] * (epi[j][v].proportion_global_infected[i]*global_infection_rate[i]) * min(0, 1 - PHA_travel_restrictions)
                dom_infected_visitors = 0
                # Confirm current variant has been introduced already
                if epi_datetime_array[i] >= epi[j][v].start_time:
                    if nregions > 1:
                        for k in range(0, nregions):
                            if k != j:
                                dom_infected_visitors += epi[k][v].Itot_prev * between_region_mobility_rate[k]/(nregions - 1)

                    # Run the model for one time step
                    epi[j][v].update(dom_infected_visitors + intl_infected_visitors,
                                  between_locality_mobility_rate[j],
                                  public_health_adjustment,
                                  PHA_isolate_at_risk,
                                  bed_occupancy_fraction,
                                  beds_per_1000[j],
                                  vaccination_max_doses[i],
                                  vaccinate_at_risk,
                                  Itot_allvars[j],
                                  comm_spread_frac_allvars[j],
                                  nvars)

                    # Update values for indicator graphs
                    new_deaths_over_time[j,i,v] = epi[j][v].new_deaths
                    deaths[j,v] += epi[j][v].new_deaths
                    deaths_reinf[j,v] += epi[j][v].new_deaths_reinf
                    #susceptible_over_time[j,i,v] = epi[j][v].S
                    exposed_over_time[j,i,v] = np_sum(epi[j][v].E_nr) + np_sum(epi[j][v].E_r)
                    reexposed_over_time[j,i,v] = np_sum(epi[j][v].RE_nr) + np_sum(epi[j][v].RE_r)
                    infective_over_time[j,i,v] = epi[j][v].Itot
                    reinfective_over_time[j,i,v] = epi[j][v].RItot
                    deaths_over_time[j,i,v] = deaths[j,v]
                    deaths_reinf_over_time[j,i,v] = deaths_reinf[j,v]
                    recovered_over_time[j,i,v] = np_sum(epi[j][v].R_nr) + np_sum(epi[j][v].R_r)
                    immune_over_time[j,i,v] = epi[j][v].Im
                    cumulative_cases[j,v] += (1 - epi[j][v].invisible_fraction_1stinfection) * (epi[j][v].I_nr[1] + epi[j][v].I_r[1]) + \
                        (1 - epi[j][v].invisible_fraction_reinfection) * (epi[j][v].RI_nr[1] + epi[j][v].RI_r[1])
                    comm_spread_frac_over_time[j,i,v] = epi[j][v].comm_spread_frac
                    mortality_rate_over_time[j,i,v] = epi[j][v].curr_mortality_rate

        # Calculate hospitalisation index across variants
        hospitalized=np_zeros(nregions)
        for j in range(0, nregions):
            # Infected by regions
            for e in epi[j]:
                hosp_per_infective_1stinfections = (1 - e.invisible_fraction_1stinfection) * e.ave_fraction_of_visible_1stinfections_requiring_hospitalization
                hosp_per_infective_reinfections  = (1 - e.invisible_fraction_reinfection) * e.ave_fraction_of_visible_reinfections_requiring_hospitalization
                hospitalized[j] += ( hosp_per_infective_1stinfections * np_sum(e.I_r + e.I_nr) + hosp_per_infective_reinfections * np_sum(e.RI_r + e.RI_nr) )
            hospitalization_index_region[j] = bed_occupancy_factor + hospitalized[j] /baseline_hosp[j] 


        hospitalization_index[i] = np_amax(hospitalization_index_region) ## check this

        # True up susceptible pools between variants
        for j in range(0, nregions):
            for v in range(0,nvars):
                if nvars>1:
                    if i==0:
                        epi[j][v].S-= (np_sum(epi[j][~v].E_nr[1]) + np_sum(epi[j][~v].E_r[1]) + np_sum(epi[j][~v].Itot))
                    if i > 0:
                        epi[j][v].S-= (np_sum(epi[j][~v].E_nr[1]) + np_sum(epi[j][~v].E_r[1]))
                susceptible_over_time[j,i,v] = epi[j][v].S

    return nvars, seir_params_multivar, nregions, regions, start_time, end_time, epi_datetime_array, susceptible_over_time, \
       exposed_over_time, infective_over_time, recovered_over_time, deaths_over_time, deaths_reinf_over_time, reexposed_over_time, reinfective_over_time, \
       immune_over_time, hospitalization_index, epi
