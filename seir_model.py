from numpy import array as np_array, zeros as np_zeros, sum as np_sum, empty as np_empty, interp as np_interp, isnan as np_isnan
from scipy.special import betainc as betainc
import yaml
from common import get_datetime, timesteps_between_dates

class SEIR_matrix:
    def __init__(self, region: dict, variant: dict, common_params):
        """ Create a new SEIR_matrix object

        Parameters
        ----------

        region : dict
            Number of localities and region population as a dict with keys

        variant: dict
            Number of variants includes as a dict with paramter keys

        common_params: dict
            Parameters common to all variants and regions as a dict with
            parameter keys

        Returns
        -------
        None.

        """

        self.variant=variant['name']
        
        self.eps = 1.0e-9

        initial_values = region['initial']

        seir_params=variant

        #-------------------------------------------------------------------
        # Basic epidemiological parameters
        #-------------------------------------------------------------------
        self.start_time= get_datetime(seir_params['start date'])
        if 'Reinfection' in seir_params['R0']:
            self.R0_1stinfection = seir_params['R0']['1st infection']
            self.R0_reinfection = seir_params['R0']['Reinfection']
        else:
            self.R0_1stinfection = seir_params['R0']['1st infection']
            self.R0_reinfection = seir_params['R0']['1st infection']

        self.k = seir_params['k factor']
        if 'population at risk fraction' in seir_params:
            self.population_at_risk_frac = seir_params['population at risk fraction']
        else:
            self.population_at_risk_frac = 0.0
        # case fatalify rate for first infection
        # Given an average x_ave, an at-risk value x_r, and a fraction at risk f_r, have
        #   x_nr = (x_ave - f_r * x_r)/(1 - f_r)
        if 'at risk' in seir_params['case fatality rate among 1st infections']:
            self.case_fatality_rate_r = seir_params['case fatality rate among 1st infections']['at risk']
            self.case_fatality_rate_nr = (seir_params['case fatality rate among 1st infections']['average'] - self.population_at_risk_frac * self.case_fatality_rate_r)/(1 - self.population_at_risk_frac)
            if self.case_fatality_rate_nr < 0:
                raise ValueError("Case fatality arising from at-risk population alone exceeds the average")
            if self.case_fatality_rate_r < self.case_fatality_rate_nr:
                raise ValueError("Case fatality rate for at-risk population should exceed the average rate")
        else:
            self.case_fatality_rate_nr = seir_params['case fatality rate']['average']
            self.case_fatality_rate_r = self.case_fatality_rate_nr
        # case fatalify rate for reinfections
        if 'at risk' in seir_params['case fatality rate among reinfections']:
            self.case_fatality_rate_reinf_r = seir_params['case fatality rate among reinfections']['at risk']
            self.case_fatality_rate_reinf_nr = (seir_params['case fatality rate among reinfections']['average'] - self.population_at_risk_frac * self.case_fatality_rate_r)/(1 - self.population_at_risk_frac)
            if self.case_fatality_rate_reinf_nr < 0:
                raise ValueError("Case fatality arising from at-risk population alone exceeds the average among reinfections")
            if self.case_fatality_rate_reinf_r < self.case_fatality_rate_reinf_nr:
                raise ValueError("Case fatality rate for at-risk population should exceed the average rate among reinfections")
        else:
            self.case_fatality_rate_reinf_nr = seir_params['case fatality rate among reinfections']['average']
            self.case_fatality_rate_reinf_r = self.case_fatality_rate_reinf_nr


        self.invisible_fraction_1stinfection = seir_params['unobserved fraction of cases']['first infection']
        self.invisible_fraction_reinfection = seir_params['unobserved fraction of cases']['reinfection']

        self.exp2inf = np_array(seir_params['matrix-params']['prob infected given exposed'])
        self.rexp2rinf = np_array(seir_params['matrix-params']['prob reinfected given reexposed'])
        
        inf2rd_ave = np_array(seir_params['matrix-params']['prob recover or death given infected'])
        if 'recovery rate for at risk as fraction of not at risk' in seir_params['matrix-params']:
            rr_for_r = seir_params['matrix-params']['recovery rate for at risk as fraction of not at risk']
        else:
            rr_for_r = 1.0
        recover_rate_ratio = rr_for_r * (1 - self.case_fatality_rate_nr)/(1 - self.case_fatality_rate_r)
        adj_factor = 1 + (1 + recover_rate_ratio) * self.population_at_risk_frac
        self.inf2rd_nr = inf2rd_ave/adj_factor
        self.inf2rd_r = recover_rate_ratio * self.inf2rd_nr
        if max(self.inf2rd_r) >= 1:
            raise ValueError("Imputed recovery rate of population at risk exceeds 100% for at least one time step")

        rinf2imm_ave = np_array(seir_params['matrix-params']['prob immunity or death given reinfected']) 
        if 'recovery rate for at risk as fraction of not at risk' in seir_params['matrix-params']:
            rr_for_r = seir_params['matrix-params']['recovery rate for at risk as fraction of not at risk among reinfected']
        else:
            rr_for_r = 1.0
        rerecover_rate_ratio = rr_for_r * (1 - self.case_fatality_rate_reinf_nr)/(1 - self.case_fatality_rate_reinf_r)
        adj_factor = 1 + (1 + rerecover_rate_ratio) * self.population_at_risk_frac
        self.rinf2imm_nr = rinf2imm_ave/adj_factor
        self.rinf2imm_r = recover_rate_ratio * self.rinf2imm_nr
        if max(self.inf2rd_r) >= 1:
            raise ValueError("Imputed recovery rate of reinfected population at risk exceeds 100% for at least one time step")
        #-------------------------------------------------------------------
        # Parameters for the statistical model
        #-------------------------------------------------------------------
        self.n_loc = region['number of localities']
        self.coeff_of_variation_i= seir_params['statistical-model']['coeff of variation of infected where spreading']
        # for first infection
        if 'at risk' in seir_params['fraction of observed cases among 1st infections requiring hospitalization']:
            self.fraction_of_visible_requiring_hospitalization_r = seir_params['fraction of observed cases among 1st infections requiring hospitalization']['at risk']
            self.fraction_of_visible_requiring_hospitalization_nr = (seir_params['fraction of observed cases among 1st infections requiring hospitalization']['average'] - self.population_at_risk_frac * self.fraction_of_visible_requiring_hospitalization_r)/(1 - self.population_at_risk_frac)
        else:
            self.fraction_of_visible_requiring_hospitalization_nr = seir_params['fraction of observed cases among 1st infections requiring hospitalization']['average']
            self.fraction_of_visible_requiring_hospitalization_r = self.fraction_of_visible_requiring_hospitalization_nr
        self.overflow_hospitalized_mortality_rate_factor = seir_params['case fatality rate among 1st infections']['overflow hospitalized mortality rate factor']
        #for reinfection
        if 'at risk' in seir_params['fraction of observed reinfection cases requiring hospitalization']:
            self.fraction_of_visible_reinfections_requiring_hospitalization_r = seir_params['fraction of observed reinfection cases requiring hospitalization']['at risk']
            self.fraction_of_visible_reinfections_requiring_hospitalization_nr = (seir_params['fraction of observed reinfection cases requiring hospitalization']['average'] - self.population_at_risk_frac * self.fraction_of_visible_reinfections_requiring_hospitalization_r)/(1 - self.population_at_risk_frac)
        else:
            self.fraction_of_visible_reinfections_requiring_hospitalization_nr = seir_params['fraction of observed reinfected cases requiring hospitalization']['average']
            self.fraction_of_visible_reinfections_requiring_hospitalization_r =self. fraction_of_visible_reinfections_requiring_hospitalization_nr
        if 'overflow hospitalized mortality rate factor' in seir_params['case fatality rate among reinfections']:
            self.overflow_hospitalized_mortality_rate_factor_reinfections = seir_params['case fatality rate among reinfections']['overflow hospitalized mortality rate factor']
        else:
            self.overflow_hospitalized_mortality_rate_factorreinfections = seir_params['case fatality rate among 1st infections']['overflow hospitalized mortality rate factor']

        #-------------------------------------------------------------------
        # Calculated parameters based on imported parameters
        #-------------------------------------------------------------------
        # Maximum infective and exposed time periods are simply the length of the coefficient arrays
        self.infective_time_period = len(inf2rd_ave)
        self.exposed_time_period = len(self.exp2inf)
        self.reexposed_time_period = len(self.rexp2rinf)
        self.reinfected_time_period = len(rinf2imm_ave)
        # Mean infectious period is calculated based on the matrix model
        self.mean_infectious_period = 0
        P = 1
        for i in range(1,self.infective_time_period):
            self.mean_infectious_period += (i + 1) * P * inf2rd_ave[i - 1]
            P *= 1 - inf2rd_ave[i - 1]

        self.baseline_hospitalized_mortality_rate_nr = self.case_fatality_rate_nr / self.fraction_of_visible_requiring_hospitalization_nr
        self.baseline_hospitalized_mortality_rate_r = self.case_fatality_rate_r / self.fraction_of_visible_requiring_hospitalization_r

        self.baseline_hospitalized_mortality_rate_reinf_nr = self.case_fatality_rate_reinf_nr / self.fraction_of_visible_reinfections_requiring_hospitalization_nr
        self.baseline_hospitalized_mortality_rate_reinf_r = self.case_fatality_rate_reinf_r / self.fraction_of_visible_reinfections_requiring_hospitalization_r

        self.ave_fraction_of_visible_1stinfections_requiring_hospitalization = (1 - self.population_at_risk_frac) * self.fraction_of_visible_requiring_hospitalization_nr + \
                                        self.population_at_risk_frac * self.fraction_of_visible_requiring_hospitalization_r
        self.ave_fraction_of_visible_reinfections_requiring_hospitalization = (1 - self.population_at_risk_frac) * self.fraction_of_visible_reinfections_requiring_hospitalization_nr + \
                                        self.population_at_risk_frac * self.fraction_of_visible_reinfections_requiring_hospitalization_r # will be updated as R_nr and R_r evolve
     
        self.base_individual_exposure_rate = self.R0_1stinfection/self.mean_infectious_period

        # Assuming that populations of localities follow the normal rank-size rule with exponent -1, calculate ratio
        # of total population to population in the largest locality
        self.largest_loc_ranksize_mult = sum([1/x for x in range(1,self.n_loc + 1)])

        proportion_global_infected_points=seir_params['proportion of global infection rate']
        proportion_global_infected_npoints = len(proportion_global_infected_points)
        proportion_global_infected_traj_start = proportion_global_infected_points[0][0]
        global_infection_traj_start = common_params['global infection rate'][0][0]
        if  get_datetime(proportion_global_infected_traj_start) > get_datetime(common_params['time']['COVID start']):
            global_infection_traj_start=common_params['time']['COVID start']
        proportion_global_infected_traj_timesteps_array = np_array(range(0,timesteps_between_dates(global_infection_traj_start, common_params['time']['end date']) + 1))
        proportion_global_infected_ts = np_empty(proportion_global_infected_npoints)
        proportion_global_infected_val = np_empty(proportion_global_infected_npoints)
        for i in range(0,proportion_global_infected_npoints):
            proportion_global_infected_ts[i] = timesteps_between_dates(global_infection_traj_start, proportion_global_infected_points[i][0])
            proportion_global_infected_val[i] = proportion_global_infected_points[i][1]
        self.proportion_global_infected =  np_interp(proportion_global_infected_traj_timesteps_array, proportion_global_infected_ts, proportion_global_infected_val)

        # protective efficacy of vaccine or previous infection since month being fully vaccinated
        ndays_month=30.42
        protective_efficacy_points = np_array(seir_params['matrix-params']['protective efficacy of previous infection or inocculation'])
        protective_efficacy_npoints = len(protective_efficacy_points)
        #protective_efficacy_traj_start = protective_efficacy_points[0][0]['month']*ndays_month
        protective_efficacy_traj_end = protective_efficacy_points[protective_efficacy_npoints-1][0]['month']
        protective_efficacy_traj_timesteps_array = np_array(range(0,round(protective_efficacy_points[len(protective_efficacy_points)-1][0]['month']*ndays_month)))
        protective_efficacy_ts = np_empty(protective_efficacy_npoints)
        protective_efficacy_val = np_empty(protective_efficacy_npoints)
        for i in range(0, protective_efficacy_npoints):
            protective_efficacy_ts[i]  = round(protective_efficacy_points[i][0]['month']*ndays_month)
            protective_efficacy_val[i] = protective_efficacy_points[i][1]
        self.protective_efficacy = np_interp(protective_efficacy_traj_timesteps_array, protective_efficacy_ts, protective_efficacy_val)
        self.recovered_time_period = len(self.protective_efficacy)
        #-------------------------------------------------------------------
        # State variables
        #-------------------------------------------------------------------
        # Total population and population at risk
        self.N = initial_values['population']
        if region['name']=='Ports of entry':
            self.initial_infected = seir_params['initial']['infected fraction']['Ports of entry'] * self.N/1000 # Entered per 1000
        if region['name']=='Other provinces':
            self.initial_infected = seir_params['initial']['infected fraction']['Other provinces'] * self.N/1000 # Entered per 1000
        initial_infected = self.initial_infected
        #initial_infected = initial_values['infected fraction'] * self.N/1000 # Entered per 1000
        self.Itot = initial_infected
        self.Itot_prev = initial_infected
        self.Itot_incl_reinf_prev = self.Itot
        self.Itot_incl_reinf = self.Itot
        self.RItot = 0
        self.RItot_prev = 0
        self.N_prev = self.N

        # Exposed, either not at risk (nr) or at risk (r)  ccw(initialised at 0)
        self.E_nr = np_zeros(self.exposed_time_period + 1)
        self.E_r = np_zeros(self.exposed_time_period + 1)
        # Infected, either not at risk (nr) or at risk (r)
        self.I_nr = np_zeros(self.infective_time_period + 1)
        self.I_r = np_zeros(self.infective_time_period + 1)

        self.E_nr[1] = 0.0
        self.E_r[1] = 0.0
        self.I_nr[1] = (1 - self.population_at_risk_frac) * initial_infected
        self.I_r[1] = self.population_at_risk_frac * initial_infected

        # Recovered and re-exposed: Assume none at initial time step
        self.R_nr = np_zeros(self.recovered_time_period+1)
        self.R_r = np_zeros(self.recovered_time_period+1)
        self.new_reexposed_nr = np_zeros(self.recovered_time_period+1)
        self.new_reexposed_r = np_zeros(self.recovered_time_period+1)
        self.recovered_pool = 0
        self.recovered_pool_nr = 0
        self.recovered_pool_r = 0

        # Vaccinated
        self.vaccinated=0
        # Re-exposed, either not at risk (nr) or at risk (r)
        self.RE_nr = np_zeros(self.reexposed_time_period+1)
        self.RE_r  = np_zeros(self.reexposed_time_period+1)

        # Re -infected, either not at risk (nr) or at risk (r)
        self.RI_nr = np_zeros(self.reinfected_time_period + 1)
        self.RI_r = np_zeros(self.reinfected_time_period+ 1)

        # Immune after 2nd infection
        self.Im = 0 

        # Susceptible population
        self.S = self.N - self.Itot - np_sum(self.R_nr) - np_sum(self.R_r)
        self.S_prev = self.S

        self.new_deaths = 0
        self.new_deaths_reinf = 0

        self.comm_spread_frac = initial_values['population with community spread']

        #-------------------------------------------------------------------
        # Misc
        #-------------------------------------------------------------------
        self.curr_mortality_rate = 0


    # Probability that num_inf cases generates at least num_inf + 1 additional cases
    def p_spread(self, num_inf: float, pub_health_factor: float) -> float:
        """ Calculates the probability that num_inf cases generates at least num_inf + 1 additional cases

        Parameters
        ----------
        num_inf : float
            Number of cases. In principle this should be an integer, but it is not required.
        pub_health_factor : float
            A value from 0 to 1 that expresses the reduction from R0 to Reff due to public health measures.

        Returns
        -------
        float
            1 - cumulative probability of <= num_inf cases, assuming a negative binomial distribution.

        """

        # Average R0 over reinfected and first-infected populations. k-factor is assumed to remain the same.
        if (self.Itot+self.RItot)>0:
            ave_R0 = (self.R0_1stinfection* self.Itot + self.R0_reinfection *self.RItot) / (self.Itot+self.RItot)
        else:
            ave_R0= self.R0_1stinfection

        return 1 - betainc(self.k * (num_inf + self.eps), num_inf + 1, 1/(1 + pub_health_factor * self.R0_1stinfection/self.k))

    def mortality_rate(self, infected_fraction: float, bed_occupancy_fraction: float, beds_per_1000: float, order_of_infection: int) -> tuple:
        """ Calculates the mortality rate, taking into account bed overflow and inhomogeneity in the infected population

        Parameters
        ----------
        infected_fraction : float
            The average infection rate in areas where there is community spread of the disease.
        bed_occupancy_fraction : float
            Occupancy rate, excluding the disease. May be less than normal if other procedures are postponed or avoided.
        beds_per_1000 : float
            Number of beds per 1000 population.

        Raises
        ------
        ValueError
            The infected_fraction, combined with the 'coeff of variation of infected where spreading' value from the SEIR
            parameters file, must be consistent with the assumption of a beta distribution of infected individuals across
            localities. If not, an exception is raised.

        Returns
        -------
        tuple
            The mortality rate for not-at-risk and at-risk populations.

        """
        # Is this a first infection, or a reinfection?
        if order_of_infection==1:
            first_infection=True
            reinfection=False
            if self.Itot < self.eps:
                return 0.0, 0.0
        if order_of_infection==2:
            first_infection=False
            reinfection=True
            if self.RItot < self.eps:
                return 0.0, 0.0

        if infected_fraction == 0:
            return 0.0, 0.0
        

        # Calculate parameters for a beta distribution
        alpha_plus_beta = max(self.eps, (1/self.coeff_of_variation_i**2) * (1/(infected_fraction + self.eps) - 1) - 1)
        alpha = alpha_plus_beta * infected_fraction
        beta = alpha_plus_beta - alpha

        # average fraction requiring hospitalisation among first infections and reinfections
        if np_sum(self.RI_nr+self.RI_r)>0:
            self.ave_fraction_of_visible_reinfections_requiring_hospitalization = np_sum(self.RI_nr)/(np_sum(self.RI_nr+self.RI_r)+self.eps)* self.fraction_of_visible_reinfections_requiring_hospitalization_nr + \
                np_sum(self.RI_r)/(np_sum(self.RI_nr+self.RI_r)+self.eps) * self.fraction_of_visible_reinfections_requiring_hospitalization_r 
        else:
            self.ave_fraction_of_visible_reinfections_requiring_hospitalization = self.fraction_of_visible_reinfections_requiring_hospitalization_nr 
        
        current_fraction_of_all_visible_requiring_hospitalization = (self.ave_fraction_of_visible_1stinfections_requiring_hospitalization * self.Itot + \
                    self.ave_fraction_of_visible_reinfections_requiring_hospitalization * self.RItot) / (self.Itot +self.RItot) 

        hospital_p_i_threshold = ((1 - bed_occupancy_fraction) * beds_per_1000 / 1000) / (current_fraction_of_all_visible_requiring_hospitalization)

        # Calculate mean exceedence fraction assuming a beta distribution
        mean_exceedance_per_infected_fraction = 1 - betainc(alpha + 1, beta, hospital_p_i_threshold) - \
                    (hospital_p_i_threshold/infected_fraction) * (1 - betainc(alpha, beta, hospital_p_i_threshold))
        if current_fraction_of_all_visible_requiring_hospitalization==0:
            mean_exceedance_per_infected_fraction = 0 # CHECK THIS HACK WITH ERIC 
        
        # Baseline fraction
        if first_infection==True:
            v = 1 - self.invisible_fraction_1stinfection
            h = self.ave_fraction_of_visible_1stinfections_requiring_hospitalization
        if reinfection==True: 
            v = 1 - self.invisible_fraction_reinfection
            h = self.ave_fraction_of_visible_reinfections_requiring_hospitalization
        
        overflow_corr = (1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * self.overflow_hospitalized_mortality_rate_factor
        # Fraction of visible requiring hospitalisation and rate for population not at risk and at risk 
        if first_infection==True:
            baseline_mortality_rate_nr = self.baseline_hospitalized_mortality_rate_nr
            baseline_mortality_rate_r = self.baseline_hospitalized_mortality_rate_r
        if reinfection==True:
            baseline_mortality_rate_nr = self.baseline_hospitalized_mortality_rate_reinf_nr
            baseline_mortality_rate_r = self.baseline_hospitalized_mortality_rate_reinf_r
        
        # Note: overflow corr is dependent on all infected, "baseline" mortality rate only on specific type of infections
        m_nr = v * h * overflow_corr * baseline_mortality_rate_nr 
        m_r = v * h * overflow_corr * baseline_mortality_rate_r 
        
        if np_isnan(m_nr):
            print(m_nr)

        return m_nr, m_r

    def social_exposure_rate(self, infected_visitors: float, pub_health_factor: float, fraction_at_risk_isolated: float) -> float:
        """ Calculates the effective exposure rate per susceptible individual, taking inhomogeneity into account

        Parameters
        ----------
        infected_visitors : float
            Number of infected individuals who have arrived into the region from outside.
        pub_health_factor : float
            A value from 0 to 1 that expresses the reduction from R0 to Reff due to public health measures.

        Raises
        ------
        ValueError
            If there is not already community spread, the function assumes that any infected visitors all arrive
            in the largest locality, with a population estimated using the rank-size rule. If the number of visitors
            exceeds that population, an exception is raised.

        Returns
        -------
        float
            The effective rate of new infections per susceptible individual arcoss the region.

        """
        # Calculate total infected individuals, taking visitors into account
        n_i = self.Itot_incl_reinf_prev + infected_visitors
        # First, check if there are no infected individuals either in the population or newly introduced
        if n_i == 0 and self.comm_spread_frac == 0:
            return [0.0, 0.0]

        # If there is not already community spread, the visitors may initiate it. Assume they all arrive in one locality and calculate the fraction.
        if self.comm_spread_frac == 0:
            adj_comm_spread_frac = self.largest_loc_ranksize_mult * n_i/(self.N_prev + self.eps)
            if adj_comm_spread_frac > 1:
                raise ValueError("Number of infected visitors exceeds estimated population in largest locality")
        else:
            adj_comm_spread_frac = self.comm_spread_frac
        adj_base_individual_exposure_rate = self.base_individual_exposure_rate/(adj_comm_spread_frac + self.eps)
        p_i = n_i/(self.N_prev + self.eps)
        cv_corr = self.coeff_of_variation_i**2 * n_i/(self.S_prev + self.eps)
        clust_corr = (1 - adj_comm_spread_frac) * self.N_prev/(self.S_prev + self.eps)
        # per person in the population
        base_rate = adj_base_individual_exposure_rate * p_i * max(0, 1 - cv_corr - clust_corr)

        pub_health_factor_r = (1 - fraction_at_risk_isolated * self.population_at_risk_frac) * pub_health_factor

        return pub_health_factor * base_rate, pub_health_factor_r * base_rate

    def vaccinations(self, max_vaccine_doses: float, vaccinate_at_risk_first: bool) -> float:
        """


        Parameters
        ----------
        max_vaccine_doses: float
            Number of doses available
        vaccinate_at_risk_first: bool
            Is at risk population vaccinated first?

        Returns
        -------
        Population removed from susceptible and added to recovered pool.

        Side effects
        ------------
        Updates population_at_risk_frac

        """

        vaccinations = min(self.S, max_vaccine_doses)

        if vaccinate_at_risk_first:
            vaccinations_nr = 0
            vaccinations_r  = 0
            if self.population_at_risk_frac * self.S > vaccinations:
                vaccinations_r  = vaccinations
            if (self.population_at_risk_frac * self.S < 0 ) & (self.population_at_risk_frac * self.S < vaccinations):
                vaccinations_r  = self.population_at_risk_frac * self.S
                vaccinations_nr = vaccinations - self.population_at_risk_frac * self.S
            if self.population_at_risk_frac * self.S <= 0:  
                vaccinations_nr = vaccinations 
            self.population_at_risk_frac = max(0, self.population_at_risk_frac * self.S - vaccinations)/(self.S + self.eps)
        else:
            vaccinations_nr = (1-self.population_at_risk_frac) * vaccinations
            vaccinations_r  = self.population_at_risk_frac * vaccinations 

        return vaccinations_nr, vaccinations_r


    def update(self, infected_visitors: float,
               internal_mobility_rate: float,
               pub_health_factor: float,
               fraction_at_risk_isolated: float,
               bed_occupancy_fraction: float,
               beds_per_1000: float,
               max_vaccine_doses: float,
               vaccinate_at_risk_first: bool,
               Itot_rgn_allvars: float,
               comm_spread_frac_allvars:float,
               nvariants: int):
        """


        Parameters
        ----------
        infected_visitors : float
            DESCRIPTION.
        internal_mobility_rate : float
            DESCRIPTION.
        pub_health_factor : float
            DESCRIPTION.
        bed_occupancy_fraction : float
            DESCRIPTION.
        beds_per_1000 : float
            DESCRIPTION.
        max_vaccine_doses: float
            DESCRIPTION.
        max_vaccine_doses: float
            DESCRIPTION.
        vaccinate_at_risk_first: float
            DESCRIPTION.
        Itot_rgn_allvars: float
            DESCRIPTION.
        comm_spread_frac_allvars: float
            DESCRIPTION.
        nvariants: int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #------------------------------------------------------------------------------------------------
        # 1: Shift reinfected pool and calculated new immune after second infection
        #------------------------------------------------------------------------------------------------
        immune_or_deceased_nr = self.RI_nr[self.reinfected_time_period]
        immune_or_deceased_r = self.RI_r[self.reinfected_time_period]
        for j in range(self.reinfected_time_period,1,-1):
            immune_or_deceased_nr = immune_or_deceased_nr + self.rinf2imm_nr[j-1]*self.RI_nr[j-1]
            immune_or_deceased_r = immune_or_deceased_r + self.rinf2imm_r[j-1]*self.RI_r[j-1]
            self.RI_nr[j] = (1 - self.rinf2imm_nr[j-1]) * self.RI_nr[j-1]
            self.RI_r[j] = (1 - self.rinf2imm_r[j-1]) * self.RI_r[j-1]
        self.RI_nr[1] = self.RE_nr[self.reexposed_time_period]
        self.RI_r[1] = self.RE_r[self.reexposed_time_period]

        #------------------------------------------------------------------------------------------------
        # 2: Separate immune/deceased into immune and deceased pools and update total population
        #------------------------------------------------------------------------------------------------
        # Ignore visitors and internal mobility for this calculation: This is due to progress of the disease alone
        if self.comm_spread_frac == 0:
            infected_fraction = 0
        else:
            infected_fraction = Itot_rgn_allvars/(max(comm_spread_frac_allvars) * self.N) 
        m_nr_reinf, m_r_reinf = self.mortality_rate(infected_fraction, bed_occupancy_fraction, beds_per_1000, 2)
        self.new_deaths_reinf = m_nr_reinf * immune_or_deceased_nr + m_r_reinf * immune_or_deceased_r
        self.curr_mortality_rate_reinf = self.new_deaths_reinf/(immune_or_deceased_nr + immune_or_deceased_r + self.eps)
        self.recovered_pool_reinf = immune_or_deceased_nr + immune_or_deceased_r - self.new_deaths_reinf
        self.recovered_pool_reinf_nr = immune_or_deceased_nr - m_nr_reinf * immune_or_deceased_nr 
        self.recovered_pool_reinf_r = immune_or_deceased_r - m_r_reinf * immune_or_deceased_r
        
        #------------------------------------------------------------------------------------------------
        # 3: Update recovered pool and total population
        #------------------------------------------------------------------------------------------------
        self.Im += self.recovered_pool_reinf
        # Update N
        self.N_prev = self.N
        self.N -= self.new_deaths_reinf

        #------------------------------------------------------------------------------------------------
        # 4: Shift reexposed pool and calculated new reinfected
        #------------------------------------------------------------------------------------------------
        for j in range(self.reexposed_time_period, 1, -1):
             new_reinfected_nr = self.rexp2rinf[j-1] * self.RE_nr[j - 1]
             new_reinfected_r = self.rexp2rinf[j-1] * self.RE_r[j - 1]
             self.RE_nr[j] = self.RE_nr[j - 1] - new_reinfected_nr
             self.RE_r[j] = self.RE_r[j - 1] - new_reinfected_r
             self.RI_nr[1] = self.RI_nr[1] +  new_reinfected_nr
             self.RI_r[1] = self.RI_r[1] + new_reinfected_r
        self.RItot_prev = self.RItot
        self.RItot = np_sum(self.RI_nr) + np_sum(self.RI_r)
        
        #------------------------------------------------------------------------------------------------
        # 4: Shift recovered pool and calculated new reexposed
        #------------------------------------------------------------------------------------------------
        soc_exp_rate_nr, soc_exp_rate_r = self.social_exposure_rate(infected_visitors, pub_health_factor, fraction_at_risk_isolated)
        new_reexposed_nr = (1- self.protective_efficacy[self.recovered_time_period-1]) * soc_exp_rate_nr * self.R_nr[self.recovered_time_period] +  \
            (1-self.protective_efficacy[self.recovered_time_period-2])* soc_exp_rate_nr * self.R_nr[self.recovered_time_period-1]
        new_reexposed_r  = (1- self.protective_efficacy[self.recovered_time_period-1]) * soc_exp_rate_r * self.R_r[self.recovered_time_period] +  \
            (1-self.protective_efficacy[self.recovered_time_period-2]) * soc_exp_rate_r * self.R_r[self.recovered_time_period-1]
        self.R_nr[self.recovered_time_period] += (self.R_nr[self.recovered_time_period-1] - new_reexposed_nr )
        self.R_r[self.recovered_time_period] += ( self.R_r[self.recovered_time_period-1] - new_reexposed_r )
        self.RE_nr[1] = new_reexposed_nr
        self.RE_r[1] = new_reexposed_r
        self.new_reexposed_nr[self.recovered_time_period] = new_reexposed_nr
        self.new_reexposed_r[self.recovered_time_period] = new_reexposed_r
        for j in range(self.recovered_time_period-1, 1, -1):
            self.new_reexposed_nr[j] = soc_exp_rate_nr * (1-self.protective_efficacy[j-2]) * self.R_nr[j-1] 
            self.new_reexposed_r[j] = soc_exp_rate_r * (1-self.protective_efficacy[j-2]) * self.R_r[j-1]
            self.R_nr[j] = self.R_nr[j-1] - self.new_reexposed_nr[j] 
            self.R_r[j]  = self.R_r[j-1] - self.new_reexposed_r[j]
            self.RE_nr[1] = self.RE_nr[1]+ self.new_reexposed_nr[j] 
            self.RE_r[1]  = self.RE_r[1] + self.new_reexposed_r[j]
        #------------------------------------------------------------------------------------------------
        # 5: Calculate new recovered or deceased and shift infected pool
        #------------------------------------------------------------------------------------------------
        recovered_or_deceased_nr = self.I_nr[self.infective_time_period]
        recovered_or_deceased_r = self.I_r[self.infective_time_period]
        for j in range(self.infective_time_period,1,-1):
            recovered_or_deceased_nr = recovered_or_deceased_nr + self.inf2rd_nr[j-1]*self.I_nr[j-1]
            recovered_or_deceased_r = recovered_or_deceased_r + self.inf2rd_r[j-1]*self.I_r[j-1]
            self.I_nr[j] = max((1 - self.inf2rd_nr[j-1]) * self.I_nr[j-1], 0)
            self.I_r[j] = max((1 - self.inf2rd_r[j-1]) * self.I_r[j-1], 0)
        self.I_nr[1] = self.E_nr[self.exposed_time_period]
        self.I_r[1] = self.E_r[self.exposed_time_period]

        #------------------------------------------------------------------------------------------------
        # 6: Shift exposed pool
        #------------------------------------------------------------------------------------------------
        for j in range(self.exposed_time_period, 1, -1):
            new_infected_nr = self.exp2inf[j-1] * self.E_nr[j - 1]
            new_infected_r = self.exp2inf[j-1] * self.E_r[j - 1]
            self.E_nr[j] = self.E_nr[j - 1] - new_infected_nr
            self.E_r[j] = self.E_r[j - 1] - new_infected_r
            self.I_nr[1] = self.I_nr[1] +  new_infected_nr
            self.I_r[1] = self.I_r[1] + new_infected_r
        self.Itot_prev = self.Itot
        self.Itot = np_sum(self.I_nr) + np_sum(self.I_r)
        self.Itot_incl_reinf_prev=self.Itot_incl_reinf
        self.Itot_incl_reinf= self.Itot + self.RItot

        #------------------------------------------------------------------------------------------------
        # 7: Update new exposures and susceptible pool, taking vaccinations into account
        #------------------------------------------------------------------------------------------------
        soc_exp_rate_nr, soc_exp_rate_r = self.social_exposure_rate(infected_visitors, pub_health_factor, fraction_at_risk_isolated)
        self.E_nr[1] = (1 - self.population_at_risk_frac) * soc_exp_rate_nr * self.S
        self.E_r[1] = self.population_at_risk_frac * soc_exp_rate_r * self.S
        # Update population at risk fraction if they are proceeding at different rates
        self.population_at_risk_frac = self.population_at_risk_frac * (1 - soc_exp_rate_r)/(1 - soc_exp_rate_nr + (soc_exp_rate_nr - soc_exp_rate_r) * self.population_at_risk_frac)
        self.S_prev = self.S
        self.S -= self.E_nr[1] + self.E_r[1]
        # Do this update after accounting for newly exposed
        vaccinated_nr,  vaccinated_r = self.vaccinations(max_vaccine_doses, vaccinate_at_risk_first)
        self.S -= (vaccinated_nr + vaccinated_r)
        self.vaccinated=vaccinated_nr+vaccinated_r
        #------------------------------------------------------------------------------------------------
        # 8: Separate recovered and deceased into recovered/deceased pools and update total population
        #------------------------------------------------------------------------------------------------
        # Ignore visitors and internal mobility for this calculation: This is due to progress of the disease alone
        m_nr, m_r = self.mortality_rate(infected_fraction, bed_occupancy_fraction, beds_per_1000, 1)
        self.new_deaths = m_nr * recovered_or_deceased_nr + m_r * recovered_or_deceased_r
        self.curr_mortality_rate = self.new_deaths/(recovered_or_deceased_nr + recovered_or_deceased_r + self.eps)
        self.recovered_pool = recovered_or_deceased_nr + recovered_or_deceased_r - self.new_deaths
        self.recovered_pool_nr = recovered_or_deceased_nr - m_nr * recovered_or_deceased_nr 
        self.recovered_pool_r = recovered_or_deceased_r - m_r * recovered_or_deceased_r
    
        #------------------------------------------------------------------------------------------------
        # 9: Update recovered pool and total population
        #------------------------------------------------------------------------------------------------
        self.R_nr[1] = self.recovered_pool_nr
        self.R_r[1] = self.recovered_pool_r
        self.R_nr[1] += vaccinated_nr
        self.R_r[1] += vaccinated_r

        # Update N
        self.N_prev = self.N
        self.N -= self.new_deaths

        #------------------------------------------------------------------------------------------------
        # 10: Update community spread fraction
        #------------------------------------------------------------------------------------------------
        # For this calculation, take visitors and internal mobility into account
        ni_addl = (internal_mobility_rate * self.Itot_incl_reinf + infected_visitors)/self.n_loc
        self.comm_spread_frac += (1 - self.comm_spread_frac) * self.p_spread(ni_addl, pub_health_factor)
