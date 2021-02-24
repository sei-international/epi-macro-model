import numpy as np
from scipy import special as sp
import yaml

class SEIR_matrix:
    def __init__(self, seir_params_file: str, initial_values: dict, geography: dict):
        """ Create a new SEIR_matrix object

        Parameters
        ----------
        seir_params_file : str
            A filename for a YAML file with epidemiological parameters.
        initial_values : dict
            A set of initial values as a dict with keys:
                total population
                exposed population
                infected fraction (as a fraction of total population)
                population with community spread (as a fraction of total population)

        geography : dict
            Number of localities and geography population as a dict with keys


        Raises
        ------
        RuntimeError
            Checks to ensure that two lists that must be the same length in the seir_params_file are the same length.

        Returns
        -------
        None.

        """
        
        self.eps = 1.0e-9
        
        with open(seir_params_file) as file:
            seir_params = yaml.full_load(file)
        
        #-------------------------------------------------------------------
        # Basic epidemiological parameters
        #-------------------------------------------------------------------
        self.R0 = seir_params['R0']
        self.k = seir_params['k factor']
        self.population_at_risk_frac = seir_params['population at risk fraction']
        self.case_fatality_rate = seir_params['case fatality rate']['not at risk']
        self.case_fatality_rate_at_risk = seir_params['case fatality rate']['at risk']

        self.invisible_fraction = seir_params['unobserved fraction of cases']
    
        self.inf2rd_nr = np.array(seir_params['matrix-params']['prob recover or death given infected']['not at risk'])
        self.inf2rd_r = np.array(seir_params['matrix-params']['prob recover or death given infected']['at risk'])
        self.exp2inf = np.array(seir_params['matrix-params']['prob infected given exposed'])
        
        #-------------------------------------------------------------------
        # Parameters for the statistical model
        #-------------------------------------------------------------------
        self.n_loc = geography['number of localities']
        self.coeff_of_variation_i= seir_params['statistical-model']['coeff of variation of infected where spreading']
        fraction_of_visible_requiring_hospitalization_nr = seir_params['fraction of observed cases requiring hospitalization']['not at risk']
        fraction_of_visible_requiring_hospitalization_r = seir_params['fraction of observed cases requiring hospitalization']['at risk']
        self.overflow_hospitalized_mortality_rate_factor = seir_params['case fatality rate']['overflow hospitalized mortality rate factor']
        
        #-------------------------------------------------------------------
        # Calculated parameters based on imported parameters
        #-------------------------------------------------------------------
        # The not-at-risk and at-risk vectors should be the same length
        if len(self.inf2rd_r) != len(self.inf2rd_nr):
            raise RuntimeError("The not-at-risk and at-risk 'prob recover or death given infected' arrays must be the same length")
        # Maximum infective and exposed time periods are simply the length of the coefficient arrays
        self.infective_time_period = len(self.inf2rd_nr)
        self.exposed_time_period = len(self.exp2inf)
        # Mean infectious period is calculated based on the matrix model
        self.mean_infectious_period = 0
        P = 1
        for i in range(1,self.infective_time_period):
            ave_rate = (1 - self.population_at_risk_frac) * self.inf2rd_nr[i - 1] + self.population_at_risk_frac * self.inf2rd_r[i - 1]
            self.mean_infectious_period += (i + 1) * P * ave_rate
            P *= 1 - ave_rate
            
        self.baseline_hospitalized_mortality_rate_nr = self.case_fatality_rate / fraction_of_visible_requiring_hospitalization_nr
        self.baseline_hospitalized_mortality_rate_r = self.case_fatality_rate_at_risk / fraction_of_visible_requiring_hospitalization_r
        
        self.ave_fraction_of_visible_requiring_hospitalization = (1 - self.population_at_risk_frac) * fraction_of_visible_requiring_hospitalization_nr + \
                                        self.population_at_risk_frac * fraction_of_visible_requiring_hospitalization_r

        self.base_individual_exposure_rate = self.R0/self.mean_infectious_period
        
        # Assuming that populations of localities follow the normal rank-size rule with exponent -1, calculate ratio
        # of total population to population in the largest locality
        self.largest_loc_ranksize_mult = sum([1/x for x in range(1,self.n_loc + 1)])
        
        #-------------------------------------------------------------------
        # State variables
        #-------------------------------------------------------------------
        # Total population and population at risk
        self.N = geography['population']
        initial_infected = initial_values['infected fraction'] * self.N
        self.Itot = initial_infected
        self.Itot_prev = initial_infected
        self.N_prev = self.N

        # Exposed
        self.E = np.zeros(self.exposed_time_period + 1)
        # Infected, either not at risk (nr) or at risk (r)
        self.I_nr = np.zeros(self.infective_time_period + 1)
        self.I_r = np.zeros(self.infective_time_period + 1)
        
        self.E[1] = initial_values['exposed population']
        self.I_nr[1] = (1 - self.population_at_risk_frac) * initial_infected
        self.I_r[1] = self.population_at_risk_frac * initial_infected
        
        # Recovered: Assume none at initial time step
        self.R = 0
        self.recovered_pool = 0

        # Susceptible population
        self.S = self.N - self.Itot - self.R - initial_infected
        self.S_prev = self.S
        
        self.new_deaths = 0
        
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
        
        return 1 - sp.betainc(self.k * num_inf, num_inf + 1, 1/(1 + pub_health_factor * self.R0/self.k))
    
    def mortality_rate(self, infected_fraction: float, bed_occupancy_fraction: float, beds_per_1000: float) -> tuple:
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
        if infected_fraction == 0:
            return 0.0, 0.0
        # Calculate parameters for a beta distribution
        alpha_plus_beta = (1/self.coeff_of_variation_i**2) * (1/infected_fraction - 1) - 1
        if alpha_plus_beta <= 0:
            raise ValueError("Parameters are inconsistent with the assumed distribution of infected individuals across localities")
        alpha = alpha_plus_beta * infected_fraction
        beta = alpha_plus_beta - alpha
        
        hospital_p_i_threshold = ((1 - bed_occupancy_fraction) * beds_per_1000 / 1000) / self.ave_fraction_of_visible_requiring_hospitalization
        
        # Calculate mean exceedence fraction assuming a beta distribution
        mean_exceedance_per_infected_fraction = 1 - sp.betainc(alpha + 1, beta, hospital_p_i_threshold) - \
                    (hospital_p_i_threshold/infected_fraction) * (1 - sp.betainc(alpha, beta, hospital_p_i_threshold))
        
        # Baseline fraction
        v = 1 - self.invisible_fraction
        h = self.ave_fraction_of_visible_requiring_hospitalization
        overflow_corr = (1 - mean_exceedance_per_infected_fraction) + mean_exceedance_per_infected_fraction * self.overflow_hospitalized_mortality_rate_factor
        # Rate for population not at risk
        m_nr = v * h * overflow_corr * self.baseline_hospitalized_mortality_rate_nr
        m_r = v * h * overflow_corr * self.baseline_hospitalized_mortality_rate_r
        
        return m_nr, m_r
 
    def social_exposure_rate(self, infected_visitors: float, pub_health_factor: float) -> float:
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
        n_i = self.Itot_prev + infected_visitors
        # If there is not already community spread, the visitors may initiate it. Assume they all arrive in one locality and calculate the fraction.
        if self.comm_spread_frac == 0:
            adj_comm_spread_frac = self.largest_loc_ranksize_mult * n_i/self.N_prev
            if adj_comm_spread_frac > 1:
                raise ValueError("Number of infected visitors exceeds estimated population in largest locality")
        else:
            adj_comm_spread_frac = self.comm_spread_frac
        adj_base_individual_exposure_rate = self.base_individual_exposure_rate/adj_comm_spread_frac
        p_i = n_i/self.N_prev
        cv_corr = self.coeff_of_variation_i**2 * n_i/self.S_prev
        clust_corr = (1 - adj_comm_spread_frac) * self.N_prev/self.S_prev
        
        return pub_health_factor * adj_base_individual_exposure_rate * p_i * (1 - cv_corr - clust_corr)

    def update(self, infected_visitors: float, internal_mobility_rate: float, pub_health_factor: float, bed_occupancy_fraction: float, beds_per_1000: float):
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

        Returns
        -------
        None.

        """
        #------------------------------------------------------------------------------------------------
        # 1: Calculate new recovered or deceased and shift infected pool
        #------------------------------------------------------------------------------------------------
        recovered_or_deceased_nr = self.I_nr[self.infective_time_period]
        recovered_or_deceased_r = self.I_r[self.infective_time_period]
        for j in range(self.infective_time_period,1,-1):
            recovered_or_deceased_nr = recovered_or_deceased_nr + self.inf2rd_nr[j-1]*self.I_nr[j-1]
            recovered_or_deceased_r = recovered_or_deceased_r + self.inf2rd_r[j-1]*self.I_r[j-1]
            self.I_nr[j] = (1 - self.inf2rd_nr[j-1]) * self.I_nr[j-1]
            self.I_r[j] = (1 - self.inf2rd_r[j-1]) * self.I_r[j-1]
        self.Itot_prev = self.Itot
        self.Itot = np.sum(self.I_nr) + np.sum(self.I_r)
        
        #------------------------------------------------------------------------------------------------
        # 2: Separate recovered and deceased into recovered/deceased pools and update total population
        #------------------------------------------------------------------------------------------------
        # Ignore visitors and internal mobility for this calculation: This is due to progress of the disease alone
        if self.comm_spread_frac == 0:
            infected_fraction = 0
        else:
            infected_fraction = self.Itot/(self.comm_spread_frac * self.N)
        m_nr, m_r = self.mortality_rate(infected_fraction, bed_occupancy_fraction, beds_per_1000)
        self.new_deaths = m_nr * recovered_or_deceased_nr + m_r * recovered_or_deceased_r
        self.curr_mortality_rate = self.new_deaths/(recovered_or_deceased_nr + recovered_or_deceased_r + self.eps)
        self.recovered_pool = recovered_or_deceased_nr + recovered_or_deceased_r - self.new_deaths
        # Update recovered pool and total population
        self.R += self.recovered_pool
        self.N_prev = self.N
        self.N -= self.new_deaths
        
        #------------------------------------------------------------------------------------------------
        # 3: Update new infections and shift exposed pool
        #------------------------------------------------------------------------------------------------
        self.I_nr[1] = (1 - self.population_at_risk_frac) * self.E[self.exposed_time_period]
        self.I_r[1] = self.population_at_risk_frac * self.E[self.exposed_time_period]
        for j in range(self.exposed_time_period, 1, -1):
            new_infected = self.exp2inf[j-1] * self.E[j - 1]
            self.E[j] = self.E[j - 1] - new_infected
            self.I_nr[1] = self.I_nr[1] + (1 - self.population_at_risk_frac) * new_infected
            self.I_r[1] = self.I_r[1] + self.population_at_risk_frac * new_infected
        
        #------------------------------------------------------------------------------------------------
        # 3: Update new exposures and susceptible pool
        #------------------------------------------------------------------------------------------------
        self.E[1] = self.social_exposure_rate(infected_visitors, pub_health_factor) * self.S
        self.S_prev = self.S
        self.S -= self.E[1]

        #------------------------------------------------------------------------------------------------
        # 4: Update community spread fraction
        #------------------------------------------------------------------------------------------------
        # For this calculation, take visitors and internal mobility into account
        ni_addl = (internal_mobility_rate * self.Itot + infected_visitors)/self.n_loc
        self.comm_spread_frac += (1 - self.comm_spread_frac) * self.p_spread(ni_addl, pub_health_factor)
        