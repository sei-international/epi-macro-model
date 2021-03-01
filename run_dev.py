# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:07:18 2021

@author: Eric
"""
# Run the epidemiological model first, to get hospitalizations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from epidemiology_model import nregions, regions, epi_datetime_array, deaths_over_time, end_time, start_time, \
                               new_deaths_over_time, recovered_over_time, mortality_rate_over_time, \
                               susceptible_over_time, exposed_over_time, infective_over_time, \
                               comm_spread_frac_over_time
from macro_model import x, macro_datetime_array, GDP, GDP_ref, X, F, I, u_ave, util, VA


print('Running epidemiological model...')
exec(open(r'epidemiology_model.py').read())

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


print('Running macroeconomic model...')
exec(open(r'macro_model.py').read())

units_conversion = x.monetary_units['scale'] * x.timesteps_per_year

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * GDP)
plt.plot(macro_datetime_array, units_conversion * GDP_ref)
ax.set_ylim([0,None])
plt.title('GDP')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * X)
ax.set_ylim([0,None])
plt.title('Exports')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * F)
ax.set_ylim([0,None])
plt.title('Final Demand, Excluding Investment')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.plot(macro_datetime_array, units_conversion * I)
ax.set_ylim([0,None])
plt.title('Investment')
plt.ylabel(x.monetary_units['currency'])
plt.show()

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.set_ylim([0.5,1.1])
plt.plot(macro_datetime_array, u_ave)
plt.title('Capacity Utilization')
plt.show()

VA.plot()

VA_perc = VA.divide(VA.sum(1), 0)
VA_perc.plot.area()

util.plot()

print('Finished')
  