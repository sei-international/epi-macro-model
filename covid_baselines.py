# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:07:18 2021

@author: Eric
"""
# Run the epidemiological model first, to get hospitalizations
import sys, getopt
import re
import pandas as pd
from epidemiology_model import nregions, regions, epi_datetime_array, deaths_over_time, end_time, start_time, \
                               recovered_over_time, susceptible_over_time, exposed_over_time, infective_over_time
from macro_model import macro_datetime_array, VA

#---------------------------------------------------------------------------------------------
#
# Check what to run
#
#---------------------------------------------------------------------------------------------
try:
   opts, args = getopt.getopt(sys.argv[1:],"hm:",["help","model="])
except getopt.GetoptError:
   print('Usage: covid_baselines.py [-m <epi or both (default)>]')
   sys.exit(2)
for opt, arg in opts:
   if opt in ("-h", "--help"):
      print("Usage: covid_baselines.py [-m <epi or both (default)>]\n\tUse option \"-m epi\" or \"--model=epi\" to only run the epidemiological model")
      sys.exit()
   elif opt in ("-m", "--model"):
      model = arg.lower()
      if not (model in ("epi", "both")):
          print("Model must be either \"epi\" (for epidemiological model only) or \"both\"")

print('Running epidemiological model...')
exec(open(r'epidemiology_model.py').read())

for j in range(0,nregions):
    info = regions[j]
    d = {'date': epi_datetime_array,
         'susceptible': susceptible_over_time[j,0:end_time-start_time],
         'exposed': exposed_over_time[j,0:end_time-start_time],
         'infected': infective_over_time[j,0:end_time-start_time],
         'recovered': recovered_over_time[j,0:end_time-start_time],
         'died': deaths_over_time[j,0:end_time-start_time]}
    pd.DataFrame(data = d).to_csv('output_populations_' + re.sub(r'\s+', '_', regions[j]['name']) + '.csv', index=False)

if model != 'epi':
    print('Running macroeconomic model...')
    exec(open(r'macro_model.py').read())
    
    VA.insert(0, 'date', macro_datetime_array)
    VA.to_csv('output_value_added.csv', index=False)

print('Finished')
  