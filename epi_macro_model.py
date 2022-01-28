# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:07:18 2021

@author: Eric
"""
# Run the epidemiological model first, to get hospitalizations
import sys, getopt
import traceback
import re
from pandas import DataFrame
from numpy import size, minimum
from epidemiology_model import epidemiology_model
from macro_model import macroeconomic_model

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
model = "both" # Default
for opt, arg in opts:
   if opt in ("-h", "--help"):
      print("Usage: covid_baselines.py [-m <epi or both (default)>]\n\tUse option \"-m epi\" or \"--model=epi\" to only run the epidemiological model")
      sys.exit()
   elif opt in ("-m", "--model"):
      model = arg.lower()
      if not (model in ("epi", "both")):
          print("Model must be either \"epi\" (for epidemiological model only) or \"both\"")

print('Running epidemiological model...')
try:
    nvars, variant_params, nrgn, rgns, start, end, epi_dts, susceptible, exposed, infective, recovered, vaccinated, deaths, deaths2, reexposed, reinfective, rerecovered, hosp_ndx = epidemiology_model()
    for v in range(0,nvars):
        for j in range(0,nrgn):
            info = rgns[j]
            d = {'date': epi_dts,
                'susceptible': susceptible[j,0:end-start,v],
                'exposed': exposed[j,0:end-start,v],
                'infected': infective[j,0:end-start,v],
                'recovered': recovered[j,0:end-start,v],
                'died during 1st infection': deaths[j,0:end-start,v],
                'reexposed': reexposed[j,0:end-start,v],
                'reinfected': reinfective[j,0:end-start,v],
                'rerecovered': rerecovered[j,0:end-start,v],
                'died during reinfection': deaths2[j,0:end-start,v]}
            DataFrame(data = d).to_csv('output_populations_' + re.sub(r'\s+', '_', rgns[j]['name']) + '_' + re.sub(r'\s+', '_', variant_params[v]['name']) + '.csv', index=False)
except Exception:
    traceback.print_exc()

if model != 'epi':
    print('Running macroeconomic model...')
    try:
        macro_dts, VA = macroeconomic_model(epi_dts, hosp_ndx)
        
        ts_per_year = 365/((macro_dts[1] - macro_dts[0]).days)
        
        VA_by_year = VA
        VA_by_year.insert(0, 'year', [x.year for x in macro_dts])
        VA_annual = VA_by_year.groupby('year').agg(sum)
        VA_annual['GDP'] = VA_annual.sum(axis = 1)
        VA_ann_cov = VA_by_year.groupby('year')['year'].agg(size)
        VA_annual.insert(0, 'coverage', round(100*minimum(VA_ann_cov/ts_per_year,1))/100)
        VA_annual.to_csv('output_value_added_annual.csv', index=False)

        VA['GDP'] = VA.sum(axis=1)
        VA.insert(0, 'date', macro_dts)
        VA.to_csv('output_value_added_detailed.csv', index=False)
    except Exception:
        traceback.print_exc()

print('Finished')
