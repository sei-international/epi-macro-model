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
    nrgn, rgns, start, end, epi_dts, susceptible, exposed, infective, recovered, deaths, hosp_ndx = epidemiology_model()
    for j in range(0,nrgn):
        info = rgns[j]
        d = {'date': epi_dts,
             'susceptible': susceptible[j,0:end-start],
             'exposed': exposed[j,0:end-start],
             'infected': infective[j,0:end-start],
             'recovered': recovered[j,0:end-start],
             'died': deaths[j,0:end-start]}
        DataFrame(data = d).to_csv('output_populations_' + re.sub(r'\s+', '_', rgns[j]['name']) + '.csv', index=False)
except Exception:
    traceback.print_exc()
    
if model != 'epi':
    print('Running macroeconomic model...')
    try:
        macro_dts, VA = macroeconomic_model(epi_dts, hosp_ndx)
        
        VA.insert(0, 'date', macro_dts)
        VA.to_csv('output_value_added.csv', index=False)
    except Exception:
        traceback.print_exc()

print('Finished')
  