# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:07:18 2021

@author: Eric
"""
# Run the epidemiological model first, to get hospitalizations

print('Running epidemiological model...')
exec(open(r'epidemiology_model.py').read())
print('Running macroeconomic model...')
exec(open(r'macro_model.py').read())
print('Finished')
