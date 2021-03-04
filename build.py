import os
import shutil

for d in ('build', 'dist'):
    if os.path.exists(d) and os.path.isdir(d):
        shutil.rmtree(d)

os.system('pyinstaller --onefile covid_baselines.py')

if os.path.exists('dist') and os.path.isdir('dist'):
    os.system('cp *.yaml dist/')
    os.system('cp input_output_data.csv dist/')
