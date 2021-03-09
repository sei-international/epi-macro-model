import os
import shutil

for d in ('build', 'dist'):
    if os.path.exists(d) and os.path.isdir(d):
        shutil.rmtree(d)

os.system('pyinstaller --onefile --icon=sei_icon-32x32.ico recovrs.py')

if os.path.exists('dist') and os.path.isdir('dist'):
    os.system('cp *.yaml dist/')
    os.system('cp input_output_data.csv dist/')
