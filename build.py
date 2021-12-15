import os
import shutil

for d in ('build', 'dist'):
    if os.path.exists(d) and os.path.isdir(d):
        shutil.rmtree(d)

os.system('pyinstaller --onefile --icon=sei_icon-32x32.ico epi_macro_model.py')

if os.path.exists('dist') and os.path.isdir('dist'):
    os.system('copy *.yaml dist')
    os.system('copy input_output_data.csv dist')
