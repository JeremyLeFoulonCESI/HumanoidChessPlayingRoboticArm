"""
Run this python script to install dependencies and build the project from source.
Only supported for Python 3.10 on Windows machines.
"""


import subprocess
import sys
import os

if sys.version_info[:2] != (3, 10):
    raise NotImplementedError("Current python version not supported. Please run this build script with python version 3.10.")


if os.name != 'nt':
    raise NotImplementedError("Current OS not supported. Please run this build script on Windows.")


sources = []

entry_point = 'hcpra'

for file in os.listdir('src'):
    name, ext = os.path.splitext(file)

    if ext in ('.py', '.pyw'):
        if name == entry_point:
            sources.insert(0, os.path.join('src', file))
        else:
            sources.append(os.path.join('src', file))


pyinstaller = os.path.join(sys.exec_prefix, 'Scripts', 'pyinstaller.exe')  # only works on Windows machines...


sources_str = ' '.join(sources)

print('installing dependencies...')
subprocess.call(f'{sys.executable} -m pip install -r requirements.txt')

if not os.path.exists(pyinstaller):
    print("Couldn't find pyinstaller, installing it...")
    subprocess.call(f'{sys.executable} -m pip install pyinstaller')

print('running build...')
subprocess.call(f'build.bat {pyinstaller} {sources_str}')

