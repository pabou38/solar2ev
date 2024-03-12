
:: comment with rem will be printed. not with ::

echo 'create requirement files'

:: ======================== pip ========================================

pip --version

:: pip install -r requirements.txt

:: if run from terminal, (type file name or double click), conda ,not found and requirement_conda is empty
rem go in conda gui, enable tf, create a powershell prompt,  and run from there

pip list --format=freeze > "my_pip_requirements.txt" 2>&1
:: pip freeze includes file path  --format=freeze is module=version 

:: pip3 install pip-chill
:: as freeze, but without dependancies, 
:: run from venv, ie conda activate tf
:: can also pip-chill --no-version
:: if not ran from venv, will output ALL installed packages
pip-chill > "my_chill_requirements.txt"

:: pip install pipreqs
:: ONLY packages needed for a given directory (SHORT LIST, with version)
:: specify encoding, else UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 3177: character maps to <undefined>
:: --print or INFO: Successfully saved requirements file in .\requirements.txt
::pipreqs --encoding utf-8 --print ..
pipreqs --encoding utf-8 --savepath my_pipreqs_requirements1.txt ..
pipreqs --encoding utf-8 --savepath my_pipreqs_requirements2.txt ../../PABOU
pipreqs --encoding utf-8 --savepath my_pipreqs_requirements2.txt ../../../Blynk


:: ======================== conda ========================================


conda list --export > "my_conda_requirements.txt" 2>&1
:: --export is module=version=source eg pypi


conda env export > "env_export.yml" 2>&1
rem env export includes dependencies (conda modules ?) and pip
:: -n tf24   or will use current env



:: conda install --file requirements.txt


:: ======================== venv ================================
:: apt install python3-pip
:: pip install virtualenv
:: cd project
:: python3.8 -m venv env_name  
:: python3 -m virtualenv -p python3 <chosen_venv_name>
::    or virtualenv env_name
:: to activate
:: linux source env_name/bin/activate
:: windows 
::  env_name/Scripts/activate.bat //In CMD
::  env_name/Scripts/Activate.ps1 //In Powershell

:: pip list , only pip and setuptools

:: which pip, which python to check in which venv we are

:: raspberry pi os 12  stackoverflow.com/questions/75602063/pip-install-r-requirements-txt-is-failing-this-environment-is-externally-mana/75696359#75696359
:: python3 -m venv tf
:: source tf/bin/activate
:: pip install