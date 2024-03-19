#!/usr/bin/env bash

####################
# start from within virtuel env
#   use CLI argument of this script as app argument
#   venv need to exist, at the same level as app
#   used by systemd Timers, or any instance of starting app from another application (ie where venv needs to be activated first)
####################

echo 'start solar2ev within (tf) environment. arguments: ' $1
dir="/home/pi/APP"

cd $dir
set -e
source "./tf/bin/activate"

cd $dir/solar2ev
python -u solar2ev.py $1