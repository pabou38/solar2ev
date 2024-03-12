#!/usr/bin/env bash
echo 'start solar2ev within (tf) environment. arguments: ' $1
dir="/home/pi/APP"

cd $dir
set -e
source "./tf/bin/activate"

cd $dir/solar2ev
python -u solar2ev.py $1