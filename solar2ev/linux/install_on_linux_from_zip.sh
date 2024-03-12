#!/usr/bin/env bash

echo 'install solar2ev from ev.zip'

# ev.zip created on development system (eg windows) and copied to linux
# please copy also this install file to linux

# zip file expected to be transfered in APP directory on linux
# content (scope of update) of zip is defined when creating it on windows

dir="/home/pi/APP"

cd $dir

# -o force overwrite
unzip -o ev.zip

cd solar2ev

rm solar2ev.log

sudo chmod +x solar2ev.py
dos2unix solar2ev.py # will be executed on linux

dos2unix linux/*  #will be executed on linux
sudo chmod +x linux/*.sh 

echo 'validating systemd timer'
systemd-analyze verify linux/solar2ev_inference.service
systemd-analyze verify linux/solar2ev_postmortem.service

systemd-analyze verify linux/solar2ev_inference.timer
systemd-analyze verify linux/solar2ev_postmortem.timer

# systemcl enable, start, status solar2ev.timer

echo '!! DONE installing from zip. run enable.sh to configure systemd timers'
echo 'systemd timers will schedule app execution (inference, postmortem, etc ..)'

