#!/usr/bin/env bash

### make sure this is executable, and run dos2unix

# customize where zip file is transfered, and app installed
dir="/home/pi/APP"

echo 'install solar2ev from ev.zip in ' $dir

# ev.zip created on development system (eg windows) and copied to linux
# please copy also this install file to linux

# zip file expected to be transfered in APP directory on linux
# content (scope of update) of zip is defined when creating it on windows


cd $dir

# -o force overwrite
unzip -o ev.zip

cd solar2ev

rm solar2ev.log

# will be executed from linux shell
sudo chmod +x solar2ev.py
dos2unix solar2ev.py 

dos2unix linux/*  # shell scripts will be executed from linux shell
sudo chmod +x linux/*.sh 

# make all executable (completion will works). still need dos2unix
sudo chmod +x *.py

echo 'verifying all systemd configuration files' 
# do not complain if ok
systemd-analyze verify linux/solar2ev_inference.service
systemd-analyze verify linux/solar2ev_postmortem.service
systemd-analyze verify linux/solar2ev_unseen.service
systemd-analyze verify linux/solar2ev_retrain.service

systemd-analyze verify linux/solar2ev_inference.timer
systemd-analyze verify linux/solar2ev_postmortem.timer
systemd-analyze verify linux/solar2ev_unseen.timer
systemd-analyze verify linux/solar2ev_retrain.timer


printf '\n\n!! DONE installing from zip. please run enable.sh to configure systemd timers'
echo 'systemd timers will schedule app execution (inference, postmortem, unseen, retrain)'

printf '\n\ndo not forget to install requirements'
#echo 'install requirements'
#cd requirements
#pip3 install -r solar2ev_req.txt

