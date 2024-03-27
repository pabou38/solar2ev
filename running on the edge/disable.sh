#!/usr/bin/env bash

echo 'disabling systemd timer'

cd /home/pi/APP/solar2ev/linux


# enable, start

echo 'DISABLING ALL systemd Timers'

sudo systemctl disable solar2ev_postmortem.timer
sudo systemctl stop solar2ev_postmortem.timer

sudo systemctl disable solar2ev_inference.timer
sudo systemctl stop solar2ev_inference.timer

sudo systemctl disable solar2ev_unseen.timer
sudo systemctl stop solar2ev_unseen.timer

sudo systemctl disable solar2ev_retrain.timer
sudo systemctl stop solar2ev_retrain.timer

sudo systemctl list-timers | grep solar