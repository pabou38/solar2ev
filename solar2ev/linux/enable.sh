#!/usr/bin/env bash

echo 'enabling systemd timer'

cd /home/pi/APP/solar2ev/linux

#systemd-analyze verify solar2ev*.service
#systemd-analyze verify solar2ev*.timer

sudo cp solar2ev*.service /lib/systemd/system
sudo cp solar2ev*.timer /lib/systemd/system

# enable, start
sudo systemctl enable solar2ev_postmortem.timer
sudo systemctl start solar2ev_postmortem.timer

sudo systemctl enable solar2ev_inference.timer
sudo systemctl start solar2ev_inference.timer

sudo systemctl enable solar2ev_unseen.timer
sudo systemctl start solar2ev_unseen.timer

sudo systemctl enable solar2ev_retrain.timer
sudo systemctl start solar2ev_retrain.timer


# status, journalctl
echo 'see how many mn left'

sudo systemctl status solar2ev_postmortem.timer
sudo journalctl -u solar2ev_postmortem.timer

sudo systemctl status solar2ev_inference.timer
sudo journalctl -u solar2ev_inference.timer

sudo systemctl status solar2ev_unseen.timer
sudo journalctl -u solar2ev_unseen.timer

sudo systemctl status solar2ev_retrain.timer
sudo journalctl -u solar2ev_retrain.timer