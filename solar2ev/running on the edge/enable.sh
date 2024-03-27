#!/usr/bin/env bash

# the various aspect of the application runs as short live process, scheduled with systemd
#  ie it runs does it job, do ample logging and exit. It is not a deamon
# this file enables systemd Timers
# scheduling information is found in respective .timer files

echo 'configure systemd timer'

cd /home/pi/APP/solar2ev/linux

#systemd-analyze verify solar2ev*.service
#systemd-analyze verify solar2ev*.timer

sudo cp solar2ev*.service /lib/systemd/system
sudo cp solar2ev*.timer /lib/systemd/system

# enable and start Timers

echo 'enable and start systemd Timers'

sudo systemctl enable solar2ev_postmortem.timer
sudo systemctl start solar2ev_postmortem.timer

sudo systemctl enable solar2ev_inference.timer
sudo systemctl start solar2ev_inference.timer

sudo systemctl enable solar2ev_unseen.timer
sudo systemctl start solar2ev_unseen.timer

sudo systemctl enable solar2ev_retrain.timer
sudo systemctl start solar2ev_retrain.timer

echo 'list systemd Timers'
sudo systemctl list-timers | grep solar

#NEXT                        LEFT          LAST                        PASSED       UNIT                       >
#Thu 2024-03-14 09:00:00 CET 1h 1min left  Wed 2024-03-13 09:17:23 CET 22h ago      solar2ev_postmortem.timer  >
#Thu 2024-03-14 20:00:00 CET 12h left      Wed 2024-03-13 20:00:29 CET 11h ago      solar2ev_inference.timer   >



# Timer status ()
echo 'status of .timer, ie waiting time and last trigerred'

# show NEXT trigger and waiting time (bullet is red if execution failed ?, green if OK, white if not triggered yet)
sudo systemctl status solar2ev_postmortem.timer
#Trigger: Thu 2024-03-14 09:00:00 CET; 50min left
#Triggers: ● solar2ev_postmortem.service


# log of enabling/starting timer
sudo journalctl -u solar2ev_postmortem.timer
# Feb 26 16:41:50 pi4 systemd[1]: Started solar2ev_postmortem.timer - "solar2ev postmortem systemd Timer".

# service execution. same as systemctl status ? Show execution error
sudo journalctl -u solar2ev_postmortem




sudo systemctl status solar2ev_inference.timer
sudo journalctl -u solar2ev_inference.timer

sudo systemctl status solar2ev_unseen.timer
sudo journalctl -u solar2ev_unseen.timer

sudo systemctl status solar2ev_retrain.timer
sudo journalctl -u solar2ev_retrain.timer

echo 'run also systemctl status <> to see triggering by timers'

# check execution 
# NOTE: not sure I can redirect stderr to BOTH journal and file
echo 'check stdout execution to log file'
echo 'check stderr execution to journal'


# sudo systemctl status solar2ev_inference
# show execution (and any exception)
#TriggeredBy: ● solar2ev_postmortem.timer

# before (first?) triggering
#Loaded: loaded (/lib/systemd/system/solar2ev_postmortem.service; static)
#Active: inactive (dead)
#TriggeredBy: ● solar2ev_postmortem.timer

#Feb 20 11:36:17 pi4 systemd[1]: Started solar2ev.service - "solar2ev".
#Feb 20 11:36:46 pi4 start_solar2ev.sh[3209]: /home/pi/APP/solar2ev/solar2ev.py:916: FutureWarning: Series.__getitem__ treating keys as posit>
#Feb 20 11:36:46 pi4 start_solar2ev.sh[3209]:   last_seen = df_model.iloc[-1] [0]
#Feb 20 11:37:31 pi4 systemd[1]: solar2ev.service: Deactivated successfully.
#Feb 20 11:37:31 pi4 systemd[1]: solar2ev.service: Consumed 2min 11.100s CPU time.


# static analyze OnCalendar. show waiting time
#systemd-analyze calendar "Mon..Sun *-*-2 09:00:*"

#  Original form: Mon..Sun *-*-2 09:00:*
#  Normalized form: *-*-02 09:00:*
#    Next elapse: Tue 2024-04-02 09:00:00 CEST
#       (in UTC): Tue 2024-04-02 07:00:00 UTC
#       From now: 2 weeks 4 days left
