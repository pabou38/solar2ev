#!/usr/bin/env bash

printf '\nstatus of postmortem\n\n'

# timer

printf '\ntimer enabling'
sudo journalctl -u solar2ev_postmortem.timer

printf '\nstatus of .timer, ie waiting time and last trigerred'
sudo systemctl status solar2ev_postmortem.timer


# execution
printf '\nLAST execution, incl stderr'
sudo systemctl status solar2ev_postmortem

printf '\njournal ONLY PREVIOUS stderr'
sudo journalctl -u solar2ev_postmortem

printf '\n\nplease also check log file of stdout\n'


:'
pi@pi4:~/APP $ sudo systemctl status solar2ev_postmortem.timer
● solar2ev_postmortem.timer - "solar2ev postmortem systemd Timer"
     Loaded: loaded (/lib/systemd/system/solar2ev_postmortem.timer; enabled; preset: enabled)
     Active: active (waiting) since Thu 2024-03-14 08:51:44 CET; 21min ago
    Trigger: Fri 2024-03-15 09:00:00 CET; 23h left
   Triggers: ● solar2ev_postmortem.service

Mar 14 08:51:44 pi4 systemd[1]: Started solar2ev_postmortem.timer - "solar2ev postmortem systemd Timer".



pi@pi4:~/APP $ sudo systemctl status solar2ev_postmortem
○ solar2ev_postmortem.service - "solar2ev postmortem execution"
     Loaded: loaded (/lib/systemd/system/solar2ev_postmortem.service; static)
     Active: inactive (dead) since Thu 2024-03-14 09:01:56 CET; 11min ago
   Duration: 1min 36.739s
TriggeredBy: ● solar2ev_postmortem.timer
    Process: 5521 ExecStart=/home/pi/APP/solar2ev/linux/start_solar2ev.sh -m (code=exited, status=0/SUCCESS)
   Main PID: 5521 (code=exited, status=0/SUCCESS)
        CPU: 2min 8.550s

Mar 14 09:00:19 pi4 systemd[1]: Started solar2ev_postmortem.service - "solar2ev postmortem execution".
Mar 14 09:00:42 pi4 start_solar2ev.sh[5522]: /home/pi/APP/solar2ev/solar2ev.py:1036: FutureWarning: Series.__g>
Mar 14 09:00:42 pi4 start_solar2ev.sh[5522]:   last_seen = df_model.iloc[-1] [0]
Mar 14 09:01:24 pi4 start_solar2ev.sh[5522]: /home/pi/APP/solar2ev/dataframe.py:342: FutureWarning: Series.__g>
Mar 14 09:01:24 pi4 start_solar2ev.sh[5522]:   last_seen_date= df.iloc[mode][0] # str '2022-11-23'
Mar 14 09:01:56 pi4 systemd[1]: solar2ev_postmortem.service: Deactivated successfully.
Mar 14 09:01:56 pi4 systemd[1]: solar2ev_postmortem.service: Consumed 2min 8.550s CPU time.


'