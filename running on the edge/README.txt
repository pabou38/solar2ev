
Requirements found in requirements/solar2ev_req.txt
pip install -r solar2ev_req.txt


To install solar2ev on target (Linux):

- create a zip on the development machine
- transfert the zip to Linux
- on Linux (eg Jetson) run install_on_linux_from_zip.sh
- run enable.sh, to enable systemd Timers

Note: one deployed on Linux, the application will regularly retrain on the edge.


The various aspect of solar2ev (inference, postmortem, retrain ..) runs as a short live process, scheduled with systemd.
 ie it runs, does it job, do ample logging and exit. It is not a deamon
enable.sh enables systemd Timers
scheduling information is found in respective .timer files


solar2ev is started from a virtual env. see start_solar2ev.sh 


script which run from main:
- solar2ev  (see solar2ev -h for all options)
- brutal_force (train on combination of input features and log metrics to xls file)
- synthetic_generate  (generate synthetic data)
- synthetic_train (retrain model with synthetic data), see -h for all options







tips for configuring systemd Timers:
------------------------------------

OnCalendar=DayOfWeek Year-Month-Day Hour:Minute:Second

 * *-*-* *:*:*
 * - To signify the day of the week eg:- Sat,Thu,Mon. Day of week. Possible values are Sun, Mon, Tue, Wed, Thu, Fri, Sat. Leave out to ignore the day of the week.

Date. Specify month and day by two digits, year by four digits. Each value can be replaced by the wildcard * to match every occurrence.
Use two dots to define a continuous range (Mon..Fri)
Use a comma to delimit a list of separate values (Mon,Wed,Fri).
can define multiple OnCalendar

systemd-analyze calendar "Mon,Tue *-*-01..04 12:00:00"


