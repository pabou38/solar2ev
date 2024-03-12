
systemd-analyze calendar "Mon,Tue *-*-01..04 12:00:00"

OnCalendar=DayOfWeek Year-Month-Day Hour:Minute:Second

 * *-*-* *:*:*
 * - To signify the day of the week eg:- Sat,Thu,Mon. Day of week. Possible values are Sun, Mon, Tue, Wed, Thu, Fri, Sat. Leave out to ignore the day of the week.


Date. Specify month and day by two digits, year by four digits. Each value can be replaced by the wildcard * to match every occurrence.
Use two dots to define a continuous range (Mon..Fri)
Use a comma to delimit a list of separate values (Mon,Wed,Fri).

can define multiple OnCalendar