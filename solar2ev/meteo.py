#!/usr/bin/env python3

#pip install beautifulsoup4

import requests
import bs4
from bs4 import BeautifulSoup
# pip install used python 3.8  (set vscode interpreter , bottom rigth from 3.10 to 3.8)
import re

import copy #deepcopy

import csv
from calendar import monthrange
from time import sleep
import datetime
import sys, os
import pandas as pd
import numpy as np

import config_features

from copy import deepcopy

# # extracted from the web and columns names for scrapped meteo csv
header_csv = ['date', 'hour', 'temp', 'humid', 'direction','wind', 'pressure'] 
month_name=['jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']

meteo_file = "meteo_scrapped.csv"  # scrapped data stored in this file

#date,                hour, temp,   humid,  direction,   wind,  pressure
#2021-03-01 00:00:00, 0,    0.7,    86%,    Sud,          2,     1029.4

# store day-1 to fill missing hours in day0, unless we run this the first time
# global 

prev_day = {}

first_time_one_day = True



# sorted to form circle, Nord-Nord-Ouest should be "close" to Nord
# site has both "Variable" , "Unknown". no need to keep this distinction, but still every row must have a value for direction.
# put "unknow" in "variable" when parsing
# where to put variable in the wind rose ? decide that unknown and variable are like nord , ie some kind of worst weather case scenario

direction = [
"Variable",
"Nord", "Nord-Nord-Est", "Nord-Est", "Est-Nord-Est",
"Est", "Est-Sud-Est", "Sud-Est","Sud-Sud-Est",
"Sud", "Sud-Sud-Ouest", "Sud-Ouest","Ouest-Sud-Ouest",
"Ouest", "Ouest-Nord-Ouest",  "Nord-Ouest", "Nord-Nord-Ouest"
]

# not very elegant, but works
direction_specific = [
"Nord-Nord-Est", "Est-Nord-Est", "Ouest-Nord-Ouest", "Nord-Nord-Ouest", "Ouest-Sud-Ouest", "Sud-Sud-Ouest","Est-Sud-Est", "Sud-Sud-Est",
"Sud-Est", "Nord-Ouest", "Sud-Ouest", "Nord-Est",
"Sud",  "Est", "Ouest", "Nord", "Variable", "Unknown"
]

assert len(direction) == 4*4  + 1
assert len(direction) == len(direction_specific) -1

# direction is burried in img onmouseot. 
# NOTE; direction in degres is also there. I wish I saw this earlier

# convert str to index into list done in dataframe module


############################
# GOAL: web scrapping for one day
############################


# could possibly used another method to get meteo data (as long as it returns the adequate dict)
def one_day(date: datetime, expected_hours:int, station_id = config_features.station_id):

    return one_day_meteo_ciel (date, expected_hours, station_id) 



def one_day_meteo_ciel(date: datetime, expected_hours:int, station_id = config_features.station_id) -> dict: # month 0 to 11

    # return dict with hour as str as key, 
    # fill missing hours is less than expected_hours

    # survive sequential calls to function. store value from previous day, to fix missing hours
    global first_time_one_day # track first time we run this. used to record previous day data
    global prev_day

    # cannot use non local, it excludes global
    # The nonlocal statement causes the listed identifiers to refer to previously bound variables in the nearest enclosing scope excluding globals.
    #nonlocal prev_day

    # input datetime 
    # if missing hour, use previous day same hour (if 1st day in incomplete, houston we have a problem)
    # dict should have exactly 24 entries (ie no missing hours), allows to get sequence, 0 to 23h

    # default is scrap expecting a full 24 hours

    timeout = 20 # sec

    # date can contains min, sec, depending on when this is called
    # there is an assert later which compare date to the content of the scrapped data, which does not contains valid mn, sec
    # so better erase mn, sec from date, not used anywhere anyway

    date = date.replace(minute=0, hour=0, second=0, microsecond =0)    

    year = int(date.strftime("%Y"))
    month = int(date.strftime("%m")) -1 # this is the way the web site records date  month start at 0
    day = int(date.strftime("%d"))

    # month is kept with "web site format", ie starts at zero. use +1 to print

    # month start at 0
    # https://www.meteociel.fr/temps-reel/obs_villes.php?code2=278&jour2=12&mois2=0&annee2=2022

    url = "https://www.meteociel.fr/temps-reel/obs_villes.php?code2=%s&jour2=%s&mois2=%d&annee2=%d" %(str(station_id), day,month, year)
    #print('processing: ', url)

    try:
        page = requests.get(url, timeout=timeout) # for both connect and read
    except requests.exceptions.Timeout:
        print("get Timed out", url)
        # handle sleeping in calling module
        return({})
    

    #print(page, type(page)) # <Response [200]> <class 'requests.models.Response'>
    #print(page.status_code) # 200
    #print(page.content)

    if page.status_code != 200: # request for future date returns 200
        print('error HTTP when scrapping meteo', page.status_code, url)
        return({})

    """
    tr row in table
    td data cell

    <table>
    <tr>
        <td>Cell A</td>
        <td>Cell B</td>
    </tr>

    """

    #with open('tt1.txt', 'w') as fp:
    #    fp.write(page.text)

    #############################
    # to crawl
    #   get a tag
    #    list(tag.children)
    #     get a tag of interest in the list
    #       tag.find_all 

    soup = BeautifulSoup(page.content, 'html.parser') # soup is list of tags
    #print(soup.prettify()) # html tags

    ##############
    # how to go 1 lever deeper
    ##############
    # .children is a <list_iterator object at 0x00000145FED30EB0>
    # cannot do .cildren[0]
    # convert to list list(xx.children)
    # len(list(xxx.children))
    # list(xx.children)[0] is a <class 'bs4.element.Tag'>

    
    def my_below_first(tag:bs4.element.Tag)-> bs4.element.Tag:
            # return FIRST 
            return list(tag.children)[0] # taking .children ALWAYS returns a list possibly ONE element


    tag = list(soup.children) # len 4

    """
    <class 'bs4.element.Doctype'> The first is a Doctype object, which contains information about the type of the document.
    <class 'bs4.element.NavigableString'> The second is a NavigableString, which represents text found in the HTML document.
    <class 'bs4.element.Tag'>    <html><head> The final item is a Tag object, which contains other nested tags.
    '\n'
    tag[0] 'HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd"'
    tag[1] '\n'
    tag[2] <html><head>  /body /html tag[3] '\n'
    """

    # this is the area of interest <html>
    h = tag[2] # <class 'bs4.element.Tag'>

    #title = h.find_all('title')
    # [<title>Meteociel - O...el</title>]
    # 0:<title>Meteociel - Observations Lans-en-Vercors - Les Allières (38) - données météo de la station  - Tableaux horaires en temps réel</title>
    #len():1

    tmp = list(h.children) # list of tags below html

    # tmp[0]  /head
    # tmp[1] \n
    # tmp[2] /body

    # data we look for is in body
    body = tmp[2] # <class 'bs4.element.Tag'> <body>
   
    b = list(body.children) # list of tags below body , len 5   contains 2 tables 
    # b[0] b[2] b[4] \n
    # b[1] table  b[3] table 

    
 
    """
    # summary table
    tmp = h.find_all('table', bgcolor='#FFFF99')
    summary_table = tmp[0] # there is only one. 

    row = list(summary_table.children) # list of tags
    summary = row[1]  # TAG of 2nd row, the one countaining data. the first row is header
    # <tr><td align="center">-3.3 °C</td><td align="center">-10.9 °C</td><td align="center">10 km/h</td><td align="center">0 mm</td><td align="center">N/A</td></tr>
    data = list(summary.children) # LIST [<td align="center">-....3 °C</td>, <td align="center">-....9 °C</td>, <td align="center">1... km/h</td>, <td align="center">0 mm</td>, <td align="center">N/A</td>]
    # list of tags
    for e in data:
        #print(d) # <td align="center">-3.3 °C</td> tag
        #print(list(d)[0]) # -3.3 °C  
        s = list(e)[0]
        #print(s)

    """


    #############################################
    # actual daily data table, one row per hour
    # for some day, some data missing on few hours
    ############################################

    # could use either h.find_all or body.find_all
    tmp = h.find_all('table', bordercolor='#C0C8FE', bgcolor='#EBFAF7') # returns list of ds4.element.tag
    assert len(tmp) == 1, ("found more than ONE daily data table. scrapping screwed")

    data_table = tmp[0] # only one


    one_day_table = list(data_table.children) # list of rows in table , ie ONE day

    """
    list of row, each row contains multiple data cells 
    00: <tr> <td align="center" width="6%"><b>Heure<br/>locale</b></td> <td align="center"><b>Temps</b></td><td width="10%"><div align="center"><b>Température</b></div></td><td><div align="center"><b>Humidité</b></div></td><td align="center"><b>Humidex</b></td><td align="center"><b>Windchill</b></td><td colspan="2"><div align="center"><b>Vent (rafales)</b></div></td><td width="12%"><div align="center"><b>Pression</b></div></td><td width="8%"><div align="center"><b>Précip.<br/> mm/h</b></div></td><td width="9%"><div align="center"><b>Max rain rate</b></div></td></tr>
    01: ' '
    02: <tr> <td align="center">23 h</td> <td><div align="center">    </div></td> <td><div align="center">-8.9 °C</div></td> <td><div align="center">92% </div></td> <td><div align="center">-8.9 </div></td> <td><div align="center">-15.5 </div></td> <td><div align="center"><img onmouseout="javascript:kill();" onmouseover="javascript:pop('Vent observé','&lt;font size=2&gt;&lt;i&gt;Direction : &lt;/i&gt;Nord &lt;small&gt;(8°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 16 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 27 km/h&lt;/i&gt;&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/vents/n.gif"/></div></td> <td><div align="center">16 km/h  (27 km/h)</div></td> <td><div align="center">1023.2 hPa <img onmouseout="javascript:kill();" onmouseover="javascript:pop('Pression observée','&lt;font size=2&gt;&lt;i&gt;Variation (sur 3h) : &lt;/i&gt;0.8 hPa&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/picto/haut2.gif"/></div></td> <td><div align="center"> aucune</div></td> <td><div align="center">0 mm/h </div></td> </tr>
    03: '\n'
    04: <tr> <td align="center">22 h</td> <td><div align="center">    </div></td> <td><div align="center">-8.8 °C</div></td> <td><div align="center">92% </div></td> <td><div align="center">-8.8 </div></td> <td><div align="center">-15.4 </div></td> <td><div align="center"><img onmouseout="javascript:kill();" onmouseover="javascript:pop('Vent observé','&lt;font size=2&gt;&lt;i&gt;Direction : &lt;/i&gt;Nord &lt;small&gt;(5°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 16 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 27 km/h&lt;/i&gt;&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/vents/n.gif"/></div></td> <td><div align="center">16 km/h  (27 km/h)</div></td> <td><div align="center">1022.8 hPa <img onmouseout="javascript:kill();" onmouseover="javascript:pop('Pression observée','&lt;font size=2&gt;&lt;i&gt;Variation (sur 3h) : &lt;/i&gt;0.8 hPa&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/picto/haut2.gif"/></div></td> <td><div align="center"> aucune</div></td> <td><div align="center">0 mm/h </div></td> </tr>
    05: '\n'
    06: <tr> <td align="center">21 h</td> <td><div align="center">    </div></td> <td><div align="center">-8.8 °C</div></td> <td><div align="center">92% </div></td> <td><div align="center">-8.8 </div></td> <td><div align="center">-15.4 </div></td> <td><div align="center"><img onmouseout="javascript:kill();" onmouseover="javascript:pop('Vent observé','&lt;font size=2&gt;&lt;i&gt;Direction : &lt;/i&gt;Nord &lt;small&gt;(1°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 16 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 27 km/h&lt;/i&gt;&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/vents/n.gif"/></div></td> <td><div align="center">16 km/h  (27 km/h)</div></td> <td><div align="center">1022.6 hPa <img onmouseout="javascript:kill();" onmouseover="javascript:pop('Pression observée','&lt;font size=2&gt;&lt;i&gt;Variation (sur 3h) : &lt;/i&gt;0.8 hPa&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/picto/haut2.gif"/></div></td> <td><div align="center"> aucune</div></td> <td><div align="center">0 mm/h </div></td> </tr>
   
    [<tr> <td align="cent...</td></tr>, ' ', <tr> <td align="cent.../td> </tr>, '\n', <tr> <td align="cent.../td> </tr>, '\n', <tr> <td align="cent.../td> </tr>, '\n', <tr> <td align="cent.../td> </tr>, '\n', <tr> <td align="cent.../td> </tr>, '\n', <tr> <td align="cent.../td> </tr>, '\n', ...]
    
    """

    one_day_table = one_day_table[1:] # ignore 1st row, ie table header  
 
    # dict storing each hour of a given day
    # KEY is hour as a str (could also use hour as int). note hours also in value
    # used later to check if missing hours, by trying accessing all keys

    day_result = {} 
    # dict {"23":[date,hour,temp,humid,wind,pressure],   }    
   

    # each row/hour start with  the same timestamp (for a given day)
    #add pandas timestamp (sec since 70, object)  - same as datetime datetime datetime64[ns]
    # time_stamp = pd.Timestamp('2022-02-09') Wednesday, February 9, 2022
    # Timestamp is the pandas equivalent of python’s Datetime and is interchangeable with it in most cases.
    #datetime = pd.to_datetime(time_s) # SAME timestamp
    # time_stamp.day_name(), time_stamp.month_name(), time_stamp.day, time_stamp.year

    # timestamp for day to be retreived
    # end up being the same as date, input parameter

    time_s = "%d-%02d-%02d" %(year, month+1, day) # '2021-03-01'
    time_stamp_day_retrieved = pd.Timestamp(time_s) # # 2021-01-01 00:00:00 # Timestamp('2021-03-01 00:00:00') 


    ##########################
    # go thru all row (line) for ONE day
    ##########################

    nb_values_per_hour = 7 # ie len(hour) : date, hour, temp , humid, direction, wind, pressure. used later to check we got everything

    not_nedeed =[1,4,5,6,10,11]
    index_wind = 7

    """
                # Nov 2023
           0     <td align="center">23 h</td>
           *1     <td><div align="center">    </div></td>
            2    <td><div align="center">3.4 °C</div></td>   temp
            3   <td><div align="center">86% </div></td>     humid
           *4   <td><div align="center">1.3 °C</div></td>   point de rosee
           *5     <td><div align="center">3.4 </div></td>     humid index
           *6     <td><div align="center">3.4 </div></td>     windchill
           7     <td><div align="center"><img onmouseout="javascript:kill();" onmouseover="javascript:pop('Vent observé','&lt;font size=2&gt;&lt;i&gt;Direction : &lt;/i&gt;Sud-Sud-Ouest &lt;small&gt;(202°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 0 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 0 km/h&lt;/i&gt;&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/vents/sso.gif"/></div></td>
           8     <td><div align="center">0 km/h </div></td>
           9     <td><div align="center">1015.7 hPa <img onmouseout="javascript:kill();" onmouseover="javascript:pop('Pression observée','&lt;font size=2&gt;&lt;i&gt;Variation (sur 3h) : &lt;/i&gt;0.1 hPa&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/picto/haut2.gif"/></div></td>
           *10     <td><div align="center"> aucune</div></td>
           *11     <td><div align="center">0 mm/h </div></td>
                len():
                12
    """

    for row_line_hour in one_day_table: # one row per hour   len 49, so includes more elements than hours (space, \n)

        # hour list of all entries for one hour. will be put in a dict later
        #use dict to build fixed lenth data , ie both 24hours per days, and each hour fixed len
        #I guess the web site garentee each hour has the same number of field (possibly nan)
        # some day do not have entries for all hours , making the input data not regulary spaced, which is a mess for RNN
        # even full day missing

        ###################
        #  list(tag.children)[0] returns tag below
        ###################

        # go thru all valid row/line/hour and create hour_list to record measurement
        if row_line_hour != ' ' and row_line_hour != '\n':  #  ' ' and '\n'  are found between actual table entries (ie between valid hours)
            
            # one valid row/line/hour
            # store all measurement for one hour. starts with time stamp/datetime colums
            hour_list = [time_stamp_day_retrieved]


            # list of all td for one row/line/hour
            # <tr> <td align="center">12 h</td> <td><div align="center">    </div></td> <td><div align="center">-8.3 °C</div></td> <td><div align="center">81% </div></td> <td><div align="center">-8.3 </div></td> <td><div align="center">-12.8 </div></td> <td><div align="center"><img onmouseout="javascript:kill();" onmouseover="javascript:pop('Vent observé','&lt;font size=2&gt;&lt;i&gt;Direction : &lt;/i&gt;Nord-Nord-Est &lt;small&gt;(25°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 9 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 24 km/h&lt;/i&gt;&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/vents/nne.gif"/></div></td> <td><div align="center">9 km/h  (24 km/h)</div></td> <td><div align="center">1020.9 hPa <img onmouseout="javascript:kill();" onmouseover="javascript:pop('Pression observée','&lt;font size=2&gt;&lt;i&gt;Variation (sur 3h) : &lt;/i&gt;0.1 hPa&lt;/font&gt;','#88ffff');" src="//static.meteociel.fr/cartes_obs/picto/haut2.gif"/></div></td> <td><div align="center"> aucune</div></td> <td><div align="center">0 mm/h </div></td> </tr>

            measurement_list = list(row_line_hour.children)
            measurement_list = [x for x in measurement_list if x != ' '] # clean list  12 elements

            # measurement list of td
            # goes thru all of them, interpret based on position in the list, ie position in the table 
            # ignore some, append some value in hour list

            ############################
            # make sure we understand the CURRENT form
            ############################
            assert len(measurement_list) == 12 , "table row do not have the expected number of columns. the web site has changed !!"



            ##########################
            # go thru all measurement in one row/line/hour
            # append relevant one into hour list
            ##########################

            for i, measurement_td in enumerate(measurement_list): #12 elements

                # added to hour_list using sequence index as in data received from web

                # hour
                if i == 0: #  <td align="center">23 h</td>
                    value = list(measurement_td)[0] # 23 h # convert td into actual data by converting to list and taking [0]
                    current_hour = value.split()[0] # remove h
                    #  used later as key in dict
                    hour_list.append(current_hour)

                # value not needed
                elif i in not_nedeed: 
                    pass  

                # wind direction as str is deeply buried 
                #   use BeautifulSoup's .find(), possibly with regex
                #   convert to str and use regex
                #   convert to str and define a specific list with str from most specific Nord-Est before Nord, Ouest-Sud-Ouest before Sud-Ouest and just do brutal str find

                # let's use regex
                # ie Nord Nord-Est Nord-Est-Est
                ###### Ouest-Sud-Ouest caputed as Sud-Ouest

                # give up on elegance

                elif i == index_wind: # gif wind

                    #print(measurement_td.prettify()) # encapsulated withing 2 levels <td> and <div align="center">

                    x = my_below_first(my_below_first(measurement_td))  # x is a bs4 tag. x.name "img"                    
                    #'<img onmouseout="javascript:kill();" 
                    # onmouseover="javascript:pop(\'Vent observé\',\'&lt;font size=2&gt;&lt;i&gt;
                    # Direction : &lt;/i&gt;Sud-Ouest &lt;small&gt;(236°)&lt;/small&gt;&lt;br&gt;&lt;i&gt;Vent moyen : &lt;/i&gt; 0 km/h &lt;br&gt; &lt;i&gt;Rafales max :&lt;/i&gt; 2 km/h&lt;/i&gt;&lt;/font&gt;\',\'#88ffff\');" src="//static.meteociel.fr/cartes_obs/vents/so.gif"/>'

                    x = str(x) # convert from bs4 tag

                    found_direction = False
                    # use brutal search in list ordered from more specific to least specific

                    for s in direction_specific:
                        if x.find(s) != -1:
                           
                            if s == "Unknown":
                                s = "Variable"

                            hour_list.append(s) # store as str, conversion to int done elsewhere
                            found_direction = True
                            break
                        else:
                            pass
                    else:
                        # The else block will NOT be executed if the loop is stopped by a break statement.
                        print("looks like we did not find an expected direction ")

                    if not found_direction:
                        raise Exception ("got unexpected wind direction %s" %x)

                    


                    """
                    # this does not work ###### Ouest-Sud-Ouest captured as Sud-Ouest

                    for s in ["Nord", "Sud", "Est", "Ouest", "Unknown", "Variable"]:
                        s = s +".*" # add regex:  any char, 0 or more time.  
                        ma = re.search(s, x) # returns match object

                        if ma is not None:
                            _ = ma.span() # (8,12)
                            d = ma.group()

                            #####################
                            # this is getting pretty ugly
                            # Sud-Ouest &lt
                            # Variable&lt;br&gt;&lt;i&gt;Vent

                            d = d.split("&")[0]
                            if d[-1] == " ":
                                d = d[:-1]

                            if d == "Unknown":
                                d = "Variable"

                            assert d in direction, "unknow direction %s" %d
                            hour_list.append(d) # store as str, conversion to int done elsewhere
                            found_direction = True
                            break
                        else:
                            # did not find match, keeps searching
                            pass
                            
                    else:
                        print("look like we did not find a valid direction")
                        # The else block will NOT be executed if the loop is stopped by a break statement.

                    """
                        
                    if not found_direction:
                        raise Exception ("got unexpected wind direction %s" %s)



                # not hour, wind or anything not needed
                # ie any needed measurement values, ie temp, humid%, wind, pressure
                else: 

                    try:
                        # .children is a list_iterator
                        # do list and [0] twice to get rid of <td> <div

                        value = list(list(measurement_td.children)[0].children)[0]
                        
                        # remove hpa, C, etc  note: % still there
                        hour_list.append(value.split()[0])

                    # empty cell in website, eg 3 april 2023, 3h, 26 may 2023 temp, humid missing 14-20h
                    # actually value is ['\xa0']  no break space, split is [], so [0] exception
                    # replace single measurement value with nan 

                    #  data[2] <td><div align="center"> </div></td>     
                    #  td <td><div align="center"> </div></td>
                    #  list(td.children)  [<div align="center"> </div>]
                    #  type(list(td.children)[0]) <class 'bs4.element.Tag'>
                    
                    except  Exception as e:

                        # one meteo value not there for this hour (eg July 16 2021)
                        # ie cannot get "below" measurement_td

                        print('exception for hour %s on %s' %(current_hour, date)) # 16 JULY 2021
                        print("one measurement missing (index %d). got %s (nothing there). use Nan for this measurement" %(i, measurement_td))

                        ###########################
                        # what to do when temp, humid etc not there in one or more line/hour
                        # use nan and clean later (in dataframe clean_meteo_df_from_csv)
                        ###########################
                        hour_list.append(np.nan)
                        

            # hour ready
            # hour is a  list  [Timestamp('2023-01-2...00:00:00'), '0', '-8.9', '92%', 'Nord', '16', '1023.2']

            # got all data we expected (possibly NAN)
            #  date, hour, temp, humid, wind, direction, pressure

            assert len(hour_list) == nb_values_per_hour  

            # update dict with this hour {"hour":list}
            # use later to check whether all hours are present
            # [Timestamp('2023-03-1...00:00:00'), '23', '7.8', '48%', 'Sud-Est', '26', '1014.6']
            # NOTE key is a string (use current_hour vs int(current_hours)). historical. 
            # above, current hour is a string, so keys are string. could also cast to int. important is consistency

            day_result.update({current_hour:hour_list})  # key hour as str, value list
            
        else:
            # not a valid hour (space or tab)
            pass

    # for row, ie go thru all row/line/hour in one day
        
    # scrapped everything for one day
        
    
    ###########
    # fix any missing hours
    # if an hour exists, it has the rigth number of values (possibly with nan  replacing missing individual measurement, eg temp)
    ###########

    # '\xa0 '  NSBP missing data &nbsp non breaking space
    # 10 aug 12h, 13h 12 aug 22h  19 aug 23 to 15h
    # 27 aug. 9 to 0h only. 9 to 6 missing 
    # 30 aug 10 oct, lot missing, incl full day

    # we want fixed size samples, ie day_result must be 24 hours
    # can assumes each row, ie hour, will have the same length. some data may be missing though

    # however some day do not have all hours (missing hours)

    # add missing hours with SAME HOUR from previous day (works as long as 1st day is complete)

    if first_time_one_day:
        # first time we call this function

        first_time_one_day = False 

        if len(day_result) >= expected_hours:   # CAN get more than expected, scapped today at 11pm, and expect until 7pm
            pass # no missing hours

        else:

            # no global prev_day exists
            # really no luck. do what we can

            ###################################
            # EMMERGENCY SITUATION
            # missing hours while scrapping first day (ie 1st time we call this function from main), so cannot use same hour previous day
            # fix with previous hour(s) same day, ie first available hour, before the missing one
            ###################################
            print("REALLY NO LUCK: missing hour while scrapping FIRST day (ie first time this function is called)")

            # prev_day global does not exists

            for h in range(expected_hours): # which hour is missing ?
                try:
                    last_available_hour = day_result[str(h)]  # list
                    # available hour, before the missing one. this is a list
                except Exception as e:
                    # this is the missing hour. day_result[] does not exits
                    print ('missing hour %d IN FIRST day scrap. filling with a previous valid hour %s' %(h, last_available_hour))

                    #for _ in range(nb_entries-1):
                    #    missing_data.append (np.nan)

                    ###### WARNING
                    # last_good_value is a pointer , below would change existing dict
                    #last_good_value[1] = str(h) # patch missing hour

                    # use previous good hour
                    # if day STARTS with missing hour, we are so SCREEWED
                    #  could also create a row with nan and replace/interpolate later in clean_meteo

                    try:
                        ####################
                        # https://docs.python.org/3/library/copy.html
                        # ###################

                        # if not deepcopy 
                        #l = deepcopy(last_available_hour)   # this bug was there since a long time. but hipefully this is a very rare case

                        l = copy.deepcopy(last_available_hour)

                        l[1] = str(h) # update hour element with missing hours

                        day_result.update({str(h):l})   # fix missing hour with a valid previous hour same day
                        print("")

                    except Exception as e:
                        print("Exception NO LUCK", str(e)) # last_available_hour not event set
                        raise Exception ('scrapping ONLY ONE day and days STARTS with missing hour !!!. HOUSTON WE HAVE A PROBLEM')

    else:

        # majority case
        # not the first time this function is called
        # so prev day (global dict) exist
        # use it to fix missing hour in current day

        assert (prev_day["0"][0] - time_stamp_day_retrieved).days == -1  #(a-b is timedelta when a datetime)

        if len(day_result) >= expected_hours:  # CAN get more than expected, scrapped today at 11pm, and expect until 7pm
            pass # no missing hours

        else:

            print("WARNING: missing total of %d hours while scrapping %s. got %d, expected %d. WILL FILL" %(24-len(day_result), time_stamp_day_retrieved, len(day_result), expected_hours))

            ################################
            # missing some hours, and previous days exists
            # use same hour from previous day
            #   note other approaches are also possible, eg previous hour
            #################################

            for h in range(expected_hours): # check if all hours exists
                try:
                    day_result[str(h)]

                except Exception as e:

                    # this hour is missing, used the entry from previous day, 
                    # eg 18 mars 2021 miss 2-4h , 30 june to 9 july 2021 entiere day missing, 15 mai 2023 miss 0,1,2
                    
                    print ('hour %d is indeed missing (starting at 0). ADD SAME HOUR FROM PREVIOUS DAY' %(h))

                    # do not just use entiere hour from previous day as it, it has the wrong timestamp. 
                    # add missing hour from global prev_day

                    ####### WTF !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # updating timestamp in day_result ALSO updates it in prev_day
                    # if using prev_day = day_result.copy() or  prev_day = day_result. normal this is a shallow copy 
                    # but also if using deepcopy ????

                    # UNLESS I read from prev_day using deepcopy
                    # even if at the end there is a prev_day = copy.deepcopy(day_result)
                    #  it seems that prev_day and day_result are still related

                    l = copy.deepcopy(prev_day[str(h)])

                    l[0] = time_stamp_day_retrieved

                    day_result.update({str(h):l}) 

                    #day_result[str(h)] [0] = time_stamp_day_retrieved # make sure timestam [Timestamp('2024-02-20 00:00:00'), '9', '2.6', '88%', 'Nord-Nord-Est', '9', '1031.2']

                    # could also create a row with nan and replace/interpolate later in clean_meteo
                    # dict key is string
            pass
            # all missing hours fixed using prev_day
        
    ########
    # missing hours fixed
    ########

    ##########################
    #### BIG WARNING: assigning dict (or list)
    # https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
    ##########################

    # Python never implicitly copies objects. When you set dict2 = dict1, you are making them refer to the same exact dict object, so when you mutate it, all references to it keep referring to the object in its current state.
    # If you want to copy the dict (which is rare), you have to do so explicitly with dict2 = dict(dict1) or dict2 = dict1.copy()
    # Also note that the dict.copy() is shallow, if there is a nested list/etc in there changes will be applied to both. IIRC. Deepcopy will avoid that
                    
    #Assignment statements in Python do not copy objects, they create bindings between a target and an object. 
    # For collections that are mutable or contain mutable items, a copy is sometimes needed so one can change one copy without changing the other.
    
    #prev_day = day_result.copy()
    # deepcopy(day_result) returns ??

    prev_day = copy.deepcopy(day_result)

    # if prev_day = day_result, later assignment in prev_day will ALSO modify day_result


    ###########################
    # double, tripple check before returning
    ###########################
    
    # could be larger, scrap at 11pm and expect until 7pm
    assert len(day_result) >= expected_hours , ("even after filling missing hours, day result does not have %d entries: %d" %(expected_hours,  len(day_result)))  # 24, always returns 24 hours. subsampling done elseweher
    
    for h in range(expected_hours):
        try:
            day_result[str(h)] #  key are str 
        except:
            assert Exception ('HOUR STILL MISSING AFTER FIXING', h)

    # day_result should have at least expected hours
    assert date == day_result["0"][0],  "date in day_result dict %s does not match input %s" %(date, day_result["0"][0])   # pandas timestamp is the same as datetime
    assert date == day_result[str(24-expected_hours)][0]

    # all timestamp should be the same
    for i in range(24):
        assert day_result[str(i)] [0] == time_stamp_day_retrieved, "inconsistent date after filling missing hours"

    return(day_result) # return dict 

    



############################
# INITIAL scrapping
############################
# scrap meteo into file

# start 1st match 2021 (1st pannels installed match 11th)
# end date yesterday
# returns (break) when scrapping in future
# return true or false

# create csv file passed as argument, 24 hours per day
#date,hour,temp,humid,wind,pressure
#2021-03-01 00:00:00,0,0.7,86%,2,1029.4

# calls one_day()


def scrap_meteo(file=meteo_file, start_date=datetime.datetime(2021,3,10), sleep_time = 1):
   
    print('INITIAL/BOOTSTRAP web scrapping meteociel from %s til yesterday. to file %s' %(start_date, file))

    # store result in this file (only meteo data)
    f = open(file, 'w', newline='') # avoid empty line
    writer = csv.writer(f)
    writer.writerow(header_csv)

    yesterday = datetime.datetime.now() - datetime.timedelta(1)
    date = start_date

    print("SCRAP from %s to %s" %(start_date, yesterday))

    # scap from start date till yesterday

    while date <= yesterday:
            
            try:
                # get one day, ie rows of hours
                # returns dict
                day_result = one_day(date, expected_hours=24) # dict hour key value list

                assert len(day_result) == 24
                assert len(list(set(day_result))) == 24

                if day_result != {}:  # empty dict is HTTP error
                    for h in range(24):
                        r = day_result[str(h)]
                        assert len(r) == len(header_csv)

                        # write hour in csv file
                        writer.writerow(r)

                    print('%s: scrap done' %(date))

                else: 
                    print('one_day() for %s returned empty day dict. BREAK' %(date))
                    # BTW, already checked with assert above
                    break

            except Exception as e: # missing data, or future date generated parsing exceptions
                print('exception %e while scrapping %s. BREAK' %(str(e), date))
                break

            date = date + datetime.timedelta(1)
            
            sleep(sleep_time)

    # date will not be yesterday if break
    print('scrap meteo web site ended. date %s, should be yesterday %s' %(date, yesterday))
    f.close()

    assert date == yesterday + datetime.timedelta(1), "scrap error. last date is not yesterday"

    return(True)



if __name__ == "__main__":

    file = meteo_file

    print('off line scrapping meteo to: %s' %file)

    # can scrap before installation (ie 1st day of full production, meteo will be truncated to start at installtion date)
    r = scrap_meteo(file, sleep_time=5, start_date=datetime.datetime(2021,3,10))





"""
for outer:
    for inner: 
        if break
    else:  
        # executed if no break in inner (day) loop
        continue

    # executed if break in inner loop, ie break from outer (month) loop
    break
    
"""