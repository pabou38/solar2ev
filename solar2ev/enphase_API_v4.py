#!/usr/bin/env python3

# https://developer-v4.enphase.com/docs/quickstart.html

# https://developer-v4.enphase.com/admin/applications 
# application config
# plan, access control
# click on app name for API key, client id, secret secret and authorization url


# to generate new auth code (in case refresh token expided).
# NOTE: must include redirect_uri
# https://api.enphaseenergy.com/oauth/authorize?response_type=code&client_id=xxxxxxxx&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri
# record new auth code in secret.py


import json
import requests
from requests.auth import HTTPBasicAuth
from requests.structures import CaseInsensitiveDict

from typing import Union, Tuple, Dict

import pandas as pd
import numpy as np

import datetime
import time, sys
import json

import logging

import base64
from os.path import exists
import pprint

#sys.path.insert(1, '../PABOU') # this is in above working dir NOT NEEDED. this propagate
# unless running this module as main
sys.path.insert(1, "../all_secret")
import my_secret

system_id = my_secret.enphase["system_id"]

base_url = 'https://api.enphaseenergy.com/api/v4/systems/'  + system_id

# https://developer-v4.enphase.com/login  pabou38 neon38    
# https://developer-v4.enphase.com/admin/applications  view statistics, create apps

# https://developer-v4.enphase.com/docs/quickstart.html#step_9

#curl --location -g --request GET 'https://api.enphaseenergy.com/api/v4/systems/{system_id}/summary?key=b2b2fd806ed13efb463691b436957798' \
#--header 'Authorization: Bearer unique_access_token'


#################################
# application various credentials
# An API Key, 
# Auth URL, 
# Client ID, 
# Client Secret are generated for the application.

#https://developer-v4.enphase.com/admin/applications 
#   list all applications: meaudre ev solar , test, plans(watt), access control (system details, site level production, consumption)
#   click app name shows api key, client id, secret id

# HO data is access using API key and access token
###################################

####### CLIENT ID and CLIENT SECRET 
# specific to a given application application
# created when app is created

# client id:client secret need to be base64 encoded
# and put in header of request (authentication) 
# Bearer is for Oauth2.0

# FOR GETTING ACCESS TOKEN, The client id and client secret of the client application must be sent 
# as bearer authorization header in base64encoded(“client_id:client_secret”) 
# https://docs.python-requests.org/en/master/user/authentication

id = my_secret.enphase["solar_ev_client"] # client_id:client_secret

message_bytes = id.encode('ascii') # convert str to bytes
base64_bytes = base64.b64encode(message_bytes) # encode , generates bytes b'

# base64 convert to str, sent as bearer auth header
# header 'Authorization: Basic
credentials_str = base64_bytes.decode('ascii') 

# test decode
base64_bytes = credentials_str.encode('ascii') # convert str to bytes
message_bytes = base64.b64decode(base64_bytes) # decode, generates bytes
id_ = message_bytes.decode('ascii') # convert bytes to str

assert id == id_


######## API KEY: key for meaudre EV solar app
solar_ev_api_key = my_secret.enphase["solar_ev_api_key"] # for API calls

#apps: meaudre EV solar  plan:Watt
#Add this as a key parameter to your API calls to report and authenticate.
# WARNING: App definition defines access to system and site level production/consumption
# --header 'Authorization: Bearer unique_access_token
#10 hits per minute
#10,000 hits per month

###############################################################
####### AUTH CODE. TO BE REGENERATED if not used API for a week
# used to generate access/refresh token
###############################################################

# auth code: returned when HO (home owner accept request from client id)
# access below url, login with home owner enlighten credentials, accept , generates auth code 
# client id specific to one app

# CLICK TO GENERATE NEW Home owner auth code. auth code needed to generate access token and refresh token

# required enphase logging credentials
url_to_get_auth_code = "https://api.enphaseenergy.com/oauth/authorize?response_type=code&client_id=insertclientid&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri"

# OUTPUT
# FROM WEB: Authorization code (gxobhB) is generated.
# The application developer requires this code to generate an access_token. Using the access_token, the application developer can access your system's data.
# !!!!! can only be used ONCE
auth_code =  my_secret.enphase["auth_code"]

# parameters. can add api call specific to this dict
#data = {"key": solar_ev_api_key} 
# rather use ?key=b2b2fd806ed13efb463691b436957798'

token_json_file = "enphase_token.json"
# validity of access_token is ‘1 day’ and of refresh_token is ‘1 week’


#### API call needs API_key and access token

##########################
# make sure json file exist and is up to date (ie with valid ACCESS token)
# try api call
# return None, or raise exception

# cover all cases
#  json file does not exist
#  json file exists and valid access token
#  json file exists and invalid access token, but valid refreshed token
#  json file exists and invalid access token, and invalid refreshed token

# use before real API call to set token

##########################
def validate_token_json_file(auth_code):

    if not exists(token_json_file):

        ###########################
        # json file does not exist
        ###########################
        print("\nenphase: START FROM SCRATCH as %s does not exist. EXPECT a VALID auth code is configured in secret file" %token_json_file)

        # create json to store valid access/refresh token, using configured auth code
        if not get_token_from_auth_code(auth_code):
            # API call to get token from auth code failed. likely too old auth code (api not used for a long time)
            s = "cannot get new access/refresh token from configured auth code. likely auth code is NOT VALID (eg too old)"
            print(s)
            print("please use:", url_to_get_auth_code)
            raise Exception (s) # FAILS
        
           
        else:
            # json file have been created
            print("stored access/token in %s" %token_json_file)

            # try API call. should work
            print("try api call (system summary) with new access token to make sure")
            access_token, refresh_token = read_token_from_json() # read bacl from file

            status = system_summary(access_token) # get envoy status (str or None)

            if status != None:
                print("try api call ok. envoy summay: status ", status)  # normal or power
                print("let's roll. token json file is ready to use")
                # function returns, with json file created
            else:
                # should not happen
                raise Exception ("enphase API call fails while using brand new access token. WTF")

    else:
        #######################
        # json file exist
        #######################
        print("token json file exists, try using stored access token")
        access_token, refresh_token = read_token_from_json()

        # try one API call to see if the access token still valid
        status = system_summary(access_token) # get envoy status
        if status == None: 
            print("API call failed with current access token. getting refreshed access token")

            # get refreshed access token using stored refresh token; update json
            if not get_refreshed_access_token(refresh_token):
                # both access and refreshed token are invalid. need to start with new auth code
                raise Exception ("cannot get refreshed token. likely json file out of date. PLEASE DELETE jsonf file and retry with VALID HO auth code")
                
            else:
                access_token, refresh_token = read_token_from_json() 
                # try with refreshed access token
                status = system_summary(access_token) # try again with new (refreshed) access token
                if status == None: 
                    raise Exception ("API calls failed with refreshed access token. Should not happen")

                else:
                    print("API ok. envoy summay: status ", status)
                    print("refreshed access token is OK; json file updated")

        else:
             print("API ok. envoy summay: status ", status)
             print("json file OK")



################################################## 
# START from scratch
# creates json file with both access/refresh token 
# from auth code (configured in secret file) - after HO accept
# works only once per HO accept; POST
# return bool
##################################################

def get_token_from_auth_code(auth_code) -> bool:

    print('getting NEW access/refresh token after HO accept, using auth code: ' , auth_code)

    headers = CaseInsensitiveDict()
    headers["Authorization"] = "Basic "+ credentials_str

    url = "https://api.enphaseenergy.com/oauth/token?grant_type=authorization_code&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri&code=" + auth_code

    res = requests.post(url, headers=headers)

    if res.status_code == 200: 

        # store token in json
        resp = res.json()
        access_token = resp['access_token']
        refresh_token = resp['refresh_token']
        save = {"access_token":access_token, 'refresh_token':refresh_token}
        tmp = json.dumps(save)

        ## save tokens in json
        with open(token_json_file, 'w') as fp:
            fp.write(tmp)
        print("new access/refresh token written in %s" %token_json_file)
        return(True)
        
    else:
        print('cannot get NEW token from auth code. error REST: ', res.status_code)  
        print(res.json())
        # {'error': 'invalid_grant', 'error_description': 'Invalid authorization code: niphHT'}
        return(False)


#############################
# get refreshed token
#############################
    
# If the access_token expires, a new access token can be generated using the Client ID, Client Secret, and refresh_token.
# update json file
def get_refreshed_access_token(refresh_token):

    headers = CaseInsensitiveDict()
    headers["Authorization"] = "Basic "+ credentials_str

    url = 'https://api.enphaseenergy.com/oauth/token?grant_type=refresh_token&refresh_token='

    url = url + refresh_token
    
    print('getting refreshed token')
    res = requests.post(url, headers=headers)
    if res.status_code == 200:
        resp = res.json()
        access_token = resp['access_token']
        refresh_token = resp['refresh_token']
        save = {"access_token":access_token, 'refresh_token':refresh_token}
        tmp = json.dumps(save)
        with open(token_json_file, 'w') as fp:
            fp.write(tmp)
        return(True)
    else:
        print('cannot get refreshed token. error REST: ', res.status_code)  # 401 unauthorized
        print(res.json())
        return(False)
    

#########################
# called after validate (which make sure the json file is updated)
# so expect a valid token
#########################
def read_token_from_json():
    try:
        with open(token_json_file, 'r') as fp:
            tmp = fp.read()
            save = json.loads(tmp)
            access_token = save['access_token']
            refresh_token = save['refresh_token']
        return(access_token, refresh_token)

    except Exception as e:
        print("cannot open %s" %token_json_file)
        return(None, None)



################################
# API request
#include an OAuth 2.0 access token as Authorization header using the Bearer scheme, 
#should also include the API key of your application in header with name 'key'.
# # --header 'Authorization: Bearer unique_access_token'    string "Bearer" must be there
################################

# Monitoring API
#   System details, OK
#   Site Level Production Monitoring,  OK
#   Site Level Consumption Monitoring,  OK
#   Device Level Production Monitoring,  rest not in WATT plan
#   System Configurations,  
#   Streaming APIs

# Commissioning API
# Activations, Users, Companies , etc 

# Rate Limit 10 hits per minute 1,000 hits per month
# Access Controls, System Details, Site level Production Monitoring, Site level Consumption Monitoring

#################################################################################################################
# system details
#################################################################################################################

# /system_id
# /system_id/summary Returns system summary based on the specified system ID.
# /system_id/devices Retrieves devices for a given system. Only devices that are active will be returned in the response.
# /inverters_summary Returns the microinverters summary based on the specified active envoy serial number or system.

# system id in base url

def system_summary(access_token)->Union[None, str]: # returns status or None

    print('Returns system summary based on the specified system ID')
    """
    seems old method to pass api_key in data dict does not work anymore
    headers = CaseInsensitiveDict()
    headers["Authorization"] = "Bearer " + access_token

    # parameters. can add api  call specific to this dict
    data = {"key": solar_ev_api_key} # or ?key=b2b2fd806ed13efb463691b436957798'
    
    url = base_url  + '/summary'
    res = requests.get(url, headers=headers , data=data)
    error REST:  401
    {'message': 'Not authorized to access requested resource.', 'details': 'Keys are missing at Header location', 'code': 4
    """

    # base url includes system_id
    url = base_url  + '/summary'

    # add api key with ? in url vs data dict
    url = url +  '?key=' + solar_ev_api_key

    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})

    if res.status_code != 200:
        print('error REST: ', res.status_code)  # 401 unauthorized
        print(res.json())
        return(None)
    else:
        resp = res.json()
        print('status: %s' %resp['status'])
        print('last report: %s' %datetime.datetime.fromtimestamp(resp['last_report_at']))
        #print('size_w: %s' %resp['size_w'])
        #print('modules: %s' %resp['modules'])
        #print('energy_today: %s' %resp['energy_today'])
        #print('current_power: %s' %resp['current_power'])
        #print('summary_date: %s' %resp['summary_date'])

        id = resp['system_id']
        #print('system_id: %d (%s)' %(id, system_id))
        return(resp['status'])


# Retrieves devices for a given system. Only devices that are active will be returned in the response.
# micro, envoy, meters,  qrelays, ac battery, encharge, npower 
# status, last report, sku, model
# 'sku': 'IQ7PLUS-72-2-FR', 'model': 'IQ7+'

def system_devices(access_token):
    print("Retrieves devices for a given system ( micro, envoy, meters,  qrelays, ac battery, encharge, npower). Only devices that are active will be returned in the response")
    
    #headers = CaseInsensitiveDict()
    #headers["Authorization"] = "Bearer " + access_token

    url = base_url  + '/devices'
    url = url +  '?key=' + solar_ev_api_key
    
    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})

    if res.status_code != 200:
        print('error REST: ', res.status_code)  # 401 unauthorized
        print(res.json())
        return(None)
    else:
        resp = res.json()
        print('total_devices: %s' %resp['total_devices'])
        envoy_sn = resp['devices'] ['gateways'] [0] ['serial_number']
        print('envoy sn: %s' % envoy_sn)
        return(envoy_sn)


# Returns the microinverters summary based on the specified active envoy serial number or system
# return dict
def inverters_summary_for_envoy_sn(access_token, system_id, envoy_sn):
    print("Returns the microinverters summary based on the specified active envoy serial number or system.")

    #headers = CaseInsensitiveDict()
    #headers["Authorization"] = "Bearer " + access_token

    # specified active envoy serial number or system.
    url = 'https://api.enphaseenergy.com/api/v4/systems/inverters_summary_by_envoy_or_site?' +  'site_id=' + system_id
    url = url + '&envoy_serial_number=' + envoy_sn
    url = url +  '?key=' + solar_ev_api_key

    # W, Wh per inverter

    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})
    if res.status_code != 200:
        print('error REST: ', res.status_code)  # 401 unauthorized
        print(res.json())
        return(False)
    else:
        resp = res.json()
        pprint.pprint(resp)
        resp = resp[0] # returns list per envoy
        print("signal: ", resp["signal_strength"])

        l = resp['micro_inverters'] # list
        print("%d microinverter" %len(l))

        return(True)


#################################################################################################################
# site level production
# all starts with /systemid/
#################################################################################################################

#in WATT plan Site level Production Monitoring ,Site level Consumption Monitoring
# excluded device level production

#https://developer-v4.enphase.com/docs.html

# /production_meter_readings 
# Returns the last known reading of each production meter on the system as of the requested time, regardless of whether the meter is currently in service or retired


# /energy_lifetime 
##### production per day 
# "production": [ ..], "micro_production": [..] "meter_production": [..]


# /telemetry/production_micro 
#### powr and enwh at 5mn intervals for a span of 15mn(3), day (288) or week
#### return intervals [ of {end at, powr, enwh}]  


# /telemetry/production_meter 
#### wh_del at 15mn intervals for a span of 15mn(1), day(96), week (672)
#### return intervals [ of {end at, wh_del}]


# ===> /rgm_stats 
####  wh_del at 15mn intervals for max 1 week
#### return intervals [ of {end at, wh_del}]
#### 1 or 2 years ago ?? doc seem to contradict




##########################################################################
# /telemetry/production_micro
# telemetry for all the production micros of a system
# returns 5mn intervals,
# start_date (YYYY-MM-DD)  or start_at (epoch) with default midnight today
# granularity (15mn, day, week), ie day = 288 = 24 x 12 (there are 12 5mn intervals per hours)
# The requested start date must be within 2 years from current date
##########################################################################

def production_micro_from_date(access_token, start_date = None, granularity = "day"):

    #https://developer-v4.enphase.com/docs.html
    #start_date passed as datetime. converted to str (or epoch) here


    print("\ntelemetry for production MICRO onduler of a system. start_date %s, granularity (ie span/duration) %s" %(start_date, granularity))


    url = base_url + '/telemetry/production_micro'
    url = url +  '?key=' + solar_ev_api_key

    if start_date is not None:
        url = url + '&start_date=' + start_date.strftime("%Y-%m-%d")
    else:
        # default is today midnite, so maybe imcomplete day
        print("no start_date specified. midnite today")

    url = url + '&granularity=' + granularity


    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})
    if res.status_code != 200:
        print('error REST: ', res.status_code)  # 401 unauthorized
        print(res.json()) # {'message': 'Not Authorized', 'details': 'Not authorized to access this resource', 'code': 401}
    else:

        resp = res.json()
        #pprint.pprint(resp)

        print("%d devices reporting" %resp["total_devices"]) # 13, ie micro onduler
        print("granularity: %s" %resp["granularity"])

        intervals = resp["intervals"]

        print("%d intervals" %len(intervals)) # list of dict
        # time stamps 2700 3000 3300 , ie 300 sec, 

        a = intervals [0] ["end_at"] # epoch
        b = intervals [1] ["end_at"]

        print("1st reported interval %s" %datetime.datetime.fromtimestamp(intervals [0] ["end_at"]))
        print("last reported interval %s" %datetime.datetime.fromtimestamp(intervals [-1] ["end_at"]))
        print("%d sec between intervals" %(b-a))


        # epoch is default, and default to 0h today

        try:
            print("start_at %s" %datetime.datetime.fromtimestamp(resp["start_at"]))
            last_interval_stamp = datetime.datetime.fromtimestamp(resp["end_at"])
            print("end_at %s" %last_interval_stamp)
            print("using epochs") # epoch is default, and default to 0h today
        except:
            print("start_date %s" %(resp["start_date"]))
            print("end_date %s" %(resp["end_date"]))
            last_interval_stamp = resp["end_date"]
            print("using date") # likely for yesterday

        meta = resp["meta"]
        print("meta: last_energy %s" %datetime.datetime.fromtimestamp(meta["last_energy_at"]))
        print("meta: last_report %s" %datetime.datetime.fromtimestamp(meta["last_report_at"]))
        print("meta: operational %s" %datetime.datetime.fromtimestamp(meta["operational_at"]))
        
        wh = 0
        for i in intervals: # go thru all dict
            wh = wh + i["enwh"]

        print("wh ", wh)

        """
           'intervals': [{'devices_reporting': 12,
                'end_at': 1693112700,
                'enwh': 0,
                'powr': 1},
               {'devices_reporting': 12,
                'end_at': 1693113000,
                'enwh': 0,
                'powr': 2},
        """

        return(wh, last_interval_stamp)
    

############################################################################
# /telemetry/production_meter
# called from get_telemetry_energy_from_date()
#############################################################################
def production_meter_from_date(access_token, start_date, granularity = 'day'):


    # telemetry for all the production meters of a system.
    # 15mn intervals
    # interval 0h to 24h
    # The requested start date must be within 2 years from current date

    # from start (no end, uses granularity for span)
    # can use start_at epoch or start_date string YYYY-MM-DD.
    # _at in response, _date in response if in request
    # defaults to midnight today

    # used to get production for ongoing day (ie today)

    #https://developer-v4.enphase.com/docs.html

    #start_date passed as datetime. converted to str here

    print("\ntelemetry from production METERS. start_date %s, granularity (ie span/duration) %s" %(start_date, granularity))

    url = base_url + '/telemetry/production_meter'
    url = url +  '?key=' + solar_ev_api_key

    url = url +  '&granularity=' + granularity

    if start_date is not None:
        url = url +  '&start_date=' + start_date.strftime("%Y-%m-%d") # use start_date vs start_at (epoch)

    else:
        # default is today midnite, so maybe incomplete day
        print("no start_date specified. midnite today. epoch (ie end_at and start_at in response)")


    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})


    if res.status_code != 200:
        print('error REST ', res.status_code)
        print(res.json())
        return(False)
    
    else:
        prod = res.json() # convert to dict
        #pprint.pprint(prod)

        # 900 sec, ie 15mn between intervals
        # not using start_date default to 0h AND epoch mode
        # device reporting = 1 (meter, ie envoy)
        """
        {'end_at': 1700739900,
        'granularity': 'day',
        'intervals': [{'devices_reporting': 1, 'end_at': 1700694900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1700695800, 'wh_del': 0},


                {'devices_reporting': 1, 'end_at': 1700739000, 'wh_del': 975},
               {'devices_reporting': 1, 'end_at': 1700739900, 'wh_del': 970}],
        'items': 'intervals',
        'start_at': 1700694000,
        'system_id': 2160737,
        'total_devices': 1}
        """

        intervals = prod["intervals"]
        print("got %d (15mn) intervals" %len(intervals))
    
        # make sure intervals are 15mn (check with first interval)
        # end_at: epochs
        a = intervals [0] ["end_at"]
        b = intervals [1] ["end_at"]
        print("%d sec between intervals. should be 15mn" %(b-a)) # delta in sec for 1st interval
        assert (b-a) == 15*60

        # end_at: epochs
        print("first interval epoch: %s" %datetime.datetime.fromtimestamp(intervals[0] ["end_at"]) )
        print("last interval epoch: %s" %datetime.datetime.fromtimestamp(intervals[-1] ["end_at"]) )
        # FULL DAY, ie yesterday first interval epoch: 2023-11-22 00:15:00 last interval epoch: 2023-11-23 00:00:00

        # not using start_date default to 0h AND epoch mode

        try:
            print("start_at: %s" %datetime.datetime.fromtimestamp(prod["start_at"]))

            last_interval_stamp = datetime.datetime.fromtimestamp(prod["end_at"])
            print("end_at: %s" %last_interval_stamp)


        except:
            print("start_at and end_at (epoch) not present in response. used start_date ?")

            try:
                print("start_date", prod["start_date"])
                print("end_date", prod["end_date"])
                last_interval_stamp = prod["end_date"]
                # FULL day, ie yesterday start_date 2023-11-22T00:00:00+01:00, end_date 2023-11-23T00:00:00+01:00
            except:
                print("cannot print start_date, end_date")


        # compute sum of all wh for all intervals
        # this will run after sunset, so all "sunny" intervals are there

        wh = 0
        for i in intervals: # go thru all dict
            wh = wh + i["wh_del"]

        print("wh total:", wh)
        print("last interval", last_interval_stamp)

        return(wh, last_interval_stamp)
        # last interval is either str or datetime. need to clean
    


##################################################################
# /rgm_stats
# Returns performance statistics as measured by the revenue-grade meters 
# start_at : epoch  default midnight today . use multiple of 15mn to avoid rouding
# end_at: epoch (NO SPAN). max one week
# CANNOT use start_date
# 15 mn intervals, start at the top of the hour.
# MAX 1 week: The requested date range in one API hit cannot be more than 7 days 
# MAX 2 YEARS AGO: the requested start at must be within 2 years from current time.

# SEEMS SAME AS /telemetry/production_meter (except can mess with non 15mn bondary intervals)
####################################################################

def envoy_rgm_stats(access_token, start_date, end_date) -> list:
    # /rgm_stats
    #https://developer-v4.enphase.com/docs.html

    # API uses start_at, so conversion from datetime to epoch done here, as start, end passed as datetime

    """
    {'intervals': [{'devices_reporting': 1, 'end_at': 1693088100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693089000, 'wh_del': 0},
    """

    #start_date = start_date.strftime("%Y-%m-%d")
    #end_date = end_date.strftime("%Y-%m-%d")

    print('getting rgm stats from: %s to %s' %(start_date, end_date))

    url = base_url + '/rgm_stats'
    url = url +  '?key=' + solar_ev_api_key

    # add parameters to data (dict), contains application key already
 
    # Start of period to report on in Unix epoch time. If no start is specified, the assumed start is midnight today,
    # .timestamp() returns float
    # End of reporting period in Unix epoch time. If no end is specified, default to the time of the request or (start time + 1 week), whichever is earlier

    url = url +  '&start_at=' + str(int(start_date.timestamp()))
    url = url +  '&end_at=' + str(int(end_date.timestamp()))

    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})

    if res.status_code != 200:
        print('error enphase REST ', res.status_code)
        print(res.json())
        return([])
    else:


        ##################
        # intervals
        ##################

        print('intervals:')
        resp = res.json()
        intervals = resp["intervals"]
        print("%d intervals" %len(intervals))

        i = intervals[0] ["end_at"]
        print("fist interval %s" %datetime.datetime.fromtimestamp(i))
        i1 = intervals[1] ["end_at"]
        print("2nd interval %s" %datetime.datetime.fromtimestamp(i1))

        print("%d sec between intervals" %(i1-i))

        i1 = intervals [-1] ["end_at"]
        print("last interval %s" %datetime.datetime.fromtimestamp(i1))

        # default starts today, 900 sec or 15mn
        # fist interval 2023-08-29 00:15:00
        # 2nd interval 2023-08-29 00:30:00

        wh_del=0
        for i in intervals:
            wh_del = wh_del + i["wh_del"]
        print("wh_del %0.1f" %wh_del)

        ##################
        # meters
        # same as intervals
        # one per envoy ?
        ##################
        print('meter intervals:')
        m_intervals = resp["meter_intervals"] [0] ["intervals"]
        print("%d meter intervals" %len(m_intervals))

        i = m_intervals[0] ["end_at"]
        print("fist interval %s" %datetime.datetime.fromtimestamp(i))
        i1 = m_intervals[1] ["end_at"]
        print("2nd interval %s" %datetime.datetime.fromtimestamp(i1))

        print("%d sec between intervals" %(i1-i))

        i1 = m_intervals[-1] ["end_at"]
        print("last interval %s" %datetime.datetime.fromtimestamp(i1))

        wh_del1=0
        for i in m_intervals:
            if i["wh_del"] is not None:
                # {'channel': 2, 'wh_del': None, 'curr_w': None, 'end_at': 1693260900}
                wh_del1 = wh_del1 + i["wh_del"]
        print("wh_del %0.1f" %wh_del1)


        meta = resp["meta"]
        print("last energy:" , datetime.datetime.fromtimestamp(meta["last_energy_at"]))
        print("last report:" , datetime.datetime.fromtimestamp(meta["last_report_at"]))
        print("operational:" , datetime.datetime.fromtimestamp(meta["operational_at"]))

        #pprint.pprint(resp)

        assert wh_del == wh_del1
        
        return(wh_del)
    

############################################################
# wrapped by get_daily_solar_from_date_to_date

# uses '/energy_lifetime'

# Returns a daily time series of energy produced by the system over its lifetime. All measurements are in Watt hours.
# The time series includes one entry for each day from the start_date to the end_date with no gaps in the time series
# start_date (str). default system’s operational date
# end_date (str). Defaults to yesterday
# date format YYYY-MM-DD
# use meter (if installed), otherwize microinverters
# production = “all”, returns the merged time series plus the time series as reported by the microinverters and the meter on the system.
##############################################################
def get_daily_energy_list(access_token, start_date, end_date) -> list:

    ####### WTF {'message': 'Not authorized to access requested resource.', 'details': 'Keys are missing at Header location',  'code': 401}

    #https://developer-v4.enphase.com/docs.html

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")   # even if end_date is time orf running -1 day (ie has hour not zero), this is ignored here

    print('enphase API: getting daily energy from: %s to %s' %(start_date, end_date))

    url = base_url + '/energy_lifetime'
    url = url +  '?key=' + solar_ev_api_key

    # add parameters to data (dict), contains application key already

    # When “all”, returns the merged time series plus the time series as reported by the microinverters and the meter on the system. Other values are ignored.
    # production = microproduction
    #url = url + '&production=' + "all"

    url = url + '&start_date=' + start_date #default system operational state
    url = url + '&end_date=' + end_date # default yesterday

    #data={}
    #data["production"] = "all"
    #data["start_date"] = start_date #default system operational state
    #data["end_date"] = end_date # default yesterday

    res = requests.get(url, headers={'Authorization': 'Bearer ' + access_token})

    if res.status_code != 200:
        print('error enphase REST ', res.status_code)
        print(res.json())
        # {'message': 'Not authorized to access requested resource.', 'details': 'Keys are missing at Header location',  'code': 401}
        return([])
    
    else:

        resp = res.json()
        #pprint.pprint(resp)

        """"
        with all

        {'meta': {'last_energy_at': 1700738100,
          'last_report_at': 1700738215,
          'operational_at': 1615473912,
          'status': 'normal'},
 'meter_production': [2999,
                      5570,
                      12578,
                      6713,
                      11374,
                      7484,
                      13232,
                      8367,
                      2196,
                      4105],
 'micro_production': [2898,
                      5468,
                      12560,
                      6622,
                      11309,
                      7366,
                      13182,
                      8258,
                      2085,
                      4022],
 'production': [2898, 5468, 12560, 6622, 11309, 7366, 13182, 8258, 2085, 4022],
 'start_date': '2023-11-13',
 'system_id': 2160737}

        """

        print('enphase API: energy date to date. response start date:', resp["start_date"])

        assert resp["start_date"] == start_date # str

        s= resp["meta"]["status"]

        if s not in  ['normal']:
            s = "WARNING: envoy status is not normal: %s" %s # internet down for multiple days
            print(s)
            logging.warning(s)
        #assert resp["meta"]["status"] == 'normal'
            
        prod_list = resp["production"]

            
        #################
        # bummer. when internet is down for multiples days, some production values are 0
        # this could confuse the model I guess
        # use pandas to interpolate.
        # if last data is 0, no luck, live with it
        #################

        # [] will return []
        # [0,1] will return [1,1] 
        # [0,1,0] will return [1,1,1]
        # [0] will return [nan] nan is float

         # make sure [0] ie only one day, and missing day, returns something we can live with, ie []
        if prod_list == [0]:
            prod_list = []
            s= "WARNING. daily production is [0]. intrnet was down and really no luck. cannot interpolate, replaces with %s. " %prod_list
            print(s)
            logging.error(s)

        else:
            if 0 in prod_list:
                s= "WARNING. daily production data has zero %s. maybe internet was down. will interpolate" %prod_list
                print(s)
                logging.warning(s)

                if prod_list[0] == 0 or prod_list[-1] == 0:
                    s= "WARNING. daily production data starts or end with zero %s. will still interpolate" %prod_list
                    print(s)
                    logging.error(s)

            
                # convert 0 to nan, and use pandas to interpolate
                # l1 = [x for x in l if x !=0 else -1]  cannot have else in list comprehension
                p1 = []
                for x in prod_list:
                    if x == 0:
                        x = np.nan
                    p1.append(x)

                df = pd.DataFrame(p1)

                # use interpolate, or fillna
                # limit_direction seems to garentee that non nan data will be interpolated if the 1st value is nan (ie no luck)
                df = df.interpolate(method='linear', limit_direction="both")
                prod_list = df[0].tolist()
            else:
                pass # no need to interpolate
        
        return(prod_list) # list


###################
# get daily production (date to date , included), returned as a list 
# wrapper for get_daily_energy_list
# WTF: cloud updated LATE the next day, 7pm CET not updated . but CET is 9h ahead of CET anyway
####################
def get_daily_solar_from_date_to_date(from_date: datetime.datetime, end_date:datetime.datetime) -> list:

    # manage token
    # complete days only, ie up to yesterday (do not use for today)

    # date can have mn, sec. will be formated just before REST API call
    print('\nget daily production from %s (included) to %s (included)' %(from_date, end_date)) # get daily production from 2023-04-19 00:00:00 to 2023-11-23 11:58:36.817388

    nb_days = (end_date - from_date).days + 1 # (end-from) is datetime.timedelta. from and end date included
    print("requesting %d days" %nb_days)

    validate_token_json_file(my_secret.enphase["auth_code"])
    # json file with correct token is there ,  or exception (eg out of date token json file)

    access_token, refresh_token = read_token_from_json()

    ###### get all solar in one go
    try:
        production_list = get_daily_energy_list(access_token, from_date, end_date) # list
        print("production list", len(production_list), production_list)

    except Exception as e:
        raise Exception("cannot get daily solar from date to date with enphase API %s" %str(e))
    

    #######################
    # WTF. 02/01 data NOT returned when running query on 02/02 at 14:50 CET. cloud not yet updated ??
    # returned list does not have the expected number, ie missing 02/01
    # ok if running 19:40 

    # WTF 2. if start date = end date (both included), returns 0 (expected 1 day in list)
    #######################
    assert len(production_list) == nb_days , "get daily solar date to date. returned %s , expected %d " %(len(production_list), nb_days)
    
    return(production_list)


##########################################
# get solar production from telemetry
# simple wrapper for production_meter
# called by inference and possibly postmortem to get today's solar
##########################################
def get_telemetry_energy_from_date(start_date = None):

    print('\nget solar production using production meter for date %s' %start_date)

    # start_date default to None, ie midnite today

    validate_token_json_file(my_secret.enphase["auth_code"])
    # json file with correct token is there ,  or exception (eg out of date token json file)

    access_token, refresh_token = read_token_from_json()

    try:
        # granularity default to day
        # those get production at 15mn interval since 0h today
        wh, end_date = production_meter_from_date(access_token, start_date=start_date, granularity='day')

        if wh == 0:
            s = "production meter for one day returns zero. maybe internet was down"
            print(s)
            logging.warning(s)

        return(wh, end_date)
    
    except Exception as e:
        raise Exception("cannot get production meter with enphase API %s" %str(e))


############################################################
# main
# unit testing
# https://developer-v4.enphase.com/docs.html
############################################################

if __name__ == "__main__":

    print("unit test enphase API v4")

    ###################
    # test token
    ###################

    # make sure token json file exist with valid access token. 
    # possibly uses refresh token 
    # or auth code is json file not there
    validate_token_json_file(auth_code) # return nothing or raise exception
    
    access_token, refresh_token = read_token_from_json() # access token should be valid

    ####################
    # test telemetry
    ####################

    """
    (wh, last_interval_stamp) = get_telemetry_energy_from_date()
    print("get today's solar with telemetry\n" , wh, last_interval_stamp)
    # running at 19:30 CET, last interval is  datetime.datetime(2024, 2, 3, 19, 15)

    (wh, last_interval_stamp) = get_telemetry_energy_from_date(start_date = datetime.date.today() + datetime.timedelta(-1))
    print("get yestarday's solar with telemetry\n" , wh, last_interval_stamp)
    # last_interval is str '2024-02-03T00:00:00+01:00'

    # but value seem ok
    """


    ####################
    # test daily production
    ####################


    print("test when cloud updated late afternoon with daily production from previous day")
    # 19:40 cet is OK

    #from_date = datetime.date(2024, 1, 30)
    from_date = datetime.date.today() + datetime.timedelta(-3)
    end_date = datetime.date.today() + datetime.timedelta(-1) 

    print("from %s to %s" %(from_date, end_date))

    delta = end_date - from_date # timedelta
    print("delta", delta)
    nb_days = delta.days + 1# with both included
    print("requesting %s days" %nb_days)

    ret =  get_daily_solar_from_date_to_date(from_date = from_date, end_date = end_date)
    print(ret)
    if len(ret) != nb_days:
        print("FAILED. returned %d expected %d" %(len(ret), nb_days))
    else:
        print("OK. returned %d expected %d" %(len(ret), nb_days))



    
    #########################
    # system 
    #########################
 
    """
    status = system_summary(access_token)
    print("system summary:" , status)

    envoy_sn = system_devices(access_token)
    print("system devices" , envoy_sn)

    r = inverters_summary_for_envoy_sn(access_token, system_id, envoy_sn)
    print("envoy inverter summary by envoy:", r)
    """

    ###########################
    # energy lifetime (date to date) as a list
    ############################

    """
    days=10
    print("\nget (full day) daily energy list for last %d days, till yesterday" %days)

    # use date to get only complete days
    # yesterday
    start_date = datetime.date.today()- datetime.timedelta(days)
    end_date = datetime.date.today() - datetime.timedelta(1)

    # function expect datetime (to convert to epoch)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)
    
    production_list = get_daily_energy_list(access_token, start_date, end_date)
    
    if production_list == []:
        print("getting energy list failed")
    else:
        assert len(production_list) == days
        print("production list:", production_list)
    """

    ########################
    # solar today
    # wrap up for production_meter_from_date 
    # no start, so epochs and 0h, granularity = days
    ########################
    #wh, last_interval_stamp = get_energy_today()
    #print("get solar today", wh, last_interval_stamp)


    ###################
    # solar from micros, from meters
    # start_date = yesterday
    ###################

    # yesterday
    start_date = datetime.date.today() - datetime.timedelta(1)

    wh, last_interval_stamp = production_meter_from_date(access_token, start_date=start_date)
    print("%d wh from METERS, start_date %s, last stamp %s" %(wh, start_date, last_interval_stamp))

    wh, last_interval_stamp = production_micro_from_date(access_token, start_date= start_date) 
    print("%d wh from MICROS, start_date %s, last stamp %s" %(wh, start_date, last_interval_stamp))
  
    # NOTE: very similar reading from meter and micros

    """
    YESTERDAY 
    
    telemetry for all the production MICROS of a system. start_date 2023-08-30, granularity (ie span/duration) day
        12 devices reporting
        granularity: day
        160 intervals
        1st reported interval 2023-08-30 06:55:00
        last reported interval 2023-08-30 20:10:00
        300 sec between intervals
        start_date 2023-08-30T00:00:00+02:00
        end_date 2023-08-31T00:00:00+02:00
        full day
        meta: last_energy 2023-08-31 11:49:08    time of running
        meta: last_report 2023-08-31 11:59:12
        meta: operational 2021-03-11 15:45:12
        wh  21926
        21926 wh production from MICROS, 2023-08-30 20:10:00 last stamp

        telemetry for all the production METERS of a system. start_date 2023-08-30, granularity (ie span/duration) day
        96 intervals
        900 sec between intervals
        first interval: 2023-08-30 00:15:00
        last interval: 2023-08-31 00:00:00
        start_date 2023-08-30T00:00:00+02:00
        end_date 2023-08-31T00:00:00+02:00
        full day
        wh  22099
        22099 wh production from METERS, 2023-08-31 00:00:00 last stamp
    """


    ###################
    # rgm
    ###################

    days = 7
    # yesterday at this time , midnite boundary
    start_date = datetime.datetime.now()- datetime.timedelta(days)
    end_date = datetime.datetime.now()-datetime.timedelta(1)

    # use date to get only complete days
    # yesterday
    days=1
    start_date = datetime.date.today()- datetime.timedelta(days)
    end_date = datetime.date.today()

    # function expect datetime (to convert to epoch)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
    end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)

    print("\nget revenue grade meter statistics for last %d days, till yesterday" %days)
    wh_del = envoy_rgm_stats(access_token, start_date, end_date)

    print("revenue grade meter: ", wh_del) # seem valuye is the same as meter

    """
    get revenue grade meter statistics for last 1 days, till yesterday
    getting rgm stats from: 2023-08-30 00:00:00 to 2023-08-31 00:00:00
    intervals:
    96 intervals
    fist interval 2023-08-30 00:15:00
    2nd interval 2023-08-30 00:30:00
    900 sec between intervals
    last interval 2023-08-31 00:00:00
    wh_del 22099.0
    meter intervals:
    288 meter intervals
    fist interval 2023-08-30 00:15:00
    2nd interval 2023-08-30 00:30:00
    900 sec between intervals
    last interval 2023-08-31 00:00:00
    wh_del 22099.0
    last energy: 2023-08-31 13:09:52
    last report: 2023-08-31 13:14:12
    operational: 2021-03-11 15:45:12
    revenue grade meter:  22099 22099.0
    """



    

    
"""
envoy_inverters_summary_by_envoy

[{'signal_strength': 5, 'micro_inverters': [{'id': 44697200, 'serial_number': '122011069614', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 36, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00135-r01-v04.27.09', 'envoy_serial_number': '121945124013', 'energy': {'value': 1110772, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:05-07:00'}, {'id': 44697201, 'serial_number': '122011068460', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 36, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00135-r01-v04.27.09', 'envoy_serial_number': '121945124013', 'energy': {'value': 1106257, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:05-07:00'}, {'id': 51977278, 'serial_number': '122127029586', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 44, 'units': 
'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00135-r01-v04.27.10', 'envoy_serial_number': '121945124013', 'energy': {'value': 870751, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:16-07:00'}, {'id': 51977306, 'serial_number': '122127036949', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 44, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00135-r01-v04.27.10', 'envoy_serial_number': '121945124013', 'energy': {'value': 855453, 
'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:17-07:00'}, {'id': 44697202, 'serial_number': '122015041609', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 36, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.28.07', 'param_table': '540-00135-r01-v04.28.03', 'envoy_serial_number': '121945124013', 'energy': {'value': 1138450, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:07-07:00'}, {'id': 46694103, 'serial_number': '122015038254', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 42, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.28.07', 'param_table': '540-00135-r01-v04.28.03', 'envoy_serial_number': '121945124013', 'energy': {'value': 1065293, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:10-07:00'}, {'id': 46694104, 'serial_number': '122015038277', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 43, 'units': 'W', 'precision': 0}, 
'proc_load': '520-00082-r01-v04.28.07', 'param_table': '540-00135-r01-v04.28.03', 'envoy_serial_number': '121945124013', 'energy': {'value': 1069915, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:11-07:00'}, {'id': 46694105, 'serial_number': '122015039301', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 44, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.28.07', 'param_table': '540-00135-r01-v04.28.03', 'envoy_serial_number': '121945124013', 'energy': {'value': 1077755, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:12-07:00'}, {'id': 46694106, 'serial_number': '122015038253', 'model': 'IQ7+', 'part_number': '800-00631-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 43, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.28.07', 'param_table': '540-00135-r01-v04.28.03', 'envoy_serial_number': '121945124013', 'energy': {'value': 1060935, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:15-07:00'}, {'id': 44697199, 'serial_number': '122010105472', 'model': 'IQ7+', 'part_number': '800-01103-r02', 'sku': 'IQ7PLUS-72-2-FR', 'status': 'normal', 'power_produced': {'value': 35, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00146-r01-v04.27.09', 'envoy_serial_number': '121945124013', 'energy': {'value': 1125185, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:01-07:00'}, {'id': 57594904, 'serial_number': '122134058207', 'model': 'IQ7A', 'part_number': '800-01138-r02', 'sku': 'IQ7A-72-2-INT', 'status': 'normal', 'power_produced': {'value': 45, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00169-r01-v04.27.09', 'envoy_serial_number': '121945124013', 'energy': {'value': 666736, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:21-07:00'}, {'id': 57594906, 'serial_number': '122135009406', 'model': 'IQ7A', 'part_number': '800-01138-r02', 'sku': 'IQ7A-72-2-INT', 'status': 'normal', 'power_produced': {'value': 46, 'units': 'W', 'precision': 0}, 'proc_load': '520-00082-r01-v04.27.04', 'param_table': '540-00169-r01-v04.27.09', 'envoy_serial_number': '121945124013', 'energy': {'value': 651659, 'units': 'Wh', 'precision': 0}, 'grid_profile': 'EN 50549-1:2019 VFR2019 France', 'last_report_date': '2023-08-27T03:03:22-07:00'}]}]

"""

# site level production
"""
# /telemetry/production_micro telemetry for all the production micros of a system

{'end_at': 1693136012,  OR 'end_date': '2023-08-31T00:00:00+02:00',
 'granularity': 'day',
 'intervals': [{'devices_reporting': 12,
                'end_at': 1693112700,
                'enwh': 0,
                'powr': 1},
               {'devices_reporting': 12,
                'end_at': 1693113000,
                'enwh': 0,
                'powr': 2},
               {'devices_reporting': 12,
                'end_at': 1693113300,
                'enwh': 0,
                'powr': 2},
               {'devices_reporting': 12,
                'end_at': 1693113600,
                'enwh': 0,
                'powr': 5},
               {'devices_reporting': 11,
                'end_at': 1693113900,
                'enwh': 1,
                'powr': 7},
               {'devices_reporting': 11,
                'end_at': 1693114200,
                'enwh': 1,
                'powr': 7},
               {'devices_reporting': 12,
                'end_at': 1693114500,
                'enwh': 1,
                'powr': 6},
               {'devices_reporting': 12,
                'end_at': 1693114800,
                'enwh': 0,
                'powr': 1},
               {'devices_reporting': 12,
                'end_at': 1693115100,
                'enwh': 0,
                'powr': 1},
               {'devices_reporting': 12,
                'end_at': 1693115400,
                'enwh': 0,
                'powr': 2},
               {'devices_reporting': 12,
                'end_at': 1693115700,
                'enwh': 1,
                'powr': 9},
               {'devices_reporting': 12,
                'end_at': 1693116000,
                'enwh': 2,
                'powr': 18},
               {'devices_reporting': 12,
                'end_at': 1693116300,
                'enwh': 2,
                'powr': 19},
               {'devices_reporting': 12,
                'end_at': 1693116600,
                'enwh': 2,
                'powr': 19},
               {'devices_reporting': 12,
                'end_at': 1693116900,
                'enwh': 2,
                'powr': 22},
               {'devices_reporting': 12,
                'end_at': 1693117200,
                'enwh': 2,
                'powr': 24},
               {'devices_reporting': 12,
                'end_at': 1693117500,
                'enwh': 2,
                'powr': 24},
               {'devices_reporting': 12,
                'end_at': 1693117800,
                'enwh': 2,
                'powr': 25},
               {'devices_reporting': 12,
                'end_at': 1693118100,
                'enwh': 2,
                'powr': 25},
               {'devices_reporting': 12,
                'end_at': 1693118400,
                'enwh': 2,
                'powr': 25},
               {'devices_reporting': 12,
                'end_at': 1693118700,
                'enwh': 3,
                'powr': 39},
               {'devices_reporting': 12,
                'end_at': 1693119000,
                'enwh': 4,
                'powr': 47},
               {'devices_reporting': 12,
                'end_at': 1693119300,
                'enwh': 4,
                'powr': 47},
               {'devices_reporting': 12,
                'end_at': 1693119600,
                'enwh': 7,
                'powr': 81},
               {'devices_reporting': 12,
                'end_at': 1693119900,
                'enwh': 8,
                'powr': 101},
               {'devices_reporting': 12,
                'end_at': 1693120200,
                'enwh': 8,
                'powr': 101},
               {'devices_reporting': 12,
                'end_at': 1693120500,
                'enwh': 10,
                'powr': 123},
               {'devices_reporting': 12,
                'end_at': 1693120800,
                'enwh': 11,
                'powr': 137},
               {'devices_reporting': 12,
                'end_at': 1693121100,
                'enwh': 11,
                'powr': 137},
               {'devices_reporting': 12,
                'end_at': 1693121400,
                'enwh': 14,
                'powr': 167},
               {'devices_reporting': 12,
                'end_at': 1693121700,
                'enwh': 16,
                'powr': 187},
               {'devices_reporting': 12,
                'end_at': 1693122000,
                'enwh': 16,
                'powr': 187},
               {'devices_reporting': 12,
                'end_at': 1693122300,
                'enwh': 18,
                'powr': 219},
               {'devices_reporting': 12,
                'end_at': 1693122600,
                'enwh': 20,
                'powr': 241},
               {'devices_reporting': 12,
                'end_at': 1693122900,
                'enwh': 20,
                'powr': 241},
               {'devices_reporting': 12,
                'end_at': 1693123200,
                'enwh': 17,
                'powr': 206},
               {'devices_reporting': 12,
                'end_at': 1693123500,
                'enwh': 15,
                'powr': 183},
               {'devices_reporting': 12,
                'end_at': 1693123800,
                'enwh': 15,
                'powr': 183},
               {'devices_reporting': 12,
                'end_at': 1693124100,
                'enwh': 14,
                'powr': 171},
               {'devices_reporting': 12,
                'end_at': 1693124400,
                'enwh': 14,
                'powr': 162},
               {'devices_reporting': 12,
                'end_at': 1693124700,
                'enwh': 14,
                'powr': 162},
               {'devices_reporting': 12,
                'end_at': 1693125000,
                'enwh': 21,
                'powr': 254},
               {'devices_reporting': 12,
                'end_at': 1693125300,
                'enwh': 26,
                'powr': 317},
               {'devices_reporting': 12,
                'end_at': 1693125600,
                'enwh': 26,
                'powr': 317},
               {'devices_reporting': 12,
                'end_at': 1693125900,
                'enwh': 35,
                'powr': 415},
               {'devices_reporting': 12,
                'end_at': 1693126200,
                'enwh': 40,
                'powr': 485},
               {'devices_reporting': 12,
                'end_at': 1693126500,
                'enwh': 40,
                'powr': 485},
               {'devices_reporting': 12,
                'end_at': 1693126800,
                'enwh': 51,
                'powr': 617},
               {'devices_reporting': 12,
                'end_at': 1693127100,
                'enwh': 60,
                'powr': 715},
               {'devices_reporting': 12,
                'end_at': 1693127400,
                'enwh': 60,
                'powr': 715},
               {'devices_reporting': 12,
                'end_at': 1693127700,
                'enwh': 74,
                'powr': 892},
               {'devices_reporting': 12,
                'end_at': 1693128000,
                'enwh': 85,
                'powr': 1025},
               {'devices_reporting': 12,
                'end_at': 1693128300,
                'enwh': 85,
                'powr': 1025},
               {'devices_reporting': 12,
                'end_at': 1693128600,
                'enwh': 78,
                'powr': 933},
               {'devices_reporting': 12,
                'end_at': 1693128900,
                'enwh': 95,
                'powr': 1143},
               {'devices_reporting': 12,
                'end_at': 1693129200,
                'enwh': 103,
                'powr': 1238},
               {'devices_reporting': 12,
                'end_at': 1693129500,
                'enwh': 103,
                'powr': 1238},
               {'devices_reporting': 12,
                'end_at': 1693129800,
                'enwh': 109,
                'powr': 1308},
               {'devices_reporting': 12,
                'end_at': 1693130100,
                'enwh': 119,
                'powr': 1430},
               {'devices_reporting': 12,
                'end_at': 1693130400,
                'enwh': 119,
                'powr': 1430},
               {'devices_reporting': 12,
                'end_at': 1693130700,
                'enwh': 130,
                'powr': 1562},
               {'devices_reporting': 12,
                'end_at': 1693131000,
                'enwh': 150,
                'powr': 1801},
               {'devices_reporting': 12,
                'end_at': 1693131300,
                'enwh': 150,
                'powr': 1801},
               {'devices_reporting': 12,
                'end_at': 1693131600,
                'enwh': 142,
                'powr': 1705},
               {'devices_reporting': 12,
                'end_at': 1693131900,
                'enwh': 128,
                'powr': 1532},
               {'devices_reporting': 12,
                'end_at': 1693132200,
                'enwh': 128,
                'powr': 1532},
               {'devices_reporting': 12,
                'end_at': 1693132500,
                'enwh': 117,
                'powr': 1409},
               {'devices_reporting': 12,
                'end_at': 1693132800,
                'enwh': 98,
                'powr': 1177},
               {'devices_reporting': 12,
                'end_at': 1693133100,
                'enwh': 98,
                'powr': 1177},
               {'devices_reporting': 12,
                'end_at': 1693133400,
                'enwh': 79,
                'powr': 945},
               {'devices_reporting': 12,
                'end_at': 1693133700,
                'enwh': 42,
                'powr': 504},
               {'devices_reporting': 12,
                'end_at': 1693134000,
                'enwh': 42,
                'powr': 504},
               {'devices_reporting': 12,
                'end_at': 1693134300,
                'enwh': 48,
                'powr': 581},
               {'devices_reporting': 12,
                'end_at': 1693134600,
                'enwh': 61,
                'powr': 733},
               {'devices_reporting': 12,
                'end_at': 1693134900,
                'enwh': 61,
                'powr': 733},
               {'devices_reporting': 12,
                'end_at': 1693135200,
                'enwh': 78,
                'powr': 931},
               {'devices_reporting': 12,
                'end_at': 1693135500,
                'enwh': 112,
                'powr': 1338},
               {'devices_reporting': 12,
                'end_at': 1693135800,
                'enwh': 112,
                'powr': 1338}],
 'items': 'intervals',
 'meta': {'last_energy_at': 1693136012,
          'last_report_at': 1693136683,
          'operational_at': 1615473912,
          'status': 'normal'},
 'start_at': 1693087200, OR 'start_date': '2023-08-30T00:00:00+02:00',
 'system_id': 2160737,
 'total_devices': 12}
"""


"""
# /telemetry/production_meter telemetry for all the production meters of a system.

{'end_at': 1693473300,
 'granularity': 'day',
 'intervals': [{'devices_reporting': 1, 'end_at': 1693433700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693434600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693435500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693436400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693437300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693438200, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693439100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693440000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693440900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693441800, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693442700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693443600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693444500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693445400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693446300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693447200, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693448100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693449000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693449900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693450800, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693451700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693452600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693453500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693454400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693455300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693456200, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693457100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693458000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693458900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693459800, 'wh_del': 2},
               {'devices_reporting': 1, 'end_at': 1693460700, 'wh_del': 13},
               {'devices_reporting': 1, 'end_at': 1693461600, 'wh_del': 21},
               {'devices_reporting': 1, 'end_at': 1693462500, 'wh_del': 26},
               {'devices_reporting': 1, 'end_at': 1693463400, 'wh_del': 39},
               {'devices_reporting': 1, 'end_at': 1693464300, 'wh_del': 38},
               {'devices_reporting': 1, 'end_at': 1693465200, 'wh_del': 48},
               {'devices_reporting': 1, 'end_at': 1693466100, 'wh_del': 58},
               {'devices_reporting': 1, 'end_at': 1693467000, 'wh_del': 108},
               {'devices_reporting': 1, 'end_at': 1693467900, 'wh_del': 178},
               {'devices_reporting': 1, 'end_at': 1693468800, 'wh_del': 355},
               {'devices_reporting': 1, 'end_at': 1693469700, 'wh_del': 534},
               {'devices_reporting': 1, 'end_at': 1693470600, 'wh_del': 620},
               {'devices_reporting': 1, 'end_at': 1693471500, 'wh_del': 653},
               {'devices_reporting': 1, 'end_at': 1693472400, 'wh_del': 691},
               {'devices_reporting': 1, 'end_at': 1693473300, 'wh_del': 741}],
 'items': 'intervals',
 'start_at': 1693432800,
 'system_id': 2160737,
 'total_devices': 1}



"""


"""

rgm stats 

{'intervals': [{'devices_reporting': 1, 'end_at': 1693088100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693089000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693089900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693090800, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693091700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693092600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693093500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693094400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693095300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693096200, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693097100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693098000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693098900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693099800, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693100700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693101600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693102500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693103400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693104300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693105200, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693106100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693107000, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693107900, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693108800, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693109700, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693110600, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693111500, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693112400, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693113300, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693114200, 'wh_del': 6},
               {'devices_reporting': 1, 'end_at': 1693115100, 'wh_del': 0},
               {'devices_reporting': 1, 'end_at': 1693116000, 'wh_del': 6},
               {'devices_reporting': 1, 'end_at': 1693116900, 'wh_del': 7},
               {'devices_reporting': 1, 'end_at': 1693117800, 'wh_del': 9},
               {'devices_reporting': 1, 'end_at': 1693118700, 'wh_del': 10},
               {'devices_reporting': 1, 'end_at': 1693119600, 'wh_del': 17},
               {'devices_reporting': 1, 'end_at': 1693120500, 'wh_del': 31},
               {'devices_reporting': 1, 'end_at': 1693121400, 'wh_del': 40},
               {'devices_reporting': 1, 'end_at': 1693122300, 'wh_del': 51},
               {'devices_reporting': 1, 'end_at': 1693123200, 'wh_del': 65},
               {'devices_reporting': 1, 'end_at': 1693124100, 'wh_del': 43},
               {'devices_reporting': 1, 'end_at': 1693125000, 'wh_del': 51},
               {'devices_reporting': 1, 'end_at': 1693125900, 'wh_del': 89},
               {'devices_reporting': 1, 'end_at': 1693126800, 'wh_del': 142},
               {'devices_reporting': 1, 'end_at': 1693127700, 'wh_del': 192},
               {'devices_reporting': 1, 'end_at': 1693128600, 'wh_del': 258},
               {'devices_reporting': 1, 'end_at': 1693129500, 'wh_del': 305},
               {'devices_reporting': 1, 'end_at': 1693130400, 'wh_del': 347},
               {'devices_reporting': 1, 'end_at': 1693131300, 'wh_del': 444},
               {'devices_reporting': 1, 'end_at': 1693132200, 'wh_del': 405},
               {'devices_reporting': 1, 'end_at': 1693133100, 'wh_del': 348},
               {'devices_reporting': 1, 'end_at': 1693134000, 'wh_del': 139},
               {'devices_reporting': 1, 'end_at': 1693134900, 'wh_del': 156},
               {'devices_reporting': 1, 'end_at': 1693135800, 'wh_del': 351},
               {'devices_reporting': 1, 'end_at': 1693136700, 'wh_del': 185},
               {'devices_reporting': 1, 'end_at': 1693137600, 'wh_del': 225},
               {'devices_reporting': 1, 'end_at': 1693138500, 'wh_del': 110},
               {'devices_reporting': 1, 'end_at': 1693139400, 'wh_del': 90},
               {'devices_reporting': 1, 'end_at': 1693140300, 'wh_del': 112},
               {'devices_reporting': 1, 'end_at': 1693141200, 'wh_del': 91},
               {'devices_reporting': 1, 'end_at': 1693142100, 'wh_del': 159}],
 'meta': {'last_energy_at': 1693142609,
          'last_report_at': 1693142986,
          'operational_at': 1615473912,
          'status': 'normal'},
 'meter_intervals': [{'envoy_serial_number': '121945124013',
                      'intervals': [{'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693088100,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693089000,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693089900,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693090800,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693091700,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693092600,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693093500,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693094400,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693095300,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693096200,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693097100,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693098000,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693098900,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693099800,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693100700,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693101600,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693102500,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693103400,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693104300,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693105200,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693106100,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693107000,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693107900,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693108800,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693109700,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693110600,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 0,
                                     'end_at': 1693111500,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': -1,
                                     'end_at': 1693112400,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 17,
                                     'end_at': 1693113300,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 3,
                                     'end_at': 1693114200,
                                     'wh_del': 6.0},
                                    {'channel': 1,
                                     'curr_w': 8,
                                     'end_at': 1693115100,
                                     'wh_del': 0.0},
                                    {'channel': 1,
                                     'curr_w': 34,
                                     'end_at': 1693116000,
                                     'wh_del': 6.0},
                                    {'channel': 1,
                                     'curr_w': 35,
                                     'end_at': 1693116900,
                                     'wh_del': 7.0},
                                    {'channel': 1,
                                     'curr_w': 36,
                                     'end_at': 1693117800,
                                     'wh_del': 9.0},
                                    {'channel': 1,
                                     'curr_w': 49,
                                     'end_at': 1693118700,
                                     'wh_del': 10.0},
                                    {'channel': 1,
                                     'curr_w': 96,
                                     'end_at': 1693119600,
                                     'wh_del': 17.0},
                                    {'channel': 1,
                                     'curr_w': 147,
                                     'end_at': 1693120500,
                                     'wh_del': 31.0},
                                    {'channel': 1,
                                     'curr_w': 193,
                                     'end_at': 1693121400,
                                     'wh_del': 40.0},
                                    {'channel': 1,
                                     'curr_w': 215,
                                     'end_at': 1693122300,
                                     'wh_del': 51.0},
                                    {'channel': 1,
                                     'curr_w': 230,
                                     'end_at': 1693123200,
                                     'wh_del': 65.0},
                                    {'channel': 1,
                                     'curr_w': 118,
                                     'end_at': 1693124100,
                                     'wh_del': 43.0},
                                    {'channel': 1,
                                     'curr_w': 270,
                                     'end_at': 1693125000,
                                     'wh_del': 51.0},
                                    {'channel': 1,
                                     'curr_w': 391,
                                     'end_at': 1693125900,
                                     'wh_del': 89.0},
                                    {'channel': 1,
                                     'curr_w': 741,
                                     'end_at': 1693126800,
                                     'wh_del': 142.0},
                                    {'channel': 1,
                                     'curr_w': 978,
                                     'end_at': 1693127700,
                                     'wh_del': 192.0},
                                    {'channel': 1,
                                     'curr_w': 910,
                                     'end_at': 1693128600,
                                     'wh_del': 258.0},
                                    {'channel': 1,
                                     'curr_w': 1393,
                                     'end_at': 1693129500,
                                     'wh_del': 305.0},
                                    {'channel': 1,
                                     'curr_w': 1453,
                                     'end_at': 1693130400,
                                     'wh_del': 347.0},
                                    {'channel': 1,
                                     'curr_w': 1823,
                                     'end_at': 1693131300,
                                     'wh_del': 444.0},
                                    {'channel': 1,
                                     'curr_w': 1439,
                                     'end_at': 1693132200,
                                     'wh_del': 405.0},
                                    {'channel': 1,
                                     'curr_w': 877,
                                     'end_at': 1693133100,
                                     'wh_del': 348.0},
                                    {'channel': 1,
                                     'curr_w': 491,
                                     'end_at': 1693134000,
                                     'wh_del': 139.0},
                                    {'channel': 1,
                                     'curr_w': 1001,
                                     'end_at': 1693134900,
                                     'wh_del': 156.0},
                                    {'channel': 1,
                                     'curr_w': 1067,
                                     'end_at': 1693135800,
                                     'wh_del': 351.0},
                                    {'channel': 1,
                                     'curr_w': 877,
                                     'end_at': 1693136700,
                                     'wh_del': 185.0},
                                    {'channel': 1,
                                     'curr_w': 519,
                                     'end_at': 1693137600,
                                     'wh_del': 225.0},
                                    {'channel': 1,
                                     'curr_w': 432,
                                     'end_at': 1693138500,
                                     'wh_del': 110.0},
                                    {'channel': 1,
                                     'curr_w': 530,
                                     'end_at': 1693139400,
                                     'wh_del': 90.0},
                                    {'channel': 1,
                                     'curr_w': 445,
                                     'end_at': 1693140300,
                                     'wh_del': 112.0},
                                    {'channel': 1,
                                     'curr_w': 371,
                                     'end_at': 1693141200,
                                     'wh_del': 91.0},
                                    {'channel': 1,
                                     'curr_w': 779,
                                     'end_at': 1693142100,
                                     'wh_del': 159.0},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693088100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693089000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693089900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693090800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693091700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693092600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693093500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693094400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693095300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693096200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693097100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693098000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693098900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693099800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693100700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693101600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693102500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693103400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693104300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693105200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693106100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693107000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693107900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693108800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693109700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693110600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693111500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693112400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693113300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693114200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693115100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693116000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693116900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693117800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693118700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693119600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693120500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693121400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693122300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693123200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693124100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693125000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693125900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693126800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693127700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693128600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693129500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693130400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693131300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693132200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693133100,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693134000,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693134900,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693135800,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693136700,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693137600,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693138500,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693139400,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693140300,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693141200,
                                     'wh_del': None},
                                    {'channel': 2,
                                     'curr_w': None,
                                     'end_at': 1693142100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693088100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693089000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693089900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693090800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693091700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693092600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693093500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693094400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693095300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693096200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693097100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693098000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693098900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693099800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693100700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693101600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693102500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693103400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693104300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693105200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693106100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693107000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693107900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693108800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693109700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693110600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693111500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693112400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693113300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693114200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693115100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693116000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693116900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693117800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693118700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693119600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693120500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693121400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693122300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693123200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693124100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693125000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693125900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693126800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693127700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693128600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693129500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693130400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693131300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693132200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693133100,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693134000,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693134900,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693135800,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693136700,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693137600,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693138500,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693139400,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693140300,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693141200,
                                     'wh_del': None},
                                    {'channel': 3,
                                     'curr_w': None,
                                     'end_at': 1693142100,
                                     'wh_del': None}],
                      'meter_serial_number': '121945124013EIM1'}],
 'system_id': 2160737,
 'total_devices': 1}
"""

"""
energy lifetime
{'meta': {'last_energy_at': 1693484702,
          'last_report_at': 1693484950,
          'operational_at': 1615473912,
          'status': 'normal'},
 'production': [21926],
 'start_date': '2023-08-30',
 'system_id': 2160737}
"""