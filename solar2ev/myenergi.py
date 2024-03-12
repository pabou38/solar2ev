#!/usr/bin/env python3

##########################
# myenergi API
##########################

import requests
import pprint
from requests.auth import HTTPDigestAuth
import sys

import my_secret


hub_serial = my_secret.myenergi['hub_serial']
hub_pwd = my_secret.myenergi['hub_pwd']

assert hub_serial != None
assert hub_pwd != None

director_url = "https://director.myenergi.net"

pst_dict = {"A": "EV Disconnected", "B1":"EV Connected", "B2":"Waiting for EV", "C1":"EV Ready to Charge", "C2":"Charging", "F":"Fault"}
zmo_dict = {1:"Fast", 2:"Eco", 3:"Eco+", 4:"Stopped"}  # 10 boost, 11 smartboost 
z_mode = {"fast":"1", "eco":"2", "eco+":"3", "stop":"4"}


def get_server_url():
    # Rather then basing the target server on the last digit of the hub serial number, 
    # the preferred approach is to make an API call to the "Director" - https://director.myenergi.net

    response = requests.get(director_url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
    #print(response)
    #print(response.headers)
    server_url = response.headers['X_MYENERGI-asn']
    print('actual myenergi server: ' , server_url)

    return(server_url)

    #{'Date': 'Wed, 02 Nov 2022 08:42:52 GMT', 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': '12', 'Connection': 'keep-alive', 'Access-Control-Expose-Headers': 'Server-Timing', 'Server-Timing': 'traceparent;desc="00-60a6a306ec536e15a6b0305451866836-658c6ee686291627-01"', 'X-Content-Type-Options': 'nosniff', 'X-DNS-Prefetch-Control': 'off', 'X-Download-Options': 'noopen', 'X-Frame-Options': 'SAMEORIGIN', 'X-XSS-Protection': '1; mode=block', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains', 'x_myenergi-asn': 's18.myenergi.net', 'Access-Control-Allow-Credentials': 'true', 'Access-Control-Allow-Headers': 'Origin, Content-Type, Accept, Cookie', 'Access-Control-Allow-Origin': 'https://admin-ui.s18.myenergi.net', 'ETag': 'W/"c-00hq6RNueFa8QiEjhep5cJRHWAI"'}
    #s18.myenergi.net



def get_zappy_status(server_url):
    url = "https://" +server_url+ '/cgi-jstatus-Z'
    print(url)
    # -E eddy, -Z zappy,  returns ['zappi'] [0]  
    # -* all available devices, and return array [0]['eddi'][0] 

    try:
        response = requests.get(url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
        #print(response)  # stream object
        if response.status_code == 200:
        #print(response.headers)
        #print(response.content)
        #print(response.json())

            #pprint.pprint(response.json())
            # [{'eddi': [...]}, {'zappi': [...]}, {'harvi': [...]}, {'asn': 's18.myenergi.net', 'fwv': '3401S3.077'}]
            pass
        
    except Exception as e:
        print(str(e))

    zappy = response.json()['zappi'][0]

    zappy_sno = zappy['sno']
    car_status = zappy['pst']
    zappy_status = zappy['zmo']

    try:
        gen =zappy['gen']
        grid = zappy['grd']
    except Exception as e: #
        pass
        #print('cannot read zappy ', str(e)) # field does no exits if zero 

    mgl = zappy['mgl']

    return(car_status, zappy_status, zappy_sno)


"""
Boost 5KWh /cgi-zappi-mode-Z10077777-0-10-5-0000 where 0 is Boost - 10 is Boost Mode - 5 is the KWh to add
Smart Boost 5KWh - complete by 2pm /cgi-zappi-mode-Z10077777-0-11-5-1400 where 0 is Boost - 11 is Smart Boost Mode - 5 is the KWh to add, 1400 is the time the boost should complete.
Stop Boost /cgi-zappi-mode-Z10077777-0-2-0-0000
"""

def set_zappy_boost_time(server_url, zappy_sno, slot=11, start_h=23, start_m=0, duration_h=1, duration_m=0):
    print('set zappy boost')

    # cgi-boost-time-Z???-{slot}-{bsh}-{bdh}-{bdd}
    # Slot is one of 11,12,13,14
    # Start time is in 24 hour clock, 15 minute intervals.

    # Duration is hoursminutes and is less than 10 hours. 
    # res = self._load(suffix='cgi-boost-time-Z{}-{}-{:02}{:02}-{}{:02}-{}'.format(zid,slot,bsh,bsm,bdh,bdm,bdd))

    s = "-%s-%02d%02d-%02d%02d-00000001" %(str(slot), start_h, start_m, duration_h, duration_m)

    url = "https://" +server_url+ '/cgi-boost-time-' + 'Z' + str(zappy_sno) + s
    # https://s18.myenergi.net/cgi-boost-time-Z16076404-1-0100-0101-00000001

    print(url) 

    response = requests.get(url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
    print(response)

    if response.status_code == 200:
        # {'status': '-14', 'statustext': ''}
        pprint.pprint(response.json())
        return(True)

    else:

        print('error http', response.status_code) # post generates error
        return (False)


def get_zappy_boost(server_url, zappy_sno):
    print('get zappy boost time')
    url = "https://" +server_url+ '/cgi-boost-time-' + 'Z' + str(zappy_sno)
    print(url)
    response = requests.get(url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
    if response.status_code == 200:
        # {'status': 0, 'statustext': '', 'asn': 's18.myenergi.net'}
        #pprint.pprint(response.json())
        return(response.json()["boost_times"])
        """
        {'boost_times': [ ]}
        
        {'bdd': '00000001', //boost days of week Monday through Sunday
                    'bdh': 3, //boost duration hour
                    'bdm': 0, //boost duration minute
                    'bsh': 0, //boost start hour
                    'bsm': 0, 
                    'slt': 11}, //Slot
        """
    else:
        print('error http', response.status_code) # post generates error
        return(None)

def set_zappy_mode(mode = 'stop'):
    
    print('change zappy mode', mode)

    # https://github.com/ashleypittman/mec/blob/master/mec/zp.py
    # /cgi-zappi-mode-Znnnnnnnn-4-0-0-0000
    #  or /cgi-zappi-mode-Znnnnnnnn-4-0
    #  data = self._load(suffix='cgi-zappi-mode-Z{}-{}-0'.format(zid, mode))
    # zappi_set_mode: "curl --digest -u {{myenergi_serial}}:{{myenergi_password}} -H 'accept: application/json' -H 'content-type: application/json' --compressed 
    # 'https://s7.myenergi.net/cgi-zappi-mode-Z{{zappi_serial}}-{{zappi_mode}}-0-0-0000'"

    url = "https://" +server_url+ '/cgi-zappi-mode-' + 'Z' + str(zappy_sno)  +'-' + z_mode[mode] +'-0-0-0000'
    print(url)
    response = requests.get(url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
    if response.status_code == 200:
        # {'status': 0, 'statustext': '', 'asn': 's18.myenergi.net'}
        print(response.json())
    else:
        print('error http', response.status_code) # post generates error


    print('read back zappy mode')
    #url = "https://" +server_url+ '/cgi-jstatus-*'
    #url = "https://" +server_url + '/cgi-jstatus-Z' + str(zappy_sno)
    url = "https://" +server_url+ '/cgi-jstatus-Z'

    response = requests.get(url, auth=HTTPDigestAuth(hub_serial, hub_pwd))
    if response.status_code == 200:
        #pprint.pprint(response.json())
        mode = response.json()['zappi'][0]['zmo']
        print(zmo_dict[mode])


def set_green_level():
    """
    # set minimum green level
    url = "https://" +server_url+ '/cgi-set-min-green-' + 'Z' + str(zappy_sno)  +'-60'
    print(url)
    #https://s18.myenergi.net/cgi-set-min-green-Z16076404-60
    #{'mgl': 60}
    """

################
# set up car charger for nightly charge
################

def set_up_car_charger(hours, slot=11, start=(1,0), duration=(1,0)):

    # default starts a 1am
    server_url = get_server_url()
    (car_status, zappy_status, zappy_sno) = get_zappy_status(server_url)

    print('car status: ', pst_dict[car_status])
    print('zappy mode: ', zmo_dict[zappy_status])

    print('set zappy boost: duration %dh' %hours)
    ret = set_zappy_boost_time(server_url, zappy_sno, slot=slot, start_h=start[0], start_m=start[1], duration_h=hours, duration_m=duration[1])
    print("set zappy boost returned %s" %ret)

    boost = get_zappy_boost(server_url, zappy_sno) # list of 4 dict, one per slot
    print('read zappy boost:')
    for b in boost:
        print(b)
    

if __name__ == "__main__":
    server_url = get_server_url()
    (car_status, zappy_status, zappy_sno) = get_zappy_status(server_url)

    print('car status: ', pst_dict[car_status])
    print('zappy mode: ', zmo_dict[zappy_status])

    print('set zappi boost')
    set_zappy_boost_time(server_url, zappy_sno, slot=1, start_h=1, start_m=0, duration_h=1, duration_m=1)

    boost = get_zappy_boost(server_url, zappy_sno) # list of 4 dict, one per slot
    for b in boost:
        print(b)

    

    # bsh , bsm boost start
    # bdh , bdm boost duration
    # bdd mask days



    """
    {'slt': 11, 'bsh': 23, 'bsm': 15, 'bdh': 5, 'bdm': 0, 'bdd': '00001010'}
    {'slt': 12, 'bsh': 0, 'bsm': 0, 'bdh': 0, 'bdm': 0, 'bdd': '00000000'}
    {'slt': 13, 'bsh': 0, 'bsm': 0, 'bdh': 0, 'bdm': 0, 'bdd': '00000000'}
    {'slt': 14, 'bsh': 0, 'bsm': 0, 'bdh': 0, 'bdm': 0, 'bdd': '00000000'}
    """




"""
[{'eddi': [{'bsm': 0, // Boost Mode - 1 if boosting
            'bst': 0,
            'che': 0.87, // total kWh tranferred this session (today)
            'cmt': 254, 
            'dat': '02-11-2022',
            'div': 610, //Diversion amount Watts
            'dst': 1,
            'ectp1': 610,
            'ectt1': 'Internal Load',
            'ectt2': 'None',
            'ectt3': 'None',
            'frq': 50.01,
            'fwv': '3200S3.051',
            'hno': 1, // Currently active heater (1/2)
            'hpri': 1,
            'ht1': 'Radiator',
            'ht2': 'None',
            'pha': 1, //phase number or number of phases?
            'pri': 2, //priority
            'r1a': 0,
            'r1b': 0,
            'r2a': 0,
            'r2b': 0,
            'rbc': 1,
            'rbt': 3585, // If boosting, the remaining boost time in of seconds
            'sno': 14466270,
            'sta': 3, //Status 1=Paused, 3=Diverting, 4=Boost, 5=Max Temp Reached, 6=Stopped
            'tim': '09:07:05',
            'tp1': 127, //temperature probe 1 (50 C)
            'tp2': 127,
            'tz': 3,
            'vol': 2268}]}, //Voltage out (divide by 10)
 {'zappi': [{'bsm': 0,
             'bss': 0,
             'bst': 0,
             'che': 2.93, //Charge added in KWh
             'cmt': 255, //Command Timer- counts 1 - 10 when command sent, then 254 - success, 253 - failure, 255 - never received any comamnds
             'dat': '02-11-2022',
             'div': 0, //Diversion amount Watts (does not appear if zero)
             'dst': 1,  // Use Daylight Savings Time
             'ectp2': -1, //Physical CT connection 1 value Watts
             'ectp4': 3,
             'ectt1': 'Internal Load', //CT 1 Name	
             'ectt2': 'None',
             'ectt3': 'None',
             'ectt4': 'None',
             'ectt5': 'None',
             'ectt6': 'None',
             'frq': 50.03,
             'fwv': '3560S3.147',
             'lck': 16, //Lock Status (4 bits : 1st digit - ? : 2nd digit 
             'mgl': 70, //Minimum Green Level
             'pha': 1,
             'pri': 1, //priority
             'pst': 'A', //Status A=EV Disconnected, B1=EV Connected, B2=Waiting for EV, C1=EV Ready to Charge, C2= Charging, F= Fault
             'pwm': 4200,
             'rac': 8,
             'sbh': 6, //Smart Boost Start Time Hour
             'sbk': 10, //Smart Boost KWh to add
             'sno': 16076404,
             'sta': 1, //Status  1=Paused 3=Diverting/Charging 5=Complete
             'tim': '08:15:07',
             'tz': 3,
             'vol': 2279, //Supply voltage
             'zmo': 3, //Zappi Mode - 1=Fast, 2=Eco, 3=Eco+, 4=Stopped
             'zs': 256,
             'zsh': 1}]},
			 
 {'harvi': [{'dat': '02-11-2022',
             'ect1p': 1,
             'ect2p': 1,
             'ect3p': 1,
             'ectp1': 144,
             'ectp2': 407,
             'ectt1': 'Grid',
             'ectt2': 'Generation',
             'ectt3': 'None',
             'fwv': '3170S0.000',
             'sno': 11324933,
             'tim': '07:59:26'}]},

 {'asn': 's18.myenergi.net', 'fwv': '3401S3.077'}]
"""
