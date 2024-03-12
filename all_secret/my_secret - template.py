#!/usr/bin/env python3

############
# my energy
############
myenergi = {
"hub_serial":"",
"hub_pwd":"",
}

#############
# enphase
#############

#https://developer-v4.enphase.com/admin/applications/

# to generate new auth code (in case refresh token expided).
# NOTE: must include redirect_uri
# https://api.enphaseenergy.com/oauth/authorize?response_type=code&client_id=&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri
# record new auth code in secret.py

# id = client_id:client_secret. used to get access token
# api key can be refreshed, used to do API call


enphase = {
"system_id" : "", 
"solar_ev_client" : ':',
"solar_ev_api_key" : '',
"auth_code" :  ""
}

# to generate new auth code (in case refresh token expided)
# # https://api.enphaseenergy.com/oauth/authorize?response_type=code&client_id&redirect_uri=https://api.enphaseenergy.com/oauth/redirect_uri
# also delete enphase_token.json

############
# new blynk
############
# template/id and device name in template screen
# auth token in device screen
# guess that only the auth token is needed by the python API

blynk = {
"BLYNK_TEMPLATE_ID" : "", 
"BLYNK_DEVICE_NAME" : "solar2ev" ,
"BLYNK_AUTH_TOKEN"  : ""
}


pushover = {
"pushover_token" : "",
"pushover_user" : ""
}


########################
# GMAIL
########################
# application password
# https://myaccount.google.com/u/1/apppasswords

email_passwd = ""

dest_email = ""
sender_email = ""

######################
# shelly cloud API
#####################

# shelly smart control app,  setting, authorization cloud key
shelly_cloud_auth_key = ""
shelly_cloud_server_uri = "https://shelly-33-eu.shelly.cloud/"

# device id in Shelly Cloud app in Device->Settings->Device Info

solar_1 = ""
solar_2 = ""
