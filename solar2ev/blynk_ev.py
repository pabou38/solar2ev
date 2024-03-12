#!/usr/bin/python

import sys
from time import sleep
import datetime
import logging
import _thread # low level

import config_features

p = "../all_secret"

sys.path.insert(1, p)

try:
    import my_secret # not in sys.path in running as main
    token = my_secret.blynk["BLYNK_AUTH_TOKEN"]
except Exception as e:
    print(str(e))
    sys.exit(1)


version = 1.1

s = "%s: v%0.1f" %(__name__, version)

# new lib
# https://github.com/vshymanskyy/blynk-library-python

logging.info(str(datetime.datetime.now())+ s)

######################
# normalize file path for linux (Pi) and Windows
######################

# root is DEEP (windows) or APP (linux)
#   Blynk/Blynk_client/ # used by python and micropython
#   PABOU/  # deep learning generic modules
#   my_modules/ # other generic
#   all_secret/my_secret.py  # passwd, monit
#   solar2*/
#   victron apps/
#   ...

# "linux" on PI

if sys.platform in ["win32"]:
    #sys.path.insert(1, '../../Blynk/Blynk_client')
    sys.path.insert(1, '../Blynk/Blynk_client') # now same as on Linux
else:
    sys.path.insert(1, '../Blynk/Blynk_client')

import my_blynk_new

import vpins


#######################
# postmortem
#######################

# sorted in same order as history. 1st entry is least recent, and on the rigth of the GUI screen
# rolling list of last n days.   yesterday to moins 6
# number of led, drive size of history json
# list below should be consistent with GUI 

# NOTE: this is not an absolute -1, -2. more last time postmortem was run, last time before last, etc ..
# if run everyday with cron/systemd this will become an absolute list
post_mortem_led = [vpins.v_moins6, vpins.v_moins5, vpins.v_moins4, vpins.v_moins3, vpins.v_moins2, vpins.v_yesterday]
post_mortem_led_label = ["day-6", "day-5", "day-4", "day-3", "day-2", "day-1"]

assert len(post_mortem_led_label) == len(post_mortem_led)

post_mortem_nb = len(post_mortem_led)


####################
# prediction
####################

# left, lowest prediction
# should be defined and fit in GUI 
prediction_led = [vpins.v_bin0, vpins.v_bin1, vpins.v_bin2, vpins.v_bin3]
assert len(prediction_led) == len(config_features.config["prod_bins_labels_str"])
# see below for associated colors

# global set in GUI call back, available in main

# charge mode
menu_charge_ev = -1

charge_0 = -1 # sliders, hours of charge per predicted bin
charge_1 = -1
charge_2 = -1
charge_3 = -1

confidence_threshold = -1

blynk_connected = False

max_time_to_connect = 60# sec

###########
# GUI color
# use color picker
###########

# prediction, postmortem, model info, unseen data
# color of what is written inside the display widget, not the color of the actual label of this display
color_text = "#808080" # grey

# drop down
color_menu = "#808080" 

# led color indicate strength
# 1st element is led on left/lower prediction
# kind of red, orange, yellow, green
# also use for charge slider
color_prediction_led = ["#eb3a34", "#eb9234", "#e5eb34", "#3deb34"]
assert len(color_prediction_led) == len(config_features.config["prod_bins_labels_str"])


# should not run create each time the module is imported
blynk_con = None

####################
# events, defined in console
####################

# to check match with console
all_events = ["prediction", "post_mortem", "system_error", "starting", "charging"]

#############################
# connect 
#############################

# define all callback (in the context of blynk_con)
# start .run() thread
# wait for connection
# sync
# various init
# return blynk connection or None

def create_and_connect_blynk() :
    global blynk_con
    global blynk_connected # set to true in connect call back

    print("%s: create blynk connection" %__name__) # Connecting to blynk.cloud:443..
    blynk_con = my_blynk_new.create_blynk(token)

    if blynk_con is None:  
        s = "%s: cannot connect to Blynk server" %__name__
        print(s)
        logging.error(s)
        return(None)

    # defines all cal backs in the context of blynk_con
    def_call_back()

    ### run blynk.run() in separate thread
    print("%s: create blynk.run() thread" %__name__)
    id1= _thread.start_new_thread(my_blynk_new.run_blynk, (blynk_con,))

    i = 0
    print("%s: waiting for blynk to connect" %__name__)

    while blynk_connected is False: # set to true in connect call back

        sleep(1)
        i = i + 1
        if i > max_time_to_connect:
            s = "%s: cannot connect to Blynk. exit" %__name__
            print(s)
            logging.error(s)
            return(None)
            #sys.exit(1) # let the caller decide

    s = "%s: blynk is connected" %__name__
    print(s)
    logging.info(s)

    ########################
    # blynk connected
    #######################

    # create event in template -> events
    print("%s: send event" %__name__)
    blynk_con.log_event("starting", "solar2ev starting") # use event code

    #########################
    #  SYNC all needed vpins
    #########################

    print("%s: sync blynk" %__name__)

    # inference tab
    # pull down program charger. set charge_ev 
    blynk_con.sync_virtual(vpins.v_charge_menu)

    # charger tab
    # slider, number of hour of charge, set charge_0, charge_1 ..

    for vp in [vpins.v_charge_0, vpins.v_charge_1, vpins.v_charge_2, vpins.v_charge_3]:
        blynk_con.sync_virtual(vp)

    # slider confidence
    blynk_con.sync_virtual(vpins.v_confidence)

    ##########################
    # set labels, colors ..
    ##########################

    # clear all prediction led
    for x in [vpins.v_bin0, vpins.v_bin1, vpins.v_bin2, vpins.v_bin3]:
        blynk_con.virtual_write(x, 0)

    # set label, color for charger slider
    # same order, color, labels as prediction led
    # NOTE: min , max for slider set in smartphone app
    for i, vp in enumerate([vpins.v_charge_0, vpins.v_charge_1, vpins.v_charge_2, vpins.v_charge_3]):
        blynk_color_and_label(vp, color_prediction_led[i], label= config_features.config["prod_bins_labels_str"] [i])

    # set up color and label label below sliders
    blynk_color_and_label(vpins.v_label_charge, color_text, "hours of overnigth charge")

    # set up color and label for drop down menu
    blynk_color_and_label(vpins.v_charge_menu, color_menu, "charge EV overnigth ?")
        
    # set up color and label for confidence threshold
    blynk_color_and_label(vpins.v_confidence, "#0000F0", "confidence threshold")

    return blynk_con


# for module importing blynk_ev, without creating the connection
# should not be used, as all those module uses blynk_write and blynk_color, whichs used blynk_con here as global
def get_blynk_con():
    global blynk_con
    return(blynk_con)



def def_call_back():

    # 'NoneType' object has no attribute 'on'
    #############################
    # connect, disconnect call back
    #############################
    @blynk_con.on("connected")
    def blynk_connected(ping):
        global blynk_connected
        blynk_connected = True

        print('Blynk connect call back. Ping:', ping, 'ms')
        # executed when blynk.run() thread is started

    @blynk_con.on("disconnected")
    def blynk_disconnected():
        print('Blynk disconnected')


    #############################
    # charge enabled button
    # set global charge_ev
    # global accessed from  main
    # ['x']  0=auto, 1=on 2=off. defined in widget
    # charge button ['1'] <class 'list'>
    #############################
    @blynk_con.on("V11")
    def my_write_handler11(value):
        global menu_charge_ev # init to -1
        
        if value[0] == '0':
            menu_charge_ev = 0 # can be monitored to make sure call back happened
            print("blynk call back: EV charge auto")

        if value[0] == '1':
            menu_charge_ev = 1 # can be monitored to make sure call back happened
            print("blynk call back: EV charge enabled")

        if value[0] == '2':
            menu_charge_ev = 2
            print("blynk call back: EV charge disabled")


    #############################
    # GOAL: call backs for widgets
    #############################

    # slider (number of hour of charge)
    # set global charge_* , accessed from  main
    # confidence threshold

    @blynk_con.on("V15")
    def my_write_handler15(value):
        global charge_0 # init to 0
        charge_0 = value[0]
        print("blynk call back: slider charge_0 ", charge_0)

    @blynk_con.on("V16")
    def my_write_handler16(value):
        global charge_1 # init to 0
        charge_1 = value[0]
        print("blynk call back: slider charge_1 ", charge_1)

    @blynk_con.on("V17")
    def my_write_handler17(value):
        global charge_2 # init to 0
        charge_2 = value[0]
        print("blynk call back: slider charge_2 ", charge_2)

    @blynk_con.on("V18")
    def my_write_handler18(value):
        global charge_3 # init to 0
        charge_3 = value[0]
        print("blynk call back: slider charge_3 ", charge_3)

    @blynk_con.on("V21")
    def my_write_handler11(value):
        global confidence_threshold 
        confidence_threshold = value[0]
        print("blynk call back: confidence threshold", confidence_threshold)
    

# assumes uses blynk_con as global

######################################
# GOAL: wrappers for blynk.virtual_write() and blynk_event
# do nothing if blynk not available
######################################

def blynk_v_write(vpin, data):
    if config_features.blynk_GUI:
        # use blynk_con from module
        blynk_con.virtual_write(vpin, data)
    else:
        pass

def blynk_event(code, string):
    if config_features.blynk_GUI:
        blynk_con.log_event(code, string) # use event code
    else:
        pass

#############################
# GOAL: change color and labels property
#############################
def blynk_color_and_label(pin, color, label=None):

    # duino
    # Blynk.setProperty(V0, "color", "#D3435C");
    # Blynk.setProperty(V0, "label", "My New Widget Label");

    if config_features.blynk_GUI:
        blynk_con.set_property(pin, "color", color) #  #FF00FF
        if label is not None:
            blynk_con.set_property(pin, "label", label) #  

    else:
        pass


#######################################
# MAIN , for debug, as module typically imported from main
#######################################

if __name__ == "__main__":

    print('%s blynk running as main' %__name__)

    con = create_and_connect_blynk()

    con1 = get_blynk_con()
    assert con == con1

    stamp = "%02d/%02d" %(3, 5)

    label = '17-25Kwh'
    conf = 0.86
    reg = 22.8
    d_=""

    prediction_text = "%s: %s (confidence:%d%%). %0.1fKwh. %s" %(stamp, label, int(conf*100), reg, d_)

    blynk_v_write(vpins.v_pred_label, prediction_text)

    blynk_color_and_label(vpins.v_pred_label, color_text, "prediction for tomorrow: %s" %stamp)

    prod_index = 2
    blynk_color_and_label(prediction_led[prod_index], color_prediction_led[prod_index])

    # turn led on
    blynk_v_write(prediction_led[prod_index], 1)

    stamp = "%02d/%02d" %(3, 4)
    s1= "SUCCESS"
    true_prod = 14.6
    prod_lb = '8-17Kwh'
    conf = 0.77
    post_mortem_text = "%s: %s. truth: %0.1fKwh. pred: %s (%d%%)" %(stamp, s1, true_prod, prod_lb, int(conf*100))

    blynk_v_write(vpins.v_post_mortem_label, post_mortem_text)

    blynk_color_and_label(vpins.v_post_mortem_label, color_text, "postmortem predicting %s" %(stamp))


    t = datetime.date.today() + datetime.timedelta(1)
    stamp = "%02d:%02d" %(t.month, t.day)

    template_header_content = "%d=%dkwh" %(t.day, round(true_prod,0))

    blynk_v_write(vpins.v_header_template, template_header_content)


    blynk_color_and_label(vpins.v_header_template, "#FF0000", "pabou") # this does not change template tile key value


    print ("endless loop. check android app. then please break")
    while True:
        sleep(1)





        





