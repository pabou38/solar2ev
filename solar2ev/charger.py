#!/usr/bin/python

from time import sleep
import numpy as np


import utils
import myenergi
import config_model
import config_features
import blynk_ev
import vpins

import logging

categorical = config_model.categorical
prod_bins=config_features.config["prod_bins"]


############################ 
# GOAL: configure charger
# read from GUI
############################

def set_charge_tommorrow(dict_result:dict)->bool:

    # param is inference result 
    # if GUI, 
    #    wait for call backs 
    #    set color and labels
    #    set charge mode (auto, on, off), charge hour list, confidence
    # set up charger based on GUI and confidence

    # not needed, uses blynk_write
    #blynk_con = blynk_ev.get_blynk_con()

    # use cat to drive charger, as we need confidence

    result_list = dict_result["cat"]

    prediction = result_list[0] # index (categorical) 
    prod_lb = result_list[1] # str (label)
    proba = result_list[2] # confidence 
    nb_disconnect = result_list[3] # n ot used

    # labels order need to be aligned with GUI
    # 0 = auto, 1= on, 2=off
    charge_label = ["auto", "force charge", "disable charge"]


    #################
    # update charge mode, charge hours and threshold
    #################

    if config_features.blynk_GUI:
        # make sure call backs for slider happened
        # NOTE: endless loop

        c = blynk_ev.menu_charge_ev == -1 or blynk_ev.charge_3 == -1 or blynk_ev.charge_2 == -1 \
            or blynk_ev.charge_1 == -1 or blynk_ev.charge_0 == -1 or blynk_ev.confidence_threshold == -1
        
        # wait until ALL call back happen
        i = 0
        i_max = 10
        while c : 
            sleep(1)
            i = i + 1
            if i> i_max:
                s = "cannot get blynk callbacks for charger"
                print(s)
                logging.error(s)
                return(False)

        ######################################
        # update charge mode, charge hours and threshold  based on GUI
        ######################################

        # global set by GUI call back. -1 is not initialized 
        menu_charge_ev = blynk_ev.menu_charge_ev  # (auto, on, off)
        confidence_threshold = blynk_ev.confidence_threshold 
        charge_hours = [blynk_ev.charge_0, blynk_ev.charge_1, blynk_ev.charge_2, blynk_ev.charge_3] # slider values


    else:
        ######################################
        # use static config for charge mode, charge hours and threshold  
        ######################################
        
        menu_charge_ev = config_features.default_charge
        confidence_threshold = config_features.confidence_threshold
        charge_hours = config_features.config["charge_hours"]
        

    #################
    # set up charge
    #################


    if menu_charge_ev == 2:  
        s = 'charging disabled. do nothing'
        utils.all_log(s, "charging")

        ## note: this means another system will drive charger. not that we configure the charger NOT to charge
        return(True)

    # charging on or auto
    # NOTE: use confidence for auto mode 

    auto_condition = proba > confidence_threshold

    if (menu_charge_ev == 0 and auto_condition) or menu_charge_ev == 1:

        ###########################
        # set up charger
        ###########################

        # build string to log/write to GUI

        #########################
        # concert prediction to metric for charger (hours, soc, kwh)
        #########################
        
        # mind order
        hours = charge_hours[prediction]

        ###################
        # configure charger
        ###################

        if hours != 0:
            ##### configure charger
            s = "SET UP EV CHARGER for %d hours. mode: %s. prediction: %s confidence (%0.1f) higher than threshold (%0.1f)" %(hours, charge_label[menu_charge_ev], prod_lb, proba, confidence_threshold) 
            utils.all_log(s, "charging")


            ###########################
            # make it generic
            ###########################
            # hour, kwh, soc

            myenergi.set_up_car_charger(hours)

        else:
            s = "charging hours set to 0"
            utils.all_log(s, "charging")
            pass # no need to mess around with car charger 

    else:

        # mode auto and categorical, but confidence too low
        s = "DO NOT SET UP EV CHARGER: mode: %s. confidence (%0.1f) lower than threshold (%0.1f)" %(charge_label[menu_charge_ev], proba, confidence_threshold)
        utils.all_log(s,  "charging")

    return(True)