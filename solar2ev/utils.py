#!/usr/bin/env python3

# https://docs.python.org/3/howto/logging.html#logging-from-multiple-modules

import logging
import os
import datetime

import pandas as pd

import blynk_ev
import pushover

import vpins

tmp_dir = "tmp"

##############################
# logging
# send blynk event (use event)
# write to blynk terminal
# send pushover
##############################

def all_log(s, event):

    assert event in blynk_ev.all_events

    print(s)
    
    logging.info(s)

    blynk_ev.blynk_event(event, s)
    
    blynk_ev.blynk_v_write(vpins.v_terminal, s)

    try:
        pushover.send_pushover(s)
    except:
        pass

#############################
# add row at end of csv
# create if needed
# 1st colum is date. check does not update twice same day
# check vs today(), ie when the code is ran (not as_if date for postmortem. anyway, there is a fixed delta, and this is going to be aggregated)
#############################

def append_to_history_csv(row, hist_csv:str, hdr:list[str]):

    if os.path.exists(hist_csv):
        # add at the end of the csv file, if not already updated
        hist_df = pd.read_csv(hist_csv) 

        # last row
        last_row =  hist_df.iloc[-1] 

        if str(last_row[hdr[0]]) ==  str(datetime.date.today()):
            s = "%s history already updated for %s" %(hist_csv, datetime.date.today())
            print(s)
            logging.info(s)

            
        else:
            s = "update history %s for %s" %(hist_csv, datetime.date.today())
            print(s)
            logging.info(s)

            # add at the end and save to csv

            hist_df.loc[len(hist_df.index)] = row


            hist_df.to_csv(hist_csv, header=True, index = False)

    else:
        # create new csv file
        s = "create new history file %s for %s" %(hist_csv, datetime.date.today()) 
        print(s)
        logging.info(s)

        # using row creates a serie (columns)
        hist_df = pd.DataFrame([row], columns = hdr)
        hist_df.loc[len(hist_df.index)] = row

        hist_df.to_csv(hist_csv, header=True, index = False)
