#!/usr/bin/env python3

import datetime
import sys, os
import pandas as pd
import numpy as np
import config_model

# record training run in csv
# pip install openpyxl

# used by train and brutal force search

record_param_hdr = ["categorical", 
        "bins", "features", "days", 
        "train_samples",
        "seq_build", "selection", "stride", 
        "acc/rmse", "precision/mse", "recall/mae", "prc/msle",
        "run", "sampling", "repeat", "shuffle",  "lstm", "units", "dense", "attention",
        "elapse", "epochs"
    ] 



####################################
# GOAL: record various parameters and metrics
# offline analyzis
####################################

def record_training_run(record_run_csv:str, param_list:list) -> bool:

    # csv file to append to passed as parameter
    # specific file for brutal search
    # generic history for train

    assert len(param_list) == len(record_param_hdr), "record run: not the same number of colums"

    # append row
    # use .append with dict, 
    # use .loc with list   .loc[len(df)]

    try:
        if os.path.isfile(record_run_csv):
            df =  pd.read_csv(record_run_csv)
            df.loc[len(df)] = param_list # append at the end as row df.index.to_list() [0]

        else:
            #df = pd.DataFrame(param_list)  # insert as serie (ie colums)
            df = pd.DataFrame(columns=record_param_hdr)
            df.loc[len(df)] = param_list

        # save pandas as cvs and excel
        df.to_csv(record_run_csv, header=record_param_hdr, index = False) # index = True save index as columns (Unnamed), so increases nb of colums
        
        f = record_run_csv.split(".")[0]+".xlsx"
        df.to_excel(f, sheet_name=config_model.app_name)

        return(True)

    except Exception as e:
        print("Exception recording training run: %s" %str(e))
        return(False)
