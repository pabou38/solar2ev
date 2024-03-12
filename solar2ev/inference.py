#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
import os
import datetime
from time import sleep, perf_counter
from typing import Tuple, Union, Optional
import json

import blynk_ev
import vpins
from utils import all_log
import config_features
import config_model
import meteo
import enphase
import enphase_API_v4
import vpins
import blynk_ev
import dataframe
import dataset
import utils
from enum import Enum, auto

import shared
model_type = shared.model_type

import tensorflow as tf

# import for cross module logging
import logging

sys.path.insert(1, '../PABOU') # this is in above working dir 
try:
    import pabou
    import various_plots
except Exception as e:
    print('cannot import modules in ..  check if it runs standalone. syntax error will fail the import')
    exit(1)


prod_bins=config_features.config["prod_bins"] # regression, used to compute nb correct

# same dir as the one used by various_plots
save_fig_dir = various_plots.save_fig_dir # for error distribution

#############################
# daily inference history 
# in case needed for further analysis
# updated in predict_solar_tomorrow(model)
#############################
predict_solar_tomorrow_history = "predict_solar_tomorrow_history.csv"
predict_solar_tomorrow_history_hdr = ["date", "prediction", "label", "confidence", "disconnect"]


#################
# postmortrem history
###################
postmortem_history_csv = "postmortem_history.csv"
postmortem_history_hdr = ["date", "cat_prediction", "scalar_prediction", "truth", "confidence", "disconnect"]

####################
# GOAL: get the class (bin number) for a given scalar production
####################

def get_solar_class(solar):
    # to be compared with prediction (argmax) 

    # "prod_bins":[0, 5, 10, 15, 20, 25, max_expected_production],


    a = np.digitize([solar],prod_bins, right=False) 
    # Return the indices of the bins to which each value in input array belongs.
    # right bool. rigth belong to bin or not 
    # indices start at 1 !!!!
    assert len(a) == 1

    return(a[0]-1)   # 14,9 returns 3, y trained with starting at 0

# apply argmax on ONE softmax . required if output is categorical
def get_argmax(a):
    #a = tf.nn.sigmoid(a)
    # By default, the index is into the flattened array, otherwise along the specified axis.
    # either one array of softmax, or multiples
    if len(a.shape) > 1:
        return(np.argmax(a, axis=1))
    else:
        return(np.argmax(a))


##################
# GOAL: .predict() on any dataset containing labels
# called from various place (train, retrain, tuner, vault, unseen, load model)
################## 

def predict_on_ds_with_labels(model, ds, verbose=False, s ="") -> Tuple[int, int, int, float, Tuple]:

    # categorical: return nb sample, 2x nb correct prediction (same value) , average confidence for OK
    # return(nb, nb_correct, nb_correct, (confidence))

    # regression: return nb sample, 2x nb correct = (nb within acceptable error AND use bins) , (mae, ae list)
    # return(nb, nb_correct , nb_correct1, (mae computed, non_absolute_errors_l))
    # nb_correct:  use ABSOLUTE , compare with acceptable_error , interpreted as ae
    # nb_correct 1: ùap to categorical bins

    # verbose = True: print for each individual prediction
    # look also at pabou.benchfull for another way to iterate on predict result (2 for loops)


    print("predict on dataset with label. source: %s" %s)

    # look at label to see if one hot
    categorical = pabou.is_categorical_ds(ds)
    
    # if set to batch 1 before .predict() , very slow
    pred = model.predict(ds, verbose=0) # predict on dataset seems to return array (of softmax), not list
    print(".predict(ds with labels): got %d predictions. categorical: %s" %(len(pred), categorical)) #  prediction: <class 'numpy.ndarray'> (1083, 4)

    # go thru all truth labels one by one
    
    sample = list(iter(ds.unbatch()))  # nb sample in dataset
    assert len(pred) == len(sample),  "nb of pred: %d and nb of sequences in dataset: %d do not match" %(len(pred), len(sample))

    # ((input,label), prediction)
    z = zip(sample, pred)

    # pred is a "flat" structure with as many softmax as sequences in ds
    # WTF: len(list  .. does not always returns the same thing. seems there is state inside. anayway, I can trust tensorflow

    ###############
    # better way to compare two numpy directly ?
    ##############
    
    nb = 0
    # both nb_correct are the same for classification
    # for regression: nb_correct is abs(error) > max and nb_correct1 maps both scalar target and prediction to bin used for classification
    nb_correct = 0 # regression: abs(error)
    nb_correct1 = 0 # regression: map scalar to bins

    confidence = 0 # for classification, means of all argmax value. cannot set to None, as will be added to float
    mae = 0 # for regression, mean of sum of abs(p-label)

    #for i, e in enumerate(ds.unbatch()): # use unbatched dataset. use enumerate as I need index for pred[]
        #input = e[0] # TensorShape([1, 116, 4])     we do not need input there
        #label = e[1] # TensorShape([1, 4])   GROUND TRUTH

    # record non absolute errors (regression only)
    non_absolute_errors_l = [] 

    ######################
    # compute nb_error, nb_error1 (same for categorical)
    ######################

    for (sample,label), p in z:  # sample TensorShape([66, 5]) label TensorShape([4]) p.shape (4,)

        # label <tf.Tensor: shape=(), dtype=float64, numpy=10.97>
        # p: array([14.848276], dtype=float32) for regression
        # p: array([0.67746025, 0.27571043, 0.04114275, 0.00568655], dtype=float32) for classification

        if categorical:
            ####################
            # categorical, nb_correct is the same as nb_correct1
            ####################

            # input TensorShape([90, 5]) , label TensorShape([4]); softmax (4,)
            # test if label == prediction. both are one hot. pred[i] is current softmax
            index_1 = np.argmax(p)
            index_2 = np.argmax(label) # ground thruth

            #### compare argmax, ie correct ?
            if index_1 == index_2:
                nb_correct = nb_correct +1
                confidence = confidence + p[index_1]  # value of argmax, ie proba/confidence
                nb_correct1 = nb_correct1 +1 # same as nb_correct for categorical

            ###############
            # print all
            ###############

            if verbose:
                print("categorical prediction %d, truth %d, confidence %0.2f. %s" %(index_1, index_2, p[index_1], index_1==index_2))

        else:

            ####################
            # regression
            # use max delta (acceptable error) AND bins to compute nb_correct and nb_correct1
            ####################

            # label <tf.Tensor: shape=(), dtype=float64, numpy=10.97>
            # p: array([14.848276], dtype=float32)

            # p is array, but can be used as scalar
            assert p == p[0]

            # compute all Non AE
            # AE= prediction - truth (ie is negative if prediction less than thru)
            non_absolute_errors_l.append(p-label) # list of non absolute
            
            ###############
            # 2 ways to assess if "correct" for regression
            ###############

            # method 1: use ABSOLUTE , compare with acceptable_error , interpreted as ae
            if abs(p-label) < config_features.regression_acceptable_absolute_error: # compare ae
                nb_correct = nb_correct +1
            

            # method 2: check if label and prediction falls in same bins, as defined by prod_bins (same one as for categorical)
            inds = np.digitize([p], prod_bins)  # Output array of indices, of same shape as x.
            assert len (inds) == 1
            index_prediction = inds[0] -1

            inds = np.digitize([label], prod_bins)  # Output array of indices, of same shape as x.
            assert len (inds) == 1
            index_true = inds[0] -1

            if index_prediction == index_true:
                nb_correct1 = nb_correct1 +1


            ###############
            # print all
            ###############

            if verbose:
                print("regression prediction %0.2f, truth %0.2f . ae error %0.2f" %(p, label, abs(p-label)))


        nb = nb + 1
    assert nb == len(pred)

    # analyze errors distribution done in caller, only after training



    if categorical:
        # NOTE # average confidence FOR OK results ONLY
        confidence = confidence/nb_correct 
        print(".predict(): %0.0f%% accuracy (should be equal to categorical accuracy from model.evaluate). average confidence for OK: %0.2f" %(100*nb_correct/nb, confidence))
        print("%0.2f%% correct" %(nb_correct/nb))
        
        return(nb, nb_correct, nb_correct, (confidence))
    
    else:

        non_absolute_errors = np.array(non_absolute_errors_l) # convert to numpy

        absolute_errors = np.abs(non_absolute_errors) # convert to abs

        mae = np.mean(absolute_errors)  # compute mae

        ####################
        # also look at % of prediction below truth, ie model too conservative, pessimistics
        ####################

        i = 0
        for x in non_absolute_errors_l:
            if x <0:
                i = i + 1

        under = i / len(non_absolute_errors_l)


        print(".predict(): mean absolute error: %0.2f Kwh" %(mae))
        print("%0.2f%% correct using %0.1f Kwh acceptable error" %((nb_correct/nb), config_features.regression_acceptable_absolute_error))
        print("%0.2f%% correct using categorical bins" %(nb_correct1/nb))
        print("%0.2f%% of under predictions (conservative/pessimistics)" %under)

        return(nb, nb_correct , nb_correct1, (mae, non_absolute_errors)) # return errors for further analysis (distribution analysis)
        
    


#################################
# GOAL: inference for a given date
# used by daily inference and postmortem
# returns (False, str)
# or (True, result_dict)
#################################

def inference_from_web_at_a_day(model, inference_date, model_param_dict, average = False , daily_inference = None, postmortem_next_day = None) -> Tuple[bool, Union[dict, str] , Optional[float] ]:

    # date is the time at which inference is ran and so last day in sequence
    # average = True get one day extra in the past, to get more than one input sequence , then compute nb_disconnect predictions not the same as average

    # model is loaded, as configured by categorical (later will load the "other one")


    # ONLY time where both model (categorical, regression) are available
    # so good time to update GUI with model info, ie result from evaluate. that is why test_ds is there

    # returned dict have "cat" and "reg" as key. using Enum
    # and value:  p1: prediction (argmax or scalar), l1: prod_lb str (mapped for scalar), confidence [-1 if scalar], nb_disconnect [-1 if not averaging], solar_ground_truth (-1 for na)

    print("\nInference from web as if %s. averaging: %s. daily_inference: %s. postmortem_next_day: %s" %(inference_date, average, daily_inference, postmortem_next_day))

    days_in_seq = model_param_dict["days_in_seq"]
    categorical = model_param_dict["categorical"]
    retain = model_param_dict["retain"]
    sampling = model_param_dict["sampling"]
    seq_len = model_param_dict["seq_len"]
    input_feature_list = model_param_dict["input_feature_list"]
    selection = model_param_dict["selection"]
    seq_build_method = model_param_dict["seq_build_method"]
    hot_size = model_param_dict["hot_size"]
    batch_size = model_param_dict["batch_size"]
    prod_bins = model_param_dict ["prod_bins"]
    prod_bin_labels = model_param_dict["prod_bin_labels"]
    stride = model_param_dict["stride"]
    shuffle = model_param_dict["shuffle"]
    prod_bins_labels_str = model_param_dict["prod_bins_labels_str"]

    #needed for .evaluate, to compute metrics
    test_ds = model_param_dict["test_ds"]

    header_csv = meteo.header_csv  # for scrapped meteo csv
    installation = enphase.installation  # installation day. 
    production_column = enphase.production_column 
    acceptable_error = config_features.regression_acceptable_absolute_error

    hour = int(inference_date.strftime("%H"))
    minute = int(inference_date.strftime("%M"))

    if inference_date > datetime.datetime.now():
        print('inference: date is future')
        return(False,"cannot predict for date in the future")

    # time limit if need today solar, need to wait until sunset
    # today_skip_hours = 6. need to scrap 0 to 17, skip 18 to 23 (ie 6)
    # not before = 18 , 17h wait, 18h ok. granularity is hour

    # WARNING: above is to make sure we get a complete "today" with telemetry
    # HOWEVER, looks like the enphase cloud is not updated with previous day (if using simple daily api) until later than 7pm CET
    #   likely a time difference issue, CET is 9h ahead of CET
    #    either wait until 7:30pm CET, or get yesteray with telemetry as well

    today_skipped_hours = config_features.today_skipped_hours
    expected_hours = 24 - today_skipped_hours  # 18h

    # enforce time limit for daily inference , or postmortem with postmortem_next_day = False
    # do not enforce time limit for postmortem with postmortem_next_day = True, BUT go one extra day back in past

    #not_until = expected_hours # need to wait a bit for to get daily solar updated in enphase cloud

    not_until_h  = config_features.not_until_h # wait for 19:30
    not_until_mn = config_features.not_until_mn

    s = "%d:%d" %(not_until_h, not_until_mn)

    if hour == not_until_h:
        time_limit_ok = minute >= not_until_mn
    elif hour > not_until_h:
        time_limit_ok = True
    else:
        time_limit_ok = False

    if daily_inference or (not daily_inference and not postmortem_next_day):
        print('enforce time limit')

        if inference_date.date()== datetime.date.today() and not time_limit_ok: # today before 
            print("\nPLEASE wait until %s to get all telemetry solar data needed for today AND cloud updated for yesterday" %(s))
            return(False,"wait until %s" %(s))
        else:
            print("time limit ok")
        
    else: 
        # should be daily_inference = False, postmortem_next_day = True 
        print("no need to enforce time limit. daily_inference %s, postmortem_next_day: %s. but will go one day back in past" %(daily_inference, postmortem_next_day))
        pass # no need to enforce (postmortem )
    

    # set how many days in the past I need to get, need one extra if averaging
    if average == False:

        # eg 3 days in the past , to get 4 consecutive days (including date parameter)
        number_days_in_past = days_in_seq - 1
        # for inference, this garentee only ONE sequence
        #   (could get a few more in inference is ran after 18)
        expected_days = days_in_seq

    else:
        # one extra day in the past, to get more than one sequence
        number_days_in_past = days_in_seq
        expected_days = days_in_seq + 1

    print("averaging multiple input: %s, go back %d days in past, expects %d days. (days_in_seq: %d)" %(average, number_days_in_past, expected_days, days_in_seq))

    # we need to get meteo/solar data from this date in the past 
    start_date = inference_date - datetime.timedelta(number_days_in_past) 

    # end date is date passed as parameter. date at which inference is ran
    end_date = inference_date 
    print("create df_model from %s (included) to %s (included). ie run inference as date:%s" % (start_date.date(), end_date.date(), end_date.date()))

    ###############################################
    # build meteo and solar df and combine to create df_model
    ###############################################

    ##### build METEO df 
    # scrap from start date to end date. 
    # reuse create a df_meteo csv, and use existing cleaning


    print("\nbuild meteo df from %s to %s" %(start_date, end_date))

    days = [] # store list of dict. each entry is one day , ie one day_result
    date_= start_date
    # https://www.meteociel.fr/temps-reel/obs_villes.php?code2=278&jour2=12&mois2=0&annee2=2022
    
    while date_ <= end_date:

        if date_ == end_date:
            # for daily inference, or not postmortem_morning , last day is going to be incomplete (query web site early evening, and late evening hours not yet available) . for postmortem it is complete
            # for postmortem_morning, 24 hours are available, even if I need less (getting more is OK on one_day())
            day_result = meteo.one_day(date_, expected_hours) 
            assert len(day_result) >= expected_hours
            
        else:
            # returns dict for FULL day
            day_result = meteo.one_day(date_, 24) 
            assert len(day_result) == 24

        # check date in day_result is indeed the date querried. for 1st and last date
        # day_result["0"][0] Timestamp('2023-02-11 00:00:00'). date datetime.datetime(2023, 2, 11, 9, 13, 44, 746554).  convert both to date()
        assert date_.date() == day_result["0"][0].date()   # date in hour 0 pandas timestamp is the same as asked date

        s = str(24-today_skipped_hours-1)
        assert date_.date() == day_result[s][0].date()   # 24-4  date in last expected hour , eg 23 or 19

        days.append(day_result)

        date_ = date_ + datetime.timedelta(1)
        sleep(1) # good citizen. do not overflow the web site

    # got all days in the past
        
    assert len(days) == expected_days

    # create # list of hours, from list of dict. each hour is a list [Timestamp('2022-11-2...00:00:00'), '0', '0.7', '96%', 'Nord', '0', '1018.8']
    hours = [] 
    for day in days[:-1]:
        assert len(day) == 24 
        for h in range(24):
            r = day[str(h)] # one row per hour
            assert len(r) == len(header_csv)
            hours.append(r)

    # last day in sequence, (ie today in case if inference, some day in the past in case of postmortem) must have enough hours. 
    day = days[-1]
    assert len(day) >= 24 - today_skipped_hours  

    # could bet more, for postmortem, last day is day before yesterday
    # could get more, ie scrap at 11pm while expecting until 7pm 

    # for LAST day, only retain hours needed for seq len
    for h in range(24-today_skipped_hours):  
        r = day[str(h)] # one row per hour
        assert len(r) == len(header_csv)
        hours.append(r)

    assert len(hours) == (expected_days -1) * 24 + 24 - today_skipped_hours
    # len(df_meteo) = 90, day_in_seq = len(df_prod) = 4 , skipped_hours = 6  . 3*24 + 24-6 = 90

    # create meteo dataframe and csv
    # same format as original scrapped meteo 
    df_meteo = pd.DataFrame(hours, columns = header_csv)  # format same as after initial scappping

    file = os.path.join("tmp", "tmp_meteo_df_4_inference.csv")
    df_meteo.to_csv(file, header=True, index=False)

    # reuse existing meteo cleaning,
    df_meteo = dataframe.clean_meteo_df_from_csv(file) # 

    # len(df_meteo) / 24 = 4.75 for 5 days (4 days plus one for average)
    if average == False:
        # df_meteo contains just one sequence
        assert len(df_meteo) / sampling == seq_len
    else:
        pass 

    assert df_meteo.isnull().sum().all() == 0

    # meteo df ready


    #######################
    # build SOLAR df
    #######################

    print("\nbuild solar df from %s to %s" %(start_date, end_date))

    # solar data needed to build input sequence
    # for postmortem_morning, can get it all in one go (get_daily_solar_from_date_to_date) as this is all in the past (and in enphase cloud)
    # for daily inference or not postmortem_morning, need to use telemetry API call (get_telemetry_energy_today) to get last date (ie today)


    if not daily_inference and postmortem_next_day:
        #additionally, for postmortem the next day we can get an extra day , ie the ground truth

        # but WTF, need to wait for next day late afternoon for enphase cloud to be updated with daily data

        # add one day, should be ok with update of enphase cloud
        # this give grounbd truth for solar, will be returned (and removed before running inference)

        production_list = enphase_API_v4.get_daily_solar_from_date_to_date(start_date, end_date+datetime.timedelta(1)) # list of wh
        assert len(production_list) == expected_days + 1

        # last element of production list is ground truth
        solar_ground_truth = production_list[-1] /1000.0

        # rebuild production list just for inference
        x = production_list.pop()
        assert x/1000.0 == solar_ground_truth
    
    else:
        # get solar except today
        end = end_date + datetime.timedelta(-1) 
        production_list = enphase_API_v4.get_daily_solar_from_date_to_date(start_date, end) # list of wh
        assert len(production_list) == expected_days -1

        # get solar so far today, using telemetry data. expect to run after 6pm, when sun is set, so that telemetry data is complete
        wh, _ = enphase_API_v4.get_telemetry_energy_from_date() # no date ie today
        production_list.append(wh)

        solar_ground_truth = -1

    assert len(production_list) == expected_days

    # beware. production in wh. if leave as it, will always predict 
    production_list = [x/1000.0 for x in production_list]


    # create df_prod, as for initial scrapping, to reuse combine code
    # assumes not need to manage outliers, all value in wh
    # WARNING: assumes no need to normalize 

    # first create list of date
    date_list = []
    date_ = start_date
    for _ in range(expected_days):
        date_list.append(date_.date())
        date_ = date_ + datetime.timedelta(1)

    # creates df_prod
    data = {
        "date" : date_list,
        production_column : production_list
    }

    # do not use index = date_list; it creates a separate index, and keep date columns
    df_prod = pd.DataFrame(data) 

    # convert to datetime. having date_list a list of datetime object does not seem to be sufficient
    df_prod["date"] = pd.to_datetime(df_prod["date"]) # df_prod["date"][0] Timestamp('2023-02-11 00:00:00')

    # set index to date
    df_prod.set_index(df_prod.columns[0], append=False, inplace=True)


    # converting to pd.to_datetime change df_prod.index
    # FROM Index([2023-02-11, 2023-02-12, 2023-02-13], dtype='object', name='date')
    # TO DatetimeIndex(['2023-02-11', '2023-02-12', '2023-02-13'], dtype='datetime64[ns]', name='date', freq=None)

    assert len(df_prod) == expected_days
    assert df_prod.isnull().sum().all() == 0
    # solar df ready

    ##### build COMBINED meteo and solar df

    df_infer = dataframe.create_feature_df_from_df(df_meteo, df_prod, production_column)
    # one line per hour. not even number of days, as last days is not complete
    # sampling not yet applied

    # enough hours to create ONE input sequence (average = False)
    # len(df_meteo) = 90, len(df_prod) = 4 , skipped_hours = 6  . 3*24 + 24-6 = 90
    # do not check integrity. len not a multiple of 24 

    if not average:
        assert len(df_infer) / sampling  == seq_len 

    assert (len(df_infer) + today_skipped_hours) / len(retain) == expected_days  

    #save for debug , future use
    # df_model type, ie ready to build dataset
    df_infer.to_csv(os.path.join("tmp","tmp_df_from_web_4_inference.csv"), header=True, index=False)

    ######################################
    # df_infer ready. create model input
    ######################################

    # either manually, ie create X numpy (but then need to care sampling, not stride), 
    # or reuse dataset creation: YES

    """
    # manual method
    # need to apply sampling (is done in creating dataset when training)
    df_infer = df_infer[::sampling]   # :: is all, last is step
    assert len(df_infer)   == seq_len 
    # There is only one sequence, so no need to apply stride
    """

    ### create dataset for model input
    input_shape = pabou.get_input_shape_from_model(model) # return list of one or more tuple
    assert len(input_feature_list) == len(input_shape) # [['temp', 'pressure']] [(68, 2)]
    print("model input shape ", input_shape)

    ### create dataset from dataframe
    # df_infer not an even number of days. just enough for seqlen

    # labels = True means ignore last X and first Y and align, ie training mode
    # labels = False for daily inference (predict tomorrow or postmortem). labels not included
    # in a way labels = not for_training

    ds_infer, nb_seq, nb_hours = \
    dataset.build_dataset_from_df(df_infer, \
    input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
    stride=stride, sampling=sampling, shuffle=shuffle, labels=False, categorical=categorical)

    print("\ndataset from df_infer: %d seq, from %d hours (seq len %d, stride %d)" %(nb_seq, nb_hours, seq_len, stride))
    
    # check number of input sequence
    if average == False:
        assert nb_seq == 1
    else:
        # we have an extra day in the past, so more than one sequence
        if seq_build_method == 1:
            assert nb_seq == len(retain)/sampling + 1 # so 24/stride extra sequences (method 1) 
        if seq_build_method == 3:
            assert nb_seq == len(selection) + 1


    i = iter(ds_infer).next()
    print('dataset for inference: ', i.shape)

    # X array or list of array. batch dim included
    #pred = model(X)  # .predict is for batches. for small number of input that fit in one batch, use model()
    # cannot use model(dataset). this needs tensor

    ##########################
    # make sure both model are available
    ##########################

    current_model = model

    # load "other" model
    if categorical:
        other_model_name = config_model.app_name + "_reg"

    else:
        other_model_name = config_model.app_name + "_cat"

    try:
        x = 'tf'
        print('\nload other %s model: %s' %(x, other_model_name))
        other_model, h5_size, _ = pabou.load_model(other_model_name, x)
        if other_model ==  None:
            raise Exception ("!!!!!! CANNOT LOAD MODEL %s %s" %(other_model_name, x))
    except Exception as e:
        raise Exception ("!!!!!! %s CANNOT LOAD MODEL %s %s" %(str(e), other_model_name, x))
    
    
    ##########################
    # run inference on both the default model (based on categorical bool) and the other one
    ##########################

    # store inference results for both models
    # value is tuple
    #{"cat": (p1, l1, confidence, nb_disconnect, solar_ground_truth)}
    results_dict = {} 

    # NOTE: cannot use model.evaluate() as we have only ONE test_ds
    # use json file created when training


    for i , model in enumerate([current_model, other_model]):

        # NOTE: cannot use global categorical anymore and for loop does not create scope 
        # global categorical = True and i = 0, means we are using "current_model" , so this is categorical
        # could also look at model's output shape

        """
        def local_categorical(i):
            if categorical and i == 0:
                return(True)
            if not categorical and i == 1:
                return(True)
            return(False)
        
        """
        # (None, 1) (None, 4)
        local_categorical = model.output_shape[1] != 1

        # keys to dict
        if local_categorical:
            k = model_type.cat # enum
        else:
            k = model_type.reg
        
        ##################################
        # at last, the prediction
        
        t1 = perf_counter()
        print (".predict(). categorical:%s" %local_categorical)
        pred = model.predict(ds_infer)
        print("prediction done in %0.1fms" %((perf_counter()-t1)*1000.0))


        print (pred)
        ##################################

        ######################
        # analyze prediction based on local_categorical and average
        #######################

        if average == False:
            assert len(pred) == 1
            nb_disconnect = -1 # not used

            if local_categorical:
                # [[0.02546321 0.15462233 0.73875284 0.08116163]]
                # pred[0] <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.03526222, 0.04678139, 0.30466294, 0.6132934 ], dtype=float32)>
                # prediction tf.Tensor([[8.4383109e-14 4.2994866e-08 7.2669776e-05 9.9992728e-01]], shape=(1, 4), dtype=float32)
                
                p1 = np.argmax(pred[0]) # index
                l1 = prod_bins_labels_str[p1] # description string

                # if using model(tensor) 
                #confidence = round(pred[0][i].numpy(),2) # array of tensor, need to convert to numpy

                # using model.predict() for dataset
                confidence = round(pred[0][p1],2) # array of float

            else:
                # scalar
                confidence = -1

                p1 = pred[0][0] # scalar

                #### 
                # label is just str(scalar) (but information already there in p1)
                l1 = str(p1)

                # better label mapping to same bins as categoricals
                # map scalar predictions using bins used for classification
                # prod_bins "prod_bins":[0, 6.9, 15.7, 21.9, 1000000],
                inds = np.digitize([p1], config_features.config["prod_bins"])  # Output array of indices, of same shape as x.
                assert len (inds) == 1
                index = inds[0] -1
                l1 = config_features.config["prod_bins_labels_str"] [index]

        else:
            ##################################
            # average = True, get n prediction
            ##################################

            # update nb_disconnect


            if seq_build_method == 1:
                assert len(pred) == len(retain)/sampling + 1
            if seq_build_method == 3:
                assert len(pred) == len(selection) + 1

            # first entries in pred starts early in day - 5, so may not be that representative of what we want

            # get a few prediction toward the end. 
            # last one is the one starting hour 0 day-3, should the same as if using average = False
            # the others starts days - 4 late in the day, 
                
            n = min(6,len(pred))
    
            pred = pred[-n:]
            print("averaging last %d predictions\n" %n, pred)
            
            # multiple softmax/[scalar] in pred.


            ####################
            # how to use averaging, and compute disconnect
            ####################

            # we have n softmax, or n scalar
            # OLD method: returned prediction as the last one, and we compute disconnect = nb of other softmax/scalar which do not agree
            # disconnect is the number of non MAIN prediction which do not match the main prediction (use acceptable_error for scalar)
            
            # NOW: returned prediction as some kind of vote, or majority softmax , or average, ie include effect of ALL softmax/scalar
            # disconnect is then number of predictions which do not match the average aka returned prediction

            if local_categorical:

                # final argmax, confidence uses AVERAGE softmax
                # nb_disconnect: number of time a preduction is not the argmax same as the average

                # print last prediction for reference
                p1 = np.argmax(pred[-1])
                l1 = prod_bins_labels_str[p1]
                confidence = round(pred[-1][p1],2)
                print('last categorical prediction (for reference as using AVERAGE): index: %d, label %s, confidence: %0.2f' %(p1, l1, confidence))

                # compute prediction to return, ie average of all softmax
                
                #[[0.810926   0.15838158 0.03069241]
                #[0.78586316 0.17737414 0.03676279]
                #[0.7432839  0.20945063 0.04726547]
                #[0.6841943  0.25273457 0.06307112]
                #[0.6090441  0.30314004 0.08781589]] 

                pred_array = np.array(pred)
                s = pred_array.sum(axis=0)
                s = s / len(pred)
                s = s.tolist()

                # prediction is AVERAGE of all softmaxes
                p1 = np.argmax(s) # argmax of average
                l1 = prod_bins_labels_str[p1] # description string
                confidence = round(s[p1],2) 

                ##########################
                # disconnect
                ##########################

                # BEFORE look at other prediction and compute nb of disconnects, ie index do not match MAIN, ie LAST prediction
                # NOW look at ALL prediction and compute nb of disconnects, ie index do not match AVERAGE prediction

                nb_disconnect = 0

                for i in range(len(pred)):
                    p2 = np.argmax(pred[i])
                    if p2 != p1:
                        print('disconnect categorical. index do not match AVERAGE: ', p1, p2)
                        nb_disconnect = nb_disconnect + 1

                print("averaging with categorical: number of disconnect (predictions does not match AVERAGE) %d" %nb_disconnect)

            else:

                # scalar is average of scalar
                # confidence = 0

                # main , ie last prediction, for reference
                p1 = pred[-1][0] 
                l1 = str(p1)
                confidence = -1
                print('last scalar prediction (for reference only, as using AVERAGE): %0.2f' %p1)
        
                # returned prediction is mean/average of all scalar
                pred_array = np.array(pred)
                s = pred_array.sum(axis=0)
                s = s / len(pred)

                p1 = s[0][0]
                l1 = str(p1) # label is just the scalar
                confidence = -1 # not used


                # look at other prediction to compute disconnect
                # disconnect is prediction - average of all prediction > max error
                nb_disconnect = 0

                for i in range(len(pred)):
                    p2 = pred[i][0]
                    if abs(p2 - p1) > acceptable_error:
                        print('disconnect for a scalar. pred - average of pred larger than acceptable error: ', p2, p1, acceptable_error)
                        nb_disconnect = nb_disconnect + 1

                print("averaging with regression. number of disconnect (prediction - AVERAGE > acceptable error (%0.1f): %d)" %(acceptable_error,nb_disconnect))

        print("\n\n===> inference_from_web: categorical %s, averaging %s : prediction (index/scalar) %0.2f. label: %s. confidence %0.1f (-1 if regression). nb_disconnect %d (-1 is not averaging)\n" %(local_categorical, average, p1, l1, confidence, nb_disconnect))
        
        ############################
        # update result_dict
        #############################
        results_dict[k] = (p1, l1, confidence, nb_disconnect) # key is enum

        # index, str, conf, -1/int
        # float, str(float), -1, -1/int

        # cannot get metrics for BOTH model, as we have only ONE test_ds
        #metrics_dict[k] = model.evaluate(test_ds, return_dict=True, verbose=0) 


    # we have now a dict with results for both inference. 
    assert len(results_dict) == 2

    #solar_ground_truth is set or -1

    return(True, results_dict, solar_ground_truth)



###########################
# prediction solar for tomorrow
###########################

def predict_solar_tomorrow(model, model_param_dict:dict)->Tuple[bool, dict]:

    # calls inference_from_web_from_a_given_day() 

    # called by daily inference and charge (charge needs confidence, so re-running it. I know this is a bit of overkill)
    # return (bool, dict)
    # test_ds is needed, as this will eventually update GUI model info (gauge) with metrics for BOTH categorical and regression

    # triggered by cron or daemon or systemd timer
    # update GUI. terminal, labels, led, events
    # add to prediction history file (offine analysis), 1st field to date.today(), ONCE per day
    # history file is only prediction, does not contains solar

    #prediction tf.Tensor([[8.3753507e-14 4.4018609e-08 7.1889539e-05 9.9992812e-01]], shape=(1, 4), dtype=float32)

    average_inference = config_model.average_inference
    expected_hours = 24 - config_features.today_skipped_hours
    s = "now is %s, and predicting solar for tomorrow. averaging multiple inference:%s. expected_hours (today): %d" %(datetime.datetime.now(), average_inference, expected_hours)
    all_log(s , "prediction")

    prod_bins = model_param_dict["prod_bins"]

    # run inference as TODAY (is last day in sequence is TODAY)
    # need input until today 7pm
    # ret = (bool, str/result_dict)
    #   index, str, conf, -1/int
    #   float, str(float), -1, -1/int

    ret, result_dict, solar_gound_truth = inference_from_web_at_a_day(model, datetime.datetime.now(), model_param_dict, daily_inference = True, postmortem_next_day = None)
    assert solar_gound_truth == -1

    if not ret:
        s = 'daily prediction failed: %s' %result_dict
        all_log(s, "system_error")
        return(False, {})


    # #{"cat": (p1, l1, confidence, nb_disconnect)}
    cat_results = result_dict[model_type.cat]
    reg_results = result_dict[model_type.reg]

    s = "inference from web OK. categorical results: %s, regression results: %s" %(cat_results, reg_results)
    print(s)
    logging.info(s)

    # (p1, l1, confidence, nb_disconnect)
    #p1: prediction (argmax or scalar), l1: prod_lb str (mapped for scalar), confidence [-1 if scalar], nb_disconnect [-1 if not averaging], solar_ground_truth (-1 for na)

    ####################################
    # build text to write to GUI (prediction tab)
    ####################################

    # time stamp for which the production is done
    # ie "tomorrow"
    # 12/25
    # in widget's label (not value) to save space and make it clearer
    t = datetime.date.today() + datetime.timedelta(1)
    stamp = "%02d/%02d" %(t.month, t.day)

    # use disconnect for categorical model
    if average_inference:
        d_ = "disconnect: %d." %cat_results[3]
    else:
        d_ = "" # nb_disconnect should be set to-1


    # build text with summary of results, for GUI
    # combine both results
    # make sure concise, clear and fit into label
    # 02/22: 8-17Kwh (81%). 12.6Kwh [disconnect: x]
        
    # index, str, conf, -1/int
    # float, str(float), -1, -1/int
    prediction_text = "%s: %s (confidence:%d%%). %0.1fKwh. %s" %(stamp, cat_results[1], int(cat_results[2]*100), reg_results[0], d_)

    # need to fit in widget
    # https://docs.blynk.io/en/getting-started/template-quick-setup/set-up-mobile-app-dashboard
    # color, size, and labels for value (title type = Key Value) can be configured in the app, but not sure with API ?
    template_header_content = "%d=%dKwh" %(t.day, round(reg_results[0],0)) # just fit when template setting -> tile design ->value size = small

    s= "prediction_text: %s" %prediction_text
    print(s)
    logging.info(s)

    s= "template header content: %s" %template_header_content
    print(s)
    logging.info(s)


    # index from categorical
    prod_index = cat_results[0]


    # "index" from regression
    # as bonus convert production scalar into index in bins (same bins as configured for categorical)
    bins = np.array(prod_bins)
    x = [reg_results[0]]

    # Return the indices of the bins to which each value in input array belongs.
    inds = np.digitize(x, bins, right=False) # Output array of indices, of same shape as x.
    # array([1], dtype=int64)  1 for FIRST INTERVAL

    assert len(inds) == 1
    prod_index_ = inds[0] - 1 # starts at 1

    s = "categorical index %d, regression index %d" %(prod_index, prod_index_)
    print(s)
    logging.info(s)


    # send to all
    all_log(prediction_text, "prediction")

    # update Blynk

    ########################################
    # prediction summary to label (persistent)
    #########################################
    blynk_ev.blynk_v_write(vpins.v_pred_label, prediction_text)

    blynk_ev.blynk_color_and_label(vpins.v_pred_label, blynk_ev.color_text, "prediction for tomorrow: %s" %stamp)


    ##########################
    # template header
    #########################
    # very concise summary
    blynk_ev.blynk_v_write(vpins.v_header_template, template_header_content)


    ######################
    # application header
    ######################
    # whatever usefull
    s= "https://medium.com/@pboudalier/"

    blynk_ev.blynk_v_write(vpins.v_header_app, s)

    #################################
    # led
    #################################

    # WARNING. number of class/intervals need to be synched with number of leds defined in Blynk
    # make sure all led are OFF and write led labels
    # write labels in white
    # v0 is left, 0-8kwh

    # turn all OFF
    for i, l in enumerate(blynk_ev.prediction_led):
        blynk_ev.blynk_v_write(l, 0) # turn off based on vpin

        # OK , do it everytime. could have done only once
        blynk_ev.blynk_color_and_label(l, "#FFFFFF", config_features.config["prod_bins_labels_str"][i] ) # write labels

    # turn one prediction led ON
        
    # led color indicates "strength"
    # 0 is lowest prediction/bins, ie led on the left
        
    blynk_ev.blynk_color_and_label(blynk_ev.prediction_led[prod_index], blynk_ev.color_prediction_led[prod_index])

    # turn led on
    blynk_ev.blynk_v_write(blynk_ev.prediction_led[prod_index], 1)

    s = "led index %s. led color %s" %(str(blynk_ev.prediction_led[prod_index]), str(blynk_ev.color_prediction_led[prod_index]))
    print(s)
    logging.info(s)
 
  
    #################################
    # add row to inference history csv
    # could be handy for later analysis
    # add 2 rows, one for reg, one for cat
    # only history of inferences. no thruth
    ##################################
    
    row1 = [datetime.date.today(), cat_results[0], cat_results[1], cat_results[2], cat_results[3]]
    assert len(row1) == len(predict_solar_tomorrow_history_hdr)

    row2 = [datetime.date.today(), reg_results[0], reg_results[1], reg_results[2], reg_results[3]]
    assert len(row2) == len(predict_solar_tomorrow_history_hdr)

    utils.append_to_history_csv(row1, predict_solar_tomorrow_history, predict_solar_tomorrow_history_hdr)
    utils.append_to_history_csv(row2, predict_solar_tomorrow_history, predict_solar_tomorrow_history_hdr)

    ##########################
    # as Blynk is running, update info and gauges 
    ##########################
    update_GUI_model_tab()

    # return dict of results (in case)
    return (True, result_dict)


#################################### 
# run one prediction (at daily inference time) or next day to validate solar production vs thruth
# calls inference_from_web_from_a_given_day
###################################

def post_mortem_known_solar(model, model_param_dict:dict, postmortem_next_day:bool)-> Tuple[bool, dict]:
    
    # update json file with history dict
    # "date": date of update of last entry in history list
    # "history" list of int, more recent at the end

    # update blynk: terminal, led based on post mortem history. update color and turn ON, persistant label

    prod_bins = model_param_dict["prod_bins"]
    categorical = model_param_dict["categorical"]
  
    average_inference = config_model.average_inference

    post_mortem_nb = blynk_ev.post_mortem_nb

    s = "post mortem initiated at %s. averaging: %s. postmortem next day: %s" %(datetime.datetime.now().replace(second=0, microsecond=0), average_inference, postmortem_next_day)
    all_log(s, "post_mortem")


    ######################
    # as Blynk is running, update model tab
    ######################
    update_GUI_model_tab() # label and metrics
    

    ##################################################################
    # when we can run postmortem depend on how we get the ground truth
        #   using telemetry, daily production is known at 8pm, and  we can start postmortem "today" at 8pm , as if current day -1 at 8m 
        #   using daily production, need to wait for enphase cloud to be updated, ie start postmortem "tomorrow" at eg 2am, as if current day -2 at 2am
    # NOTE: from user perspective, getting feedack sooner than later is better
    ###################################################################

    # all those date are relative to .now(), ie when the code actually run

    if postmortem_next_day:

        # run next day. 
        #as_if_date = datetime.datetime.now() + datetime.timedelta(-2)  
        #ground_truth_solar_date = datetime.datetime.now() + datetime.timedelta(-1)

        # ACTUALLY, cloud not updated yet, so go one extra day in the past
        # inference_from_web_at_a_day() will not enforce timelimit


        # as_if , ie date at which the inference is ran, to predict next day
        as_if_date = datetime.datetime.now() + datetime.timedelta(-3)  
        ground_truth_solar_date = datetime.datetime.now() + datetime.timedelta(-2)
        
    else:
        # run at same time as daily inference. inference_from_web_at_a_day() enforce time limit
        as_if_date = datetime.datetime.now() + datetime.timedelta(-1)  
        ground_truth_solar_date = datetime.datetime.now() + datetime.timedelta(0)
        

    s = "post mortem, running inference as if: %s (ie predicting the following day) , ie ground truth date: %s" %(as_if_date, ground_truth_solar_date)
    print(s)
    logging.info(s)


    #####################
    # prediction
    #####################

    # run inference as as_if_date (datetime)

    # nb_disconnect is relevant if average = true
    # results_dict has results for both categorical and regression
    ret, results_dict, solar_ground_truth = inference_from_web_at_a_day(model, as_if_date, model_param_dict, average=average_inference, daily_inference = False, postmortem_next_day= postmortem_next_day)
    

    if ret is None:
        s  = "postmortem: error running inference"
        print(s)
        logging.error(s)
        return(False, {})


    # #{"cat": (p1, l1, confidence, nb_disconnect, solar_ground_truth)}
    s = "result dict for postmortem: categorical: %s, regression: %s" %(results_dict[model_type.cat], results_dict[model_type.reg])
    print(s)
    logging.info(s)

    ####################
    # get actual solar production ie ground truth
    ###################

    print("\nget ground truth solar production")

    # if inference_morning, already know, ie last element of production list
    #   NOTE: 1) it is more efficient to get all solar data in one go, 2) doing a subscequent API call seems tp fail

    #                    # WTF: 
    #                    # using same date {'message': 'Unprocessable Entity', 'details': 'Requested date range is invalid for this system. {:start_at=>2024-01-31, :end_at=>2024-01-30}', 'code': 422}
    #                    # using different date: list returned is empty ??? is this because of API rate limit ?

    #                    # anyway, I have already the data IF post mortem is ran early moning next day. this was needed for running inference (I assume model includes solar as input feature)

    #                    #production_list = enphase_API_v4.get_daily_solar_from_date_to_date(ground_truth_solar_date+datetime.timedelta(0), ground_truth_solar_date)
    #                    #assert len(production_list) == 1
    #                    #true_prod = float(production_list[0])  # Wh

    if postmortem_next_day:
        # already known from call to cloud daily API
        # Feb 7th in the morning. as if 4th, to predict 5th.   5th solar is updated in cloud, 6th is not yet
        assert solar_ground_truth != -1
        true_prod = solar_ground_truth

    else:
        # get solar today using telemetry
        # Feb 7th in the evening. as if 6th, to predict 7th.   5th solar is updated in cloud, 6th is not yet , but using telemetry. 
        (wh, telemetry_end_date) = enphase_API_v4.get_telemetry_energy_from_date() # no date ie today
        s = "production today using telemetry %0.1f. telemetry ends %s" %(wh, telemetry_end_date)
        print(s)
        logging.info(s)

        true_prod = wh/1000.0

    s = "ground truth actual solar producted on: %s is %0.1f" %(ground_truth_solar_date, true_prod)
    print(s)
    logging.info(s)

    
    #####################
    # check prediction vs thruth
    # for regression, ie scalar prediction, we use same bins definition as in classification
    #  I guess more relevant than comparing scalar error to some value (eg mae, mse)
    #####################

    # get the indices of the bins to which the each value is belongs
    # convert true production from float into index in bins, to compare with softmax
    # bins: [0, 8.4, 16.7, 22.6, 33]
    # np.digitize([5],bins) array([1], dtype=int64)
    # np.digitize([30],bins) array([4], dtype=int64)
    # index = 1 for first bin

    #### convert true production (scalar) into index_true 

    # led will indicate if prediction is good (ie fit in bins)

    # prod_bins "prod_bins":[0, 6.9, 15.7, 21.9, 1000000],
    inds = np.digitize([true_prod], prod_bins)  # Output array of indices, of same shape as x.
    assert len (inds) == 1
    index_true = inds[0] -1


    #{enum: (p1, l1, confidence, nb_disconnect)}

    if not categorical: 
        # map scalar to bins to create index
        scalar_prediction = results_dict[model_type.reg][0]
        ### convert predicted scalar into index_prediction
        # use same prod_bins as the one used for categorical
        inds = np.digitize([scalar_prediction], prod_bins)  # Output array of indices, of same shape as x.
        assert len (inds) == 1
        index_prediction = inds[0] -1

    else:
        # prediction is already an index
        index_prediction = results_dict[model_type.cat][0]
        prod_lb = results_dict[model_type.cat][1]
        conf = results_dict[model_type.cat][2]

    
    #####################
    # accurate ?
    #####################

    if index_true == index_prediction: # is index returned by softmax the same as the bin index for actual production Wh
        correct = 1    # use int vs Bool, as -1 will mean not initialized in history
        s1 = "SUCCESS"
    else:
        correct = 0
        s1 = "FAILED"

    ############################
    # create postmortem text and log to all
    ############################
        
    # as_if_date is datetime
    # for better clarity, use as_if +1 in text, ie the PREDICTED day
        
    pred_day = as_if_date + datetime.timedelta(1)
    
    m = pred_day.date().month
    d = pred_day.date().day

    # american way month/day
    stamp = "%02d/%02d" %(m,d)

    # NOTE: same text also used for GUI content (not label)
    # time stamp is for predicted day, not as_if day
    # time stamp included, as this is extra context for pushover

    if categorical:
        post_mortem_text = "%s: %s. truth: %0.1fKwh. pred: %s (%d%%)" %(stamp, s1, true_prod, prod_lb, int(conf*100))
    else:
        post_mortem_text = "%s: %s. truth: %0.1fKwh. pred: %0.1fKwh" %(stamp, s1, true_prod, scalar_prediction)

        # led will indicate if prediction is good (ie fit in bins)

    # logging, blynk event, blynk terminal, pushover
    # "post_mortem" is name of blynk event. not used in other cases    
    all_log(post_mortem_text, "post_mortem")


    ###################################################
    # save last n days post mortem in rolling json file.
    # used to update postmorte rolling led
    # led move "to the rigth" when updated (ie left led is most recent)

    # history: list of 0 and 1 , -1 means un initialized
    # date: last date updated , ie the day postmortel is run
    # history[-1] is prediction for yesterday (so that solar is known), and ergo ran as is day before yesterday
    ###################################################

    # date recorded is defined as the as_if_date , ie day for which the inference is ran , with for prediction next day,   
    # not WHEN postmortem is actually ran), neither the next day solar date

    """{
    "as_if_date": "2024-02-05",
    "history": [
        -1,
        -1,
        -1,
        -1,
        1,
        0
    ]
    }"""

    ########################
    # update history json file (for led)
    #######################

    if os.path.exists(config_model.post_mortem_json):

        # get history dict from json
        with open(config_model.post_mortem_json) as f:
            post_mortem_dict = json.load(f)

        history = post_mortem_dict["history"]

        # do not updated twice 
        # date is a str in the dict
        # as_if_date.date() is datetime.date(2024, 2, 5)

        if post_mortem_dict["as_if_date"] == str(as_if_date.date()):
            # no need to update history dict, already updated for that day
            s = "post mortem json file already updated for inference date: %s" %as_if_date.date()
            print(s)
            logging.info(s)

        else:
            # update history dict.
            s = "update post mortem json file for inference date: %s" %as_if_date.date()
            print(s)
            logging.info(s)

            # list
            history.pop(0)
            history.append(correct)  # updated from the end , <=====

            assert len(history )== post_mortem_nb

            post_mortem_dict["history"] = history
            post_mortem_dict["as_if_date"] = str(as_if_date.date())


    else:
        # create new history dict 
        s = "create post mortem json file"
        print(s)
        logging.info(s)

        # create json file
        # 
        history = [-1] * post_mortem_nb # -1 is uninitialized

        # updated from the end , <=====
        history[-1] = correct # integer, 0, 1 or -1

        # store date as str

        post_mortem_dict = {
            "as_if_date": str(as_if_date.date()),
            "history" : history
        }

    # save history dict to json
    s = "post mortem dict: %s" %str(post_mortem_dict)
    print(s)
    logging.info(s)

    # save dict to json file
    j = json.dumps(post_mortem_dict, indent = 4)
    with open(config_model.post_mortem_json, "w") as f:
        f.write(j)
        #or json.dump(post_mortem_dict, f)



    ####################
    # postmortem accuracy so far 
    # cummulative since the json file is created
    # manage json (creates, update)
    ####################
        
    update_json = False
        
    if os.path.exists(config_model.post_mortem_so_far_json):

        # get current dict from json
        with open(config_model.post_mortem_so_far_json) as f:
            so_far_dict = json.load(f)

        # be carefull to use same keys when creating
        total = so_far_dict["total"]
        accurate = so_far_dict["accurate"]
        d = so_far_dict["date"] # last updated



        # only update if not already (ie running postmortem several time the same day)
        if d == str(datetime.date.today()):
            s = "postmortem accuracy cummulative %s already updated" %config_model.post_mortem_so_far_json
            print(s)
            logging.info(s)
        
        else:

            s = "update postmortem accuracy cummulative %s" %config_model.post_mortem_so_far_json
            print(s)
            logging.info(s)

            total = total + 1
            if correct == 1:
                accurate = accurate +1
            d = str(datetime.date.today())
            update_json = True
 
    else:
        # create new history dict 
        s = "create new post mortem accuracy so far json file"
        print(s)
        logging.info(s)

        total = 1
        if correct == 1:
            accurate = 1
        else:
            accurate = 0

        d = str(datetime.date.today())
        update_json = True



    if update_json:
        # save updated/created to json    
        
        so_far_dict = {
                "total": total,
                "accurate" : accurate,
                "date": d
        }

        s = "update post mortem so far to json: %s" %str(so_far_dict)
        print(s)
        logging.info(s)

        # save dict to json file
        # write whole dict each time
        j = json.dumps(so_far_dict, indent = 4)
        with open(config_model.post_mortem_so_far_json, "w") as f:
            f.write(j)
            #or json.dump(post_mortem_dict, f) 

        

    ################################
    # GUI: 
    # accurate in % and total
    # update GUI, even if json was not updated 
    ################################
        
    a = int(100 * accurate / total)
    s = "%d%% accurate so far (%d/%d)" %(a, accurate, total)
    blynk_ev.blynk_v_write(vpins.v_post_mortem_so_far, s)
    blynk_ev.blynk_color_and_label(vpins.v_post_mortem_so_far, blynk_ev.color_text, "accurate so far %") # Green


    ################################
    # GUI: 
    # terminal 
    ################################

    # history to terminal, debug
    s = "postmortem history: " + post_mortem_dict["as_if_date"] + " " + str(post_mortem_dict["history"])
    blynk_ev.blynk_v_write(vpins.v_terminal, s)

    ################################
    # GUI: 
    # led based on post mortem history. update color and turn ON
    ################################
    # history is mapped to led (one to one)
    # MAKE SURE GUI is consistent with history. there is NOTHING the code can do here

    for i, correct in enumerate(history):

        # post_mortem_led sorted from least recent to more recent, ie same order as history
        
        led = blynk_ev.post_mortem_led[i]  
        label = blynk_ev.post_mortem_led_label[i]

        # set color and then turn ON
        # color is the result. led always turned on
        if correct == 1: # means correct
            blynk_ev.blynk_color_and_label(led, "#00FF00", label) # Green
        if correct == 0: 
            blynk_ev.blynk_color_and_label(led, "#FF0000", label) # Red
        if correct == -1: 
            blynk_ev.blynk_color_and_label(led, "#0000FF", label) # not initialized. #FFFFFF is white

        blynk_ev.blynk_v_write(led,1)

    ################################
    # GUI: 
    # persistant label
    ################################

    # GUI text
    # persistant labels, cannot use \n in labels 
    # clear and concise text
    

    blynk_ev.blynk_v_write(vpins.v_post_mortem_label, post_mortem_text)

    blynk_ev.blynk_color_and_label(vpins.v_post_mortem_label, blynk_ev.color_text, "postmortem predicting %s" %(stamp))


    #####################
    # postmortem history file
    # eg for offline analysis
    #####################
    #postmortem_history_hdr = ["date", "cat_prediction", "scalar_prediction", "truth", "confidence", "disconnect"]
    # result_dict
        # index, str, conf, -1/int
        # float, str(float), -1, -1/int
    
    print("update postmortem history (eg for offline analysis): %s" %postmortem_history_csv)
    
    cat_results = results_dict[model_type.cat]
    reg_results = results_dict[model_type.reg]
    
    row1 = [datetime.date.today(), cat_results[1], reg_results[0], true_prod, cat_results[2], cat_results[3]]
    assert len(row1) == len(postmortem_history_hdr)

    utils.append_to_history_csv(row1, postmortem_history_csv, postmortem_history_hdr)


    return(True, results_dict)


    
####################################
# update model tab in GUI
# info and metrics gauges from json
####################################
def update_GUI_model_tab()-> None: 

    # called from predict_solar_tomorrow() and post_mortem_known_solar(), and unseen
    # model 
        
    # metrics from json file created at training time (ie do not run Blynk when training, kind of overkill)

    print("update GUI model tab (info and metrics gauges)")
    # metrics_dict used for gauge
    # text and label for persistant value display

    label = "current model"
    # would be good to have context about the model, ie model size, number of days
    # do it brutaly, re open df_model
 
    # could have been extended with retrain
    df_model =  pd.read_csv(config_features.features_input_csv)
    nb_days = len(df_model) / 24
    del(df_model)

    last_date_trained = dataframe.get_last_cvs_date(config_features.features_input_csv) # datetime
    last_date_trained = last_date_trained.date() #make it short

    # use param count from "default model"
    #_ = model.count_params()

    text = "trained on %d days (last %s)" %(nb_days, last_date_trained) 

    # update display and gauges  widget

    ##### persistant label
    blynk_ev.blynk_v_write(vpins.v_model_info, text)
    blynk_ev.blynk_color_and_label(vpins.v_model_info, "#A0A0A0", label) #FFFFFF is white

    #############################
    # update gauge with metrics from json (updated after training (.evaluate, sklearn) , see train_and_assess.metric_to_json())
    #############################

    # get metrics for both model from json

    if not os.path.exists(config_model.metrics_json):
        s = "%s does not exist, cannot update metrics gauge" %config_model.metrics_json
        print(s)
        logging.error(s)

    else:
        s = "%s exist, update metrics gauge" %config_model.metrics_json
        print(s)
        logging.info(s)

        with open(config_model.metrics_json) as f:
            x = json.load(f)

            reg_dict = x[model_type.reg.value]  # model_type.reg is enum. used value (str) when storing as json {'mae': 2.581, 'rmse': 3.946}
            cat_dict = x[model_type.cat.value] # {'categorical accurary': 0.834, 'precision': 0.83, 'recall': 0.828}

            ##### metrics gauge defined as 0 to 100

            # make sure the keys are not mistyped
            acc = int(cat_dict["categorical accuracy"]*100)
            pre = int(cat_dict["precision"]*100)
            rec = int(cat_dict["recall"]*100)

            blynk_ev.blynk_v_write(vpins.v_model_acc, acc)
            blynk_ev.blynk_color_and_label(vpins.v_model_acc, "#00FF00", "accuracy")

            blynk_ev.blynk_v_write(vpins.v_model_pre, pre)
            blynk_ev.blynk_color_and_label(vpins.v_model_pre, "#00FF00", "precision")

            blynk_ev.blynk_v_write(vpins.v_model_rec, rec)
            blynk_ev.blynk_color_and_label(vpins.v_model_rec, "#00FF00", "recall")

            rmse = reg_dict["rmse"]
            #mse = reg_dict["mse"]
            mae = reg_dict["mae"]

            blynk_ev.blynk_v_write(vpins.v_model_rmse, rmse)
            blynk_ev.blynk_color_and_label(vpins.v_model_rmse, "#00FF00", "rmse")

            blynk_ev.blynk_v_write(vpins.v_model_mae, mae)
            blynk_ev.blynk_color_and_label(vpins.v_model_mae, "#00FF00", "mae")

            # update mse, but maybe not a good idea to show it (in kwh2, not kwh)
            #blynk_ev.blynk_v_write(vpins.v_model_mse, mse)
            #blynk_ev.blynk_color_and_label(vpins.v_model_mse, "#00FF00", "mse")


