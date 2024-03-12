#!/usr/bin/env python3

import os

app_name = "solar2ev"

############################
# features
############################

days_in_seq = 4

# regression (ie predict scalar) or classification (use bins definition in config_features.py)
# NOTE: for scalar, 2 methods to assess accuracy:
#  compare to configured max acceptable absolute error (mae)
#  map scalar preductions to same bins used for categorical
categorical = True

############################
# dataset
############################

repeat = 1 # dataset repeat before fit. artificial increase of samples. OVERFIT

shuffle = True # shuffle list of sequences (within one sequence , temporal informations is maintained)

# interval between time steps in sequence. increasing will decreases sequence len, 
# eg 4 days, sampling = 1 24*3 + 18 = 90
sampling = 1 

seq_build_method = 1
# METHOD 1
# tf.keras.utils.timeseries_dataset_from_array
# simplest, but no way to only select sequences which start around midnigth
# use stride. eg stride = 24 returns one sequence per day (starting at 0h)

# METHOD 2
# use tf data .window()
# added logic to select sequences of interest, eg sequences starting "around" 0h
# uses selection
# !!!!! Somehow deprecated. replaced by method 3

# METHOD 3
# manual, process numpy 
# has to fit in memory, but not a problem
# uses selection, ie allow to specify starting hour for input
# does not use stride
# can be used in replacement of method 1

# method 3 with selection [0] and stride 1 is equivalent to method 1 with stride 24
# selection shared by method 2 and method 3
# selection not used in method 1

# both stride or selection drives the number of available sequences (training samples)

# ONLY USED for method 1. ignored for method 2,3
# 1: sequences starts every hours.  24: sequences starts every midnigth 
stride = 1  # interval between start of sequences. increasing decreases number of sequence available for training

# ONLY USED for method 2 and 3. stride ignored
selection = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8] # only use sequences starting at those hours.
#selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # only use sequences starting at those hours.


assert sampling % 2 == 0 or sampling == 1 , "sampling must be 1 or multiple of 2"
assert stride <= 24 , "stride must less than 24"

# "undefine" stride and selection when not used
if seq_build_method in [2,3]:
    stride = -1

if seq_build_method in [1]:
    selection=[]

############################
# model
############################

#use_cudnn = False  # no perf improvement  .  keras automatically use cudnn implementation 
#use_bidirectional = True  #no perf improvement
use_attention = False # does not seems to improve much

nb_lstm_layer = 1
nb_unit = 384
nb_dense = 128

#use_dropout = True   deprecated, use dropout_value = 0 to disable dropout (easier for search space)
dropout_value = 0.3

####################
# training
####################

batch_size = 16
epochs = 40
split=(0.8,0.1) # train valid

######################
# sdv training
# "retrain", "continue", "freeze" 
######################

sdv_train = "retrain" # concatenate synthetic dataset and real data dataset in one dataset and train from scratch
# "continue". train on one dataset and continue training on the other one. (should be equivalent to train from scratch)
# "freeze". transfert learning, ie train on real and finetune. NOT IMPLEMENTED


######################
# inference
#####################

average_inference = False
# average multiple input for inference (daily or postmortem)
# ie instead of running ONE inference on a sequence starting day -n at 0h, run multiple starting day-(n+1) 23h, 22h 
# calculate disconnect, ie not the same as "average"

# used in predict_solar_tomorrow() and post_mortem_prev_day() , both calling inference_from_web_from_a_given_day()


#################
# others
#################

# store date in pandas. template to convert from str to datetime
format_date = '%Y-%m-%d'

# validation split. NOT USED YET
#validation_choices = ["fixed", "kfold", "timeseriesplit"]
#validation = "fixed"
#if not validation in validation_choices:
#    raise Exception  ("%s incorrect validation" %validation)

# WARNING: RETAIN ALL. not used anymore. get all hours from pandas and subsample in tf.data (vs subsampling in pandas)
retain = [23, 22, 21, 20, 19,18, 17, 16, 15, 14, 13, 12, 11, 10, 9 , 8, 7, 6,  5, 4, 3, 2, 1, 0]
assert len(retain) == 24

# type of keras tuner   "random", "hyperband"
kt_type_list = ["random" , "hyperband" , "bayesian"]
kt_type= "hyperband"
kt_type= "bayesian"
kt_type= "random"


# comprehensive check of timing data in df_model. set to True if paranoiac, or just got a new set of data. False speed loading a bit
# also used for df_model created for inference
assert_timing = True

# file name used in multiple modules

sdv_dir = "sdv" # store all sdv artifacts 

if os.path.exists(sdv_dir):
    pass
else: 
    os.makedirs(sdv_dir)

# synthetic generated by sampling from trained sdv model 
# will be saved in sdv
# run synthetic.py as main to generate
# file used by solar2ev to train
synthetic_data_csv = os.path.join(sdv_dir,"synthetic_data.csv")

# save previous daily inference for showing post mortem in GUI, rolling list
post_mortem_json = "post_mortem_history.json"

# all postmortem so far, % accurate
post_mortem_so_far_json = "post_mortem_so_far.json"

# save metrics (dict from .evaluate) for both catagorical and regression model
metrics_json = "metrics.json"

# if <=, start date too close to end date. do not create delta and do not update/create feature untrain#ed
# no need to retrain for such a small delta
min_unseen_days = 2

# downloaded csv. fix values below the min (including zero, which could be caused by internet down for several days)
enphase_interpolate = True
enphase_min_fix = 0.1 






