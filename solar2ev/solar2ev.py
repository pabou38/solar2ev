#!/usr/bin/env python3

#######################################################
# to run on windows

# from anaconda powershell prompt (miniconda3)
# > conda activate tf
# > code

# to start in a Linux virtual environment
"""
#!/usr/bin/env bash
echo 'start solar2ev within (tf) environment. arguments: ' $1
dir="/home/pi/APP"

cd $dir
set -e
source "./tf/bin/activate"

cd $dir/solar2ev
python -u solar2ev.py $1
"""
########################################################

version = 3.2
version = 3.3 # fev 2024
version = 3.31 # 20 fev 2024. running on Linux (PI)
version = 3.32 # add postmortem history csv. secret move to all_secret/my_secret. GUI finetuning
version = 3.34 # add postmortem accurate so far, add header vpin (template, app)
version = 3.35 # a lot of plot cleaning
version = 3.36 # precision, recall from sklearn
version = 3.37 # analyze confusion matrix
version = 3.38 # enphase, interpolate production (internet down)
version = 3.39 # deep copy
version = 3.40 # sdv
version = 3.41 # clean systemd config files
version = 3.42 # beautifulsoup4 jetson
version = 3.43 # sdv, freeze LSTM
version = 3.44 # automate sdv exploration

# https://medium.com/@shouke.wei/a-practical-example-of-transfer-learning-for-time-series-forecasting-9519f3e94087
# https://medium.com/@shouke.wei/a-stacked-lstm-based-time-series-model-for-multi-step-ahead-forecasting-a387e4020faf


#pip install py-spy #to profile

# import early,  cannot allocate memory in static TLS block on Jetson
from sklearn.metrics import classification_report # precision, accuracy

import sys
import datetime

# print early for timestamp in std output.
# will go in stdout looging to file
print("%s: version %0.2f" %(datetime.datetime.now(), version))

import my_arg

#######################
# parse arguments early
#######################
try:
  arg = my_arg.parse_arg()
  print("arguments:",  arg)

except Exception as e:
  print('exception parsing argument:', str(e))
  # when running with vscode remote container from DEEP, launch.json is in DEEP/.vscode


######################################
# run code from conda env to get GPU in vscode. will run conda active tf 
# having vscode python being 3.9.18("tf": conda) is not enough
# ######################################

# see vscode setting.json "python.defaultInterpreterPath": "/home/pi/tf/bin/python3.9"
# !! python.pythonpath deprecated
print('python executable:\n', sys.executable)  
# /bin/python3 when running in VScode remote container
# 'C:\\Users\\pboud\\miniconda3\\envs\\tf\\python.exe'

print('import:\n', sys.path)

import tensorflow as tf
try:
    print("tf: ", tf.version.VERSION) # bug in tf 2.11, ok with 2.10.  tensorflow.__version__ works on colab 2.11 .
except:
    print("tf: ", tf.__version__)

# make sure runs yields same results. can compare
tf.random.set_seed(1234)

# Caution: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11,
#  you will need to install TensorFlow in WSL2, or install tensorflow-cpu 
# Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at 
# https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.Skipping registering GPU devices...
# The nvcc compiler driver is installed in /usr/local/cuda/bin, 
# and the CUDA 64-bit runtime libraries are installed in /usr/local/cuda/lib64.

print("make sure tf runs on GPU: ", tf.config.list_physical_devices('GPU')) 
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


print('GPU name: ', tf.test.gpu_device_name())
# deprecated, but shows compute capability
# '/device:GPU:0'
# 2023-12-11 06:38:28.535447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] 
# Created device /device:GPU:0 with 5412 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4070 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9

print('built with CUDA? ', tf.test.is_built_with_cuda())

# pip install scikit-learn
#from sklearn.model_selection import KFold
#from sklearn.model_selection import TimeSeriesSplit
#from sklearn.model_selection import train_test_split

# test normality
from scipy.stats import shapiro

from bs4 import BeautifulSoup

from calendar import monthrange
from time import sleep, time, perf_counter
import datetime
import os
import logging
import sys
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


print( "system:", platform.system())
print( "processor:", platform.processor())
print( "machine:", platform.machine())
print( "system:", platform.system())
print( "version:", platform.version())
print( "uname:", platform.uname())


# bool to determine if we are running on the Jetson/PI 
# eg do not evaluate model after loading, 
running_on_edge = platform.processor in ["aarch64"]

if not running_on_edge:

    # pip install keras-tuner 
    import keras_tuner as kt # cannot import name 'keras' from 'tensorflow' with tf 2.11  2.10 ok
    print("kt: ", kt.__version__)

    import sdv as sdv
    print("sdv: ", sdv.__version__)

else:
    print("\n\nrunning on edge\n\n")


from typing import Tuple
# -> Tuple[int, int]
# since 3.9 or 3.10 you can just outright say tuple[int, int] without importing anything.
# function always returning one object; using return one, two simply returns a tuple.
# -> Union[int, Tuple[int, int]]   can returns different type

#################################
# logging
# https://docs.python.org/3/howto/logging.html
#################################

root = './'
# debug, info, warning, error, critical
log_file = root + "solar2ev.log"
print ("logging to:  " , log_file)

if os.path.exists(log_file) == False:
    with open(log_file, "w") as f:
        pass  # create empty file

# The call to basicConfig() should come before any calls to debug(), info()
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages
    
# encoding not supported on ubuntu/jetson ?
try:
    logging.basicConfig(filename=log_file,  encoding='utf-8', format='%(levelname)s %(name)s %(asctime)s %(message)s',  level=logging.INFO, datefmt='%m/%d/%Y %I:%M')
except:
    try:
        logging.basicConfig(filename=log_file, format='%(levelname)s %(name)s %(asctime)s %(message)s',  level=logging.INFO, datefmt='%m/%d/%Y %I:%M')
    except Exception as e:
        print(str(e))
        sys.exit(1)

# define a name (used in %(name)s )
# name are any hiearchy
# use root if not defined (ie use logging.info vs logger.info )
# importing logging in all modules easier

logger = logging.getLogger(__name__) # INFO __main__ 2024-02-01 08:52:45,142 logger defined with name
logger.info("logger defined with name")

# https://docs.python.org/3/howto/logging.html#logging-from-multiple-modules



s =  '-------------- solar2ev starting v%0.2f ...............' %version 
print(s)
logging.info(s)  # INFO root 2024-02-01 08:52:46,327 -----

p = "../PABOU"
sys.path.insert(1, p)
try:
    import pabou
    import various_plots
    #print('pabou path: ', pabou.__file__)
    #print(pabou.__dict__)
    print("import from %s OK" %p)
except:
    print('%s: cannot import modules from %s. check if it runs standalone. syntax error will fail the import' %(__name__, p))
    exit(1)


p = "../my_modules"
sys.path.insert(1, p)
try:
    import pushover
except:
    print('%s: cannot import modules from %s. check if it runs standalone. syntax error will fail the import' %(__name__, p))
    exit(1)

p = "../all_secret"
sys.path.insert(1, p)
try:
    import my_secret
except:
    print('%s: cannot import modules from %s. check if it runs standalone. syntax error will fail the import' %(__name__, p))
    exit(1)


###### import application module
import my_arg
import model_solar
import config_features
import config_model
import record_run
import dataset
import dataframe
import feature_plot
import train_and_assess
import inference
import utils
import charger
import vpins
import blynk_ev
import tuner 

import shared


##### site specific, ie car charger, solar hardware and meteo web site
import meteo
import enphase
import vpins

# to save model, weigths, json
models_dir = pabou.models_dir


###### housekeeping
print('current directory: ', os.getcwd())  # /workspaces/DEEP when running in VScode remote container

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

########################################
# key directory, file name and config
########################################
app_name = config_model.app_name

# 2 config files: config_features and config_model

#############
# user level config
#   set input features, bins, skipped hours and charge
#   use blynk, default charge mode, confidence
# #############

input_feature_list = config_features.config["input_feature_list"] # select series for model input(s). handle multi head

# we want to run prediction before 23h, so last day will have fewer hours
# skipped 6: need to scrap 0 to 17
# meaning we need to wait until 17h, 5pm included (ie allow time for site to update), ie skip last 6 hour
today_skipped_hours = config_features.today_skipped_hours 

#max_expected_production = config_features.max_expected_production
# need to be able to distinghish 34 as a valid production ie 34kwh, from 36 a production issue, ie 36w
# this value need to be set manually
# 29 Kwh seems the max with current pannel configuration. 31kw (normalized) 16 juin 2021
# all value above are consider w (not kw) and divided by 1000
# deprecated

prod_bins=config_features.config["prod_bins"]

# cluster solar production into those brackets. 
# "prod_bins":[0, 5, 10, 15, 20, 25, max_expected_production],

prod_bin_labels = [i for i in range(len(prod_bins)-1)]  # to be converted to onehot [0, 1, 2, 3, 4, 5]
print('panda bin labels. to be converted to onehot', prod_bin_labels)
# used by pandas.cut to associate each production to a bracket (bin).
# labels associated to each bin. will be converted to one hot

prod_bins_labels_str = config_features.config["prod_bins_labels_str"]
# human display string

charge_hours = config_features.config["charge_hours"]
# number of hours to charge overnigth for each next day solar production's prediction

assert len(prod_bins_labels_str)  == len(prod_bin_labels)
assert len(charge_hours) == len(prod_bin_labels)
assert len(prod_bins) == len(prod_bin_labels) +1


############
# developpr
#  features, model
#############

retain = config_model.retain

epochs=config_model.epochs
batch_size = config_model.batch_size  # one dataset element = one batch of samples

###### param to model definition
nb_lstm_layer = config_model.nb_lstm_layer
nb_unit = config_model.nb_unit
nb_dense = config_model.nb_dense

use_attention = config_model.use_attention

dropout_value = config_model.dropout_value

categorical = config_model.categorical

# file/dir name to save

#############################
# for all options expect daily inference , we use only one model , based on categorical:bool
# we assumes both model (categorical/regression) are created (by running the app twice)
# for daily inference, the "other model" is loaded and used as well

model_name_cat = app_name + "_cat"
model_name_reg = app_name + "_reg"

if categorical:
    model_name = model_name_cat
else:
    model_name = model_name_reg


# used by pabou.bench_full() and inference.predict_on_ds_with_labels()
# NOTE: for scalar, 2 methods to assess accuracy:
#  compare to configured max acceptable absolute error (mae)
#  map scalar preductions to same bins used for categorical
acceptable_absolute_error = config_features.regression_acceptable_absolute_error

# based on categorical (classification) or regression
# watch used for early stop and reduce lr callbacks
(watch, metrics, loss) = model_solar.get_metrics(categorical)

###### param to dataset preparation
split=config_model.split
# train, val, rest is test. note: int is abs, so 39 give 31,3,5

repeat = config_model.repeat
# ds.repeat() before training, to increase sample count. test set not repeated

shuffle = config_model.shuffle 

days_in_seq = config_model.days_in_seq   
# input is sequence of 3 days, possibly including today (last day truncated a bit to not to have no wait midnigth to run inference)
# meteo from day -2 to 0 used to  predict day +1 solar production

stride = config_model.stride
#stride: interval between start of sequences , ie do seq starts every 0h, or each hours , etc ..
#stride = len(retain) # each seq starts at one day boundary.

sampling = config_model.sampling 
#sampling rate: interval between individual timesteps, used for subsampling hours in sequence

seq_build_method = config_model.seq_build_method

selection = config_model.selection

# hot size and seq len from config
hot_size = len(prod_bin_labels)  # len of softmax, aka one hot
seq_len = len(retain) * days_in_seq - today_skipped_hours # input is n days x temp, or [temp, humid] , etc .. last day ends early to be able to run inference late in afternoon for next day
seq_len = int(seq_len / config_model.sampling) # use for range, so must be int

print("hotsize, sequence length:" , hot_size, seq_len)


######
# models
# training artifacts stored there. result from training, sdv
######

models_dir = pabou.models_dir

# training results
history_training_csv = "history_training.csv" # accumulated result of one training 


######
# tmp
######

tmp_dir = utils.tmp_dir
if os.path.exists(tmp_dir):
    pass
else: 
    os.makedirs(tmp_dir)
	
# evaluate metrics vs baseline, # result of various model.evaluate(ds)
eval_csv = os.path.join(tmp_dir, "baseline_eval.csv")

# to build delta
unseen_prod_csv = os.path.join(tmp_dir, "unseen_energy.csv")
unseen_meteo_csv = os.path.join(tmp_dir, "unseen_meteo.csv") 

# concatenate of unseen and few days to build sequences on unsee data
# build when option unseen used
unseen_csv = os.path.join(tmp_dir, "unseen_plus_few_days.csv")

######
# meteo and solar
######

production_file = enphase.production_file # downloaded from enphase cloud
meteo_file = meteo.meteo_file  # scrapped from web
header_csv = meteo.header_csv  # for scrapped meteo csv
production_column = enphase.production_column # added when creating input feature. used in config as well

month_name= meteo.month_name # list to convert int to str
installation = enphase.installation  # installation day. ie 1st day of complete production (ignore 1st partial day)

###########
# pandas features
###########

# created after cleaning and combining meteo and solar. 
# file interface to use this application with different meteo and solar data sources
# final pandas dataframe contains additional serie: "month" and "production"
# month is sin()
#features_input_csv = arg["features"]  # FILE NAME
features_input_csv = config_features.features_input_csv  # available from different modules

# increment from last data used in training
# created or extended each time retrain or test on unseen is done
# retrain: concatenated with trained data to created updated feature input  and then retrain
# test on unseen: concatenated some trained data (previous days) to created a feature. create ds and test prediction
features_input_untrained_csv = "features_input_untrained.csv"

hot_size = len(prod_bin_labels)  # len of softmax, aka one hot

# compute len of sequence
seq_len = len(retain) * days_in_seq - today_skipped_hours # input is n days x temp, or [temp, humid] , etc .. last day ends early to be able to run inference late in afternoon for next day
seq_len = int(seq_len / config_model.sampling) # use for range, so must be int

print("hotsize, sequence length:" , hot_size, seq_len)

# no processing (beside init variables) before here

#########################################################################
################### COMMAND LINE ARGUMENTS PARSING ######################
#########################################################################

# start GUI

##### features #######
# if get_data: scrap, combine, save as csv
# else: read from csv

# plot
# if unseen/retrain: create delta features
# if retrain: concatenate features
# energy distribution

##### dataset #######
# create dataset
# validation split
# histograms (dataset)

##### training #######
# if train/retrain/vault: create evaluation dataframe

# if tuner: 
#    search and train on best model . EXIT

# else if train/retrain/vault:  (ie all case doing "manual training")
#    if retrain/vault, get previous metrics
#    train, evaluate, predict on ds, plot, update evaluate dataframe, save, upate record run
#    if retrain: update gui
#    if vault: evaluate on real test data

# else 
#     load trained model, evaluate, predict on ds, update GUI 

##### process #######
#inference: (18h)
#charge: (23h)
#postmortem (4h)

##### operation #######
#bench
#unseen (only if retrain false)

#train/retrain/vault: plot evaluate dataframe

#GUI: sleep
#END

#################################
# GOAL: start blynk for all options which requires updating or reading from GUI
#################################

use_GUI = arg['inference'] or arg["postmortem"] or arg["charge"] or arg["unseen"] or arg["test_blynk"]

# NOTE: not started for training, vault, tuner, retrain, benchmark
# this means retrain improvement cannot be communicated, and possibly a less accurate model will overwrite the current one ?

if use_GUI:

    # those will interact with GUI
    # starts blynk in separate thread, so that call backs and virtual write can happens
    # will then proceed to separate processing for each options
    # some sleep at the end for those, to make sure blynks run. Then exit

    ###########
    # blynk GUI ; is optional
    ###########

    if config_features.blynk_GUI:

        s = "app requires Blynk. connecting to blynk"
        print(s)
        logging.info(s)

        # connect (create blyn_con)
        # start blynk.run() thread
        # define call back
        # wait for connect call back
        # send event
        # sync
        # clear all prediction led

        ret = blynk_con = blynk_ev.create_and_connect_blynk()
        # note blynk_con (ret) not used, as all modules (including main) uses blynk_write and blynk_color (uses blynk_ev global blynk_con)
        
        if ret is None:
            s = "cannot connect to Blynk. EXIT"
            print(s)
            logging.error(s)
            sys.exit(1)

        # time for Blynk to start and get call backs 
        sleep(5)

        # set global from do_blynk_ev name space, could be needed by call backs, eg refresh 
        # could have also hardcoded tho values or created a commun file to import   
        post_mortem_json = config_model.post_mortem_json
        post_mortem_nb = blynk_ev.post_mortem_nb

    else:
        s = "blynk not enabled in config file"
        print(s)
        logging.info(s) 

else:
    s = "command line parameters do not need blynk"
    print(s)
    logging.info(s)


###############################
# GOAL: get a set of input features (df_model)
##############################

# case 1: get_features (bootstrap) to build initial df_model, ie starts from scratch
# case 2: load existing df_model from csv file (feature_input.csv)

######################################
# case 1: bootstrap by doing intial meteo scrap and combine meteo and solar into df_model
######################################
if arg['get_features']:

    ##################################### 
    # build INITIAL feature input csv

    # clean/postprocess meteo csv into df
    # clean solar csv into df
    # combine meteo and solar df into feature input: df_model
    # save df_model as csv

    # analyze solar (quartile, histograms) from dataframe
    ######

    print("\nbuild initial features from scrapped meteo and downloaded production")

    ##################
    # meteo 
    ##################

    # do not scrap if already exist
    # this initial file is time consuming to build

    if os.path.isfile(meteo_file):
        print('!!!! meteo file: %s exist. MAKE SURE VALID, OR run scrapping offline to update' %meteo_file)

        df =  pd.read_csv(meteo_file)

        # check timing
        dataframe.assert_df_timing(df)
       
    else:

        ## offline scrapping 
        print('%s file do not exist. run scrapping offline and come back' %meteo_file)
        sys.exit(1)


    ##################
    # solar
    ##################

    # solar csv is supposed to be downloaded already (from emphase cloud)
        
    if os.path.isfile(production_file):
        print('!!!! production file: %s exist. MAKE SURE VALID, OR download again from enphase cloud' %production_file)

    else:

        print('%s file do not exist. download from enphase cloud and come back' %production_file)
        # https://enlighten.enphaseenergy.com/systems/xxxxxxx/reports
        sys.exit(1)


    ##################
    # clean meteo and solar (load from csv, create dataframe)
    ##################
    print('clean/postprocess meteo and solar csv to create %s' %(features_input_csv))

    ##### create CLEAN Pandas dataframe from meteo csv
    df_meteo = dataframe.clean_meteo_df_from_csv(meteo_file) # subsampled

    #####################
    # test normality of meteo
    #####################
    for x in ["temp", "pressure", "humid"]:

        (_,p) = shapiro(df_meteo[x])
        if p<0.05:
            print("%s is not normal" %x)
        else:
            print("%s is normal" %x)
    
    
    # should not have changed
    dataframe.assert_df_timing(df_meteo) # 

    ##### create CLEAN Pandas dataframe from solar production csv
    # outliers, normalize, build fixed width histograms after/before normalization. 
    # plot histo (pd, subplot before, after)
    df_prod , max_prod = enphase.clean_solar_df_from_csv(production_file,  plot_histograms=True)  
    print("downloaded production csv cleaned. max prod %0.2f" %max_prod)

    ##################
    # analyze solar dataframe
    # quartile, 1D histogram, hexagonal histograms
    ##################

    print("\nanalyzing solar downloaded from enphase cloud: quentile, plot")

    # observe quartile to configure bins to get same number of samples per output bucket
    # NOTE: this is only called when building the initial feature

    enphase.plot_downloaded_solar(df_prod)
 
    #####################
    # test normality of solar AFTER normalization
    #####################
    
    x = "production"
    (_,p) = shapiro(df_meteo[x])
    if p<0.05:
        print("%s AFTER normization is not normal" %x)
    else:
        print("%s AFTER normalization is normal" %x)



    ##################
    # combine dataframe into df_model (feature input)
    ##################

    ##### COMBINE meteo and solar production, to create input features

    print("\ncombine meteo and solar into df_model, ie feature input")

    # columns names automatically generated
    df_model = dataframe.create_feature_df_from_df(df_meteo, df_prod, production_column ) # associate solar production to meteo data (same day, all day's row have same production)

    # available to other modules
    # set when initially creating df_model, or when loading from csv
    shared.feature_input_columns = list(df_model.columns)

    # integrity check. timing alone
    dataframe.assert_df_timing(df_model, s = "timing for created feature input/df_model")

    start_date = df_model.iloc[0][0] # already timestamp
    end_date = df_model.iloc[-1][0]

    delta = end_date - start_date
    assert delta.days == len(df_model) / len(retain)  -1  #  2 samples, delta = 1
    nb_days = delta.days+1

    ##### SAVE as csv file . Nan are saved as empty cells
    # file interface to use this application with other meteo and solar sources
    # date format should be consistent with delta and get_last_date
    df_model.to_csv(features_input_csv, header=True, index=False, date_format='%Y-%m-%d') # header bool or list of str

    print("\n====> initial df_model ready and saved. %d days. start date: %s. end date: %s. %d rows" %(nb_days, start_date, end_date, len(df_model)))
    
    #################################### df_model, aka feature input ready ##############################


    # df_model input feature ready


#######################################################
# case 2: production (ie running inference every day) case. 
# load existing df_model
#######################################################
else:

    #### load existing df_model, feature input csv
    print('\nload saved df model/feature input from csv %s' %features_input_csv)
    df_model =  pd.read_csv(features_input_csv)

    shared.feature_input_columns = list(df_model.columns)

    #####################
    # integrity of loaded df_model
    # timing
    # remove to speed loading
    #####################
    if not running_on_edge:
        print("not running on edge, check integrity of loaded df_model")
        dataframe.assert_df_model_integrity(df_model, s="loaded df_model from csv") # as we just set colums, will only validate timing
    else:
        print("running on edge, skip check integrity of loaded df_model to speed up")


    # look at statictics of key columns. handy for sdv constraints
    print("some stats on loaded df_model (eg for sdv constraints)")
    for c in df_model.columns:
        if c in ["production", "temp", "pressure"]:
            d = df_model[c]
            print("     %s. max %0.1f, min %0.1f, median %0.1f mean %0.1f" %(c, d.max(), d.min(), d.median(), d.mean()))


    # in case the model was saved without index=False, and an unmaned column is created
    #x  = df_model.drop(df_model.columns[0], axis=1)
    #x.to_csv(features_input_csv, index=False, header=True)

#############################################################
# end of case 1,2
# df_model is loaded, from get_data or loaded from csv
#############################################################

## df_model can be extended in case of retrain
## dataset can be extended in case of vault

############################################
# GOAL: create various plots to analyze input features 
# df_model, ie input features
############################################
if arg["plot"]:
    print("\ncreate various plots for input features/training data (ie from df_model)")
    feature_plot.plot_input_features(df_model)


####################################
# GOAL: create/update (ie accumulate into) : features_input_untrained_csv
# NOTE: actual prediction comes later (as retrain takes priority)
#####################################
if arg["unseen"] or arg["retrain"]:

    # build df_untrained and features_input_untrained_csv

    # used later for:
    #  1- unseen (test existing model on unseen features):
    #    concatenate with a few days from existing features (just enough to create sequences)
    #    create dataset from dataframe (ds with labels) and call .predict()
    
    #  2- retrain (from scratch, with combination of existing and unseen features):
    #    concatenate with existing feature (and delete features_input_untrained_csv), then call train

    print('\nUNSEEN or RETRAIN: updating (or creating): %s' %features_input_untrained_csv)

    # creates df_delta, 
    #  starting from previous features_input_untrained_csv or from features_input_csv
    #  starting from last date +1 to yesterday
    #  save/concatenate to features_input_untrained_csv

    # call create_delta_df_from_web(start_date, end_date, solar=True)

    # same format and cleaning as features used for training, ie included production.

    # WTF WHY yesterday -1: to make sure cloud is updated. if this is ran day n in the morning, day n-1 not yet there in cloud (until 7:30pm)
    # yesterday -1 insures solar production is available - used to test model on unseen data


    ###########
    # STEP 1: set last seen date, either from df_model or from last untrained features
    ###########

    if os.path.exists(features_input_untrained_csv):

        print('%s exists. add to it' %features_input_untrained_csv)

        ############
        # delta (ie features_input_untrained_csv) already exist. 
        # will start from there and add at the end
        # set last date
        ############
        last_date = dataframe.get_last_cvs_date(features_input_untrained_csv)

        if last_date == None:
            raise Exception("cannot get last date from %s" %features_input_untrained_csv)
        else:
            print("last recorded date from %s is %s" %(features_input_untrained_csv, last_date))

        # current , to accumulate into
        df_untrained = pd.read_csv(features_input_untrained_csv)

        dataframe.assert_df_model_integrity(df_untrained , s="loaded df_untrained. will add to it at the end")
       
        
    else:
        ############
        # delta (features_input_untrained_csv) does not exist. 
        # will start from trained
        ############
        print('!! %s does not exists. start from trained features' %features_input_untrained_csv)
        # get new data from last date in trained features
        last_date = dataframe.get_last_cvs_date(features_input_csv)
        if last_date == None:
            raise Exception ('cannot get last date from %s' %features_input_csv)
        else:
            print("last recorded date from %s is %s" %(features_input_csv, last_date))



    ##### last date is set. from features_input_untrained_csv or from features_input_csv
    print("last recorded date (from trained model or from existing untrained)", last_date)
    # create delta feature from last date + 1  to yesterday -1

    days_go_back  = -2 # yesterday -1 from today

    #days_go_back  = -33

    ######################################
    # STEP 2: scrap df_delta from last_date to yesterday -1. same cloud story, as this can be executed anytime (no time limit)
    ######################################
    # if last date is yesterday -1, nothing to do. in case this is executed multiple time the same day
    today = datetime.date.today()  # # use datetime.date , vs datetime.datetime 

    y = today + datetime.timedelta(days_go_back)

    # check if existing feature untrained is already up to date

    if y.day == last_date.day and y.month == last_date.month:
        print("%s already up to date. today %s, last seen date %s is yesterday -1. DO NOTHING" %(features_input_untrained_csv, today, last_date))

    else:
    
        # creates features for date range (df_delta)
        # from last_date + 1 til yesterday -1
        # WTF WHY yesterday -1: to make sure cloud is updated. if this is ran day n in the morning, day n-1 not yet there in cloud (until 7:30pm)
        # scrap and enphase API
        
        start_date = last_date + datetime.timedelta(1) # last captured + 1
        end_date = datetime.datetime.now() + datetime.timedelta(days_go_back)  # day before yesterday (as of running)

        # WTF: API call fails if asking ONE day #
        # enphase API: getting daily energy from: 2024-02-05 to 2024-02-05 return empty list
        # check here (anyway, this is not supposed to be called for small update)

        if (end_date - start_date).days <= config_model.min_unseen_days: #end_date - start_date is timedelta. .days =0 if same day
            s = "!! start date %s too close to end date %s. do not create delta and do not update/create %s" %(start_date.date(), end_date.date(), features_input_untrained_csv)
            print(s)
            logging.error(s)
            # even if not updated here, inference (or retraining) will still ran on EXISTING features_input_untrained_csv

        else:

            # update features_input_untrained_csv

            # same structure as one used for training
            s = "\ncreate df_delta from %s (after last recorded) to %s (yesterday -1). both included. no time limit, ie can run query abnytime" %(start_date.date(), end_date.date())
            print(s)
            logging.info(s)

            ####################
            # creates df_delta
            ####################
            df_delta = dataframe.create_unseen_delta_input_df_from_web(start_date, end_date, days_go_back) # date if there is already Timestamp, not str

            dataframe.assert_df_model_integrity(df_delta, s="unseen df_delta (unseen or retrain)")

            # can concatenate (for retrain)
            #dataframe.assert_df_model_concatenate(df_model, df_delta, s="concatenate df_model and df_delta (unseen or retrain)")


            # integrity check done in function
            s = "df_delta available and well. (meteo scrap and enphase API). start %s end %s" %(start_date.date(), end_date.date()) 
            print(s)
            logging.info(s)


            ######################################
            # STEP 3: save df_delta as features_input_untrained_csv, or concatenate df_delta to existing features_input_untrained_csv
            #  and save result as csv
            ######################################

            # untrain = delta (features_input_untrained_csv does not exist)
            # untrain = untrain + delta (features_input_untrained_csv exists)
            # specify date format to make sure. BUT not really needed, as date was previously converted to str
            # format expected when getting last date

            if not os.path.exists(features_input_untrained_csv): 

                # features_input_untrained_csv does not exist, create it
                print('save df_delta as %s csv' %features_input_untrained_csv)

                # specify date format to make sure. BUT not really needed, as date was previously converted to str
                # make sure format is consistent with existing features_input.csv and get_last_date
                df_untrained = df_delta
                df_untrained.to_csv(features_input_untrained_csv, index=False, date_format="%Y-%m-%d")  
                # index = False to avoid creating new colums


            else:

                # untrain = untrain + delta (features_input_untrained_csv exist)
                # concatenate df_delta with existing df_untrained

                dataframe.assert_df_model_concatenate(df_untrained, df_delta, s="concatenate existing df_untrained and scrapped df_delta")

                # df_untrain = df_untrain + df_delta
                print('\nconcatenate df_delta dataframe to EXISTING %s csv' %features_input_untrained_csv)

                l = len(df_untrained)

                df_untrained = pd.concat([df_untrained, df_delta], ignore_index=True, axis = "index") #axis{0/’index’, 1/’columns’}, default 0

                dataframe.assert_df_model_integrity(df_untrained, s="concatenate df_untrained and df_delta (unseen or retrain)")

                assert len(df_untrained) == l + len(df_delta)

                df_untrained.to_csv(features_input_untrained_csv, header=True, index=False, date_format="%Y-%m-%d") # index = False, 


            ######################################
            # STEP 4: done. df_uptrained updated or created. (and its csv version is )
            ######################################
            

            assert len(df_untrained) % len(retain) == 0
            nb_days_untrained = len(df_untrained) / len(retain)

            print("\ncreated or updated %s. %d unseen/untrained days. what next depend on unseen or retrain" %(features_input_untrained_csv, nb_days_untrained ))

    # df_untrained created/updated and saved as features_input_untrained_csv, or left as it
    


##################################
# GOAL: create updated df_model (and csv) by concatenating df_untrained with existing df_model. 
# delete features_input_untrained_csv 
##################################
if arg["retrain"]:

    # RETRAIN:  for continous training while in PRODUCTION
    # do not use same time as unseen as this delete features_input_untrained_csv. use unseen, then rerun with retrain
    # NOTE: this retrain from scratch , vs continue training (ie using existing weigths)
    # this only extend df_model. actual training later

    print("\nRETRAIN: merge untrained features with existing features into COMBINED df_model to retrain from scratch on larger dataset")
    
    df_model = pd.read_csv(features_input_csv) 

    # check concat was not already done today, ie last seen is yesterday
    df_model[df_model.columns[0]] = pd.to_datetime(df_model[df_model.columns[0]])

    last_seen = df_model.iloc[-1] [0].date()
    if datetime.date.today() - datetime.timedelta(1) == last_seen:
        print("df_model already updated today. likely running retrain multiple time the same day")

    else:

        if os.path.exists(features_input_untrained_csv):

            print("concatenate df_model and df_untrained to NEW df_model. will train a new model based on this")

            df_untrained =  pd.read_csv(features_input_untrained_csv)
            l = len(df_model) + len(df_untrained)

            # check integrity before concatenate
            dataframe.assert_df_model_concatenate(df_model, df_untrained , s="df_model and df_untrained can be concatenated")

            # concatenate df_model and df_untrained into new df_model
            #axis{0/’index’, 1/’columns’}, default 0

            df_model = pd.concat([df_model, df_untrained], ignore_index=True, axis = "index")

            assert l == len(df_model) # new df_model has both trained and untrained features

            dataframe.assert_df_model_integrity(df_model , s="concatenation of df_model and df_untrained into larger df_model")

            ##### save new (extended) df_model
            print("save as new (extended) feature input as: %s" %features_input_csv)
            df_model.to_csv(features_input_csv, index=False, date_format="%Y-%m-%d")  # index = False, otherwize will create new unamed columns with 0,1,2..

            ###########################
            # delete untrained csv
            ###########################
            print('delete untrained %s, it is now merged into main feature_input' %features_input_untrained_csv)
            os.remove(features_input_untrained_csv)

        else:
            print("!! retrain on what ? %s does not exist. pass" %(features_input_untrained_csv))

        # updated features_input_csv and df_model available

    # UPDATED df_model is concat df_untrained with existing df_model and saved as features_input_csv
    # df_untrained deleted
            
    if not running_on_edge:
        pass


####################################################
# df_model (features) ready for dataset creation
# real data (initial scrapped or loaded from csv, possibly concatenated with untrained),
####################################################

print("\n")
print("##################################################################################################")
print("df_model (and csv) ready for dataset creation")
print("real data (just scrapped or loaded from csv, then possibly concatenated with untrained for retraining")
print("##################################################################################################")

last_seen = df_model.iloc[-1] [0]
print("df_model last date: %s" %last_seen) 


##############################
# simulate smaller dataset , ie 1 year only
##############################

"""
print("simulate smaller dataset")
df_model = df_model.iloc[:365*24]
print(len(df_model), len(df_model)/24)
"""

#########################
# can we think of a naive model to compare with ??
#########################
# eg constant ration betwen production and average of temperature over last n days 


####################################################
# GOAL: analyze energy distribution, ie production quentile
####################################################

# from df_model
# guide for bin configuration "prod_bins":[0, 7.74, 16.51, 22.89, 1000000], to get equal number of sample per bin

# to be used as a guide to configure nb of bins and bins BOUNDARIES (if one wants same number of samples in each bins)
# Note: having equal number of input in each bin may not be the optimum, may be care more about lower and upper end of the spectrum
# later will build histograms of solar from dataset. if using boundaries below, nb of samples per histogram should be equal

print("\nshow energy quentile in df_model. can be used this to configure bins boundaries (to get equal number of input per bin)")
      
energy = df_model["production"]

max_prod = energy.max(skipna=True)
min_prod = energy.min(skipna=True)
median_prod = energy.median() # 16.8
mean_prod = energy.mean() # 15.2
print("max prod: %0.1f, min prod: %0.1f, mean: %0.1f, median: %0.1f" %(max_prod, min_prod, mean_prod, median_prod))

# number of bins is a design decision. then if using boundaries below, df_model should have equal number of input per bin
print("CURRENTLY configured production bins: ", config_features.config["prod_bins"])
print("PLEASE use below to update bins to get equal number of samples per bins")

quartile = energy.quantile([.33, .66]) # 3 bins serie
print("energy as 3 bins: ", quartile.to_list())

quartile = energy.quantile([.25, .5, .75]) # quartile , returns serie
print("energy as 4 bins (quartile): ", quartile.to_list())

quartile = energy.quantile([.5]) # quartile , returns serie
print("energy as 2 bins (median): ", quartile.to_list())

quartile = energy.quantile([.2, 0.4, 0.6, 0.8]) # quartile , returns serie
print("energy as 5 bins: ", quartile.to_list())

############################################################
# GOAL: build sequence dataset ds
# from panda df_model and based on many other input (which all are design decision)
############################################################
# feature_list specifies list of list(s) of serie(s) to includes as input

print("\nbuild sequence dataset from df_model") 

ds, nb_seq, nb_hours = \
dataset.build_dataset_from_df(df_model, \
input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
stride=stride, sampling=sampling, shuffle=shuffle, categorical=categorical)


print("\n####\ndataset ready for training\n####\n")

##############################
# GOAL: split dataset for validation strategy
#############################

# dataset split. needed for benchmark, keras tuner.
# MODIFIED before training proper ??? WHY IS THIS NEED TO BE REDONE ???
# not sure if kfold can be used together with tuner
(train_ds, val_ds, test_ds, ds_dict) = dataset.fixed_val_split(ds, nb_seq, nb_hours, split, retain, batch_size)

####################################
# GOAL: build solar production histograms from dataset
####################################

# build histogram with configured bins (should set for BOTH classification and regression)
#  should correlate with bin boundaries (if set with quantile)
#  model performance should beat this (ie with 4 bins and EQUAL number of sample per bins, ramdon performance is 25%)

# those may be needed for retrain, so on PI as well

print("\nhistogram of solar production from dataset, using %s" %prod_bins)
histo_prod, majority_class_percent, majority_y_one_hot = enphase.build_solar_histogram_from_ds_prior_training(ds, prod_bins)

# majority_y (one hot) needed for baseline model which returns majority class
print("majority_y %s, majority class: %0.2f %%. Histogram: %s" %(majority_y_one_hot, majority_class_percent, histo_prod))

# go back to source data about production
# NOTE: this is production from df_model, ie same production for every hour for same day. 
# assumes this is the same a distribution from original downloaded production csv
print("validate with quentile from df_model")
energy = df_model["production"]
enphase.solar_quentile_from_df(energy)


### create big dict containing eveything we need to know about the model
# some values are dynamic/computed, but many are straigth from config files (and should not have changed, so COULD ALSO HAVE been gotten from there in the called module)
model_param_dict = {
    "ds_dict": ds_dict,
    "input_feature_list" : input_feature_list,
    "days_in_seq": days_in_seq,
    "seq_len": seq_len,
    "retain": retain,
    "repeat": repeat,
    "selection": selection,
    "seq_build_method": seq_build_method,
    "hot_size": hot_size,
    "batch_size":  batch_size,
    "prod_bins": prod_bins,
    "prod_bin_labels" : prod_bin_labels,
    "prod_bins_labels_str": prod_bins_labels_str,
    "stride": stride,
    "sampling": sampling,
    "shuffle": shuffle,
    "categorical": categorical,
    "majority_class_percent": majority_class_percent,
    "majority_y": majority_y_one_hot,
    "nb_hours" : nb_hours,
    "test_ds" : test_ds
}


#########################################
# GOAL: create evaluation dataframe
# record training metrics 
########################################

if arg['train'] or arg['retrain']:

    # Index(['model', 'train_test', 'categorical accuracy', 'precision', 'recall', 'prc']
    # for every case involding training
    # used to plot training metrics 

    # create dataframe to store evaluate result from actual model and baselines
    # header is from list of metrics from metrics.name

    evaluate_df = pd.DataFrame(columns=[m.name for m in metrics]) # ['categorical accuracy', 'precision', 'recall', 'prc']
    evaluate_df.insert(0,"model", np.nan)
    evaluate_df.insert(1,"train_or_test", np.nan)

    print('\ncreate training evaluate dataframe: ' , evaluate_df.columns.to_list())

# NOTE: no evaluation dataframe created for search (offline) or tuner (tuner manages it itself)


###########################################
# GOAL: get a trained model
#    tuner or train or load
###########################################

# if tuner: 
#    search and train on best model . EXIT

# else if train/retrain:  (ie all case doing "manual training")
#    if retrain, get previous metrics
#    train, evaluate, predict on ds, plot, update evaluate dataframe, save, upate record run
#    if retrain: update gui
#    if vault: evaluate on real test data. EXIT

# else 
#     load trained model, evaluate, predict on ds, update GUI 


####################################################
# GOAL: search and exit
####################################################
if arg["kerastuner"]:

    #############################
    # type of tuner
    #############################

    #kt_type_list = ["random" , "hyperband" , "bayesian"]

    kt_type = config_model.kt_type

    #### single step creates and train model .  hyper parameters 
    # validation split must be available, as this run training. 
    # also ds is needed for lstm model build (hot size, multi heads)
    # returns best model trained and best_empty is to be trained myself, ie on a larger dataset
    # callback list defined in .tune

    ##########################
    # start the search
    ##########################
    print("\n############\nStart searching best hyper parameters with keras tuner: %s\n############\n" % kt_type)

    best_to_train, best_trained = tuner.tune(train_ds, val_ds, categorical, kt_type=kt_type) 

    # model structure should be the same for best_empty and best_trained
    #pabou.see_model_info(best_to_train, os.path.join(models_dir , 'best model tuner.png') )

    # do not bother saving. will just capture "manually" best hyper
    #print("TUNER: save best model (empty and trained)")
    #pabou.save_full_model(app +"best_tuner_empty", best_empty)
    #pabou.save_full_model(app +"best_tuner_trained", best_trained)

    ########################
    # best model already trained by keras tuner
    ########################

    r  = best_trained.evaluate(test_ds, return_dict=True, verbose=0) 
    print("\nkeras tuner: best model trained by kt: metrics from evaluate:\n", r)

    print("\nkeras tuner:  best model trained by kt: running inference")
    nb, nb_correct, _, _ = inference.predict_on_ds_with_labels(best_trained, test_ds, s ="keras tuner")


    #########################
    # retrain with best hyperparameters
    #########################

    # train model with best hyperparameters on entiere dataset
    print('\nkeras tuner: best hyperparameters: RETRAIN from scratch using train_ds')
    (new_model, history, elapse) = train_and_assess.train(best_to_train, epochs, train_ds, val_ds, model_param_dict, checkpoint = False, verbose =0)

    (m_test, m_train, ep) = train_and_assess.metrics_from_evaluate(new_model, test_ds, train_ds, history)
    print('\nkeras tuner: best hyperparameters: RETRAINED, metrics from evaluate, test %s, train %s' %(m_test, m_test))

    r1  = new_model.evaluate(test_ds, return_dict=True, verbose=0)
    print("\nkeras tuner: best hyperparameters: RETRAINED, evaluate on test ds", r1)

    print("keras tuner: best hyperparameters: RETRAINED, inference")
    nb1 ,nb_correct1, _ ,  _ = inference.predict_on_ds_with_labels(new_model, test_ds, s= "keras tuner, RETRAIN")

    # could also add to history

    #########################
    # compare model trained by keras tuner and retrained from scratch
    # NOTE: seems better to retrain
    #########################
    print("\nkeras tuner: comparing model trained by tuner, and model retrained on train_ds from scratch using best hyperparameters")

    print("correct percent (using .predict()) kt trained: %0.2f, best hyper retrained on train_ds: %0.2f" %(nb_correct/nb, nb_correct1/nb1))
    print("kt trained:",r)
    print("best hyperparameters, retrained on train_ds:", r1)

    print('\n#####\nTuner: EXIT. please analyze results\n#####\n')

    sys.exit(0)
    # end of tuner



###########################################################
# GOAL: train model using ds
# all cases involving training
#   features
#   or features + untrained
###########################################################

elif arg['train'] or arg['retrain']:

    #############################
    # if trying to improve from existing trained model , using new unseen data,  first get starting point
    # WARNING: assumes a "baseline" trained model already exists
    # load the model just to get "current" metrics
    #############################

    # could also look at stored metrics
    if arg["retrain"]:

        print("\nretrain. load model just to get current metrics:")

        # load model before retrain  just to get accuracy (and later check if/how improved)
        try:
            model, h5_size, _ = pabou.load_model(model_name, 'tf')

            if model is  None :
                raise Exception ("!!!!!! CANNOT LOAD MODEL ")
        except Exception as e:
            raise Exception ("!!!!!! CANNOT LOAD MODEL ", str(e))

        # evaluate previous model. use to check if we improved. also used in GUI label


        model_metrics_before_retrain_dict  = model.evaluate(test_ds, return_dict=True, verbose=0)
        print("about to retrain: current metrics:\n", model_metrics_before_retrain_dict)

        del(model) # make sure


    ##### CREATE model
    # use model parameter defined for kerasTuner 
    # DO NOT USE ds (only train ds). this would means norm.adapt() is done on total ds and model "see" test set. MAYBE BAD
    # BUT on the other end, using ds allows to handle validation split strategy later
        

    ########################
    # build model
    ########################
    
    model = model_solar.build_lstm_model(ds, categorical,
    name='base_lstm', units = nb_unit, dropout_value=dropout_value, num_layers=nb_lstm_layer, nb_dense=nb_dense, use_attention=config_model.use_attention)

    # plot model architecture
    pabou.see_model_info(model, os.path.join(models_dir , model_name +'_architecture.png') )

    ################################
    # train FROM SCRATCH with ds
    ###############################

    # do I need to return model ?
    # can use checkpoint call back 
    model, history, elapse = train_and_assess.train(model, epochs, train_ds, val_ds, model_param_dict, checkpoint = True, verbose=0)

    # history contains metrics configured in model_solar.get_metrics(categorical) BOTH met and val_met  , plus loss and lr

    #########################
    ## model trained
    #########################

    train_samples = ds_dict["samples"]["train"] * repeat
    val_samples = ds_dict["samples"]["val"] * repeat

    actual_epochs = len(history.history["loss"])

    print('\n\nTRAINING ENDED ===>trained in %0.1f sec on %d samples. max %d epochs. actual_epochs %d, ie %0.1f sec/epochs\n\n' %(elapse, train_samples, epochs, actual_epochs, elapse/actual_epochs))

    ##########################
    # .evaluate()
    ##########################

    # wrapper for .evaluate() , do it for BOTH test and train
    # returned values used to update evaluation dataframe
    # ['categorical accuracy', 'precision', 'recall', 'prc']
    (metrics_test, metrics_train, ep) = train_and_assess.metrics_from_evaluate(model, test_ds, train_ds, history)

    # model.metrics_names include loss. not included in metrics_test
    print("\n############## ARE YOU HAPPY? ##############\n")
    print("metrics used: %s" %model.metrics_names)
    print("evaluate on test set: %s\n" %metrics_test)

    # also get test metric as dict.
    # should be then same as metrics_test above
    trained_model_metrics_dict  = model.evaluate(test_ds, return_dict=True, verbose=0)

    ######################
    # save metrics as json
    #######################
    # used later to update GUI model tab gauges
    # WARNING: tf does not support precison, recall for multi class. need to get that separatly (per class, macro average) with sklearn
    #  dict parameter is ONLY the metrics we which to update

    # names defined in model_solar.get_metrics()
    # for categorical, we will also update precision, recall later

    ##### I misspelled accurary. Not sure if there is a way not to type the str twice. just be CAREFULL
    
    if categorical:
        d = {"categorical accuracy": trained_model_metrics_dict["categorical accuracy"]}
    else:
        d = {
            "mae" : trained_model_metrics_dict["mae"],
            "rmse" : trained_model_metrics_dict["rmse"]
        }

    train_and_assess.metric_to_json(categorical, d)

    #####################
    # did the retrain on larger dataset improved metrics ?
    # NOTE: GUI not started, so cannot communicated any improvement (unless messing arpund with json)
    # NOTE: the new model will overwrite the previous one, even if less accurate. 
    #   I guess that it life. there may be some noise in computing. I assume it will re get better overtime. anyway the more data, the better
    #####################

    if arg["retrain"]:
        print("===> RETRAIN: for reference, metrics before retrain:" , model_metrics_before_retrain_dict)

        if categorical:
            if trained_model_metrics_dict["categorical accuracy"] > model_metrics_before_retrain_dict["categorical accuracy"]:
                s= "!! RETRAIN: accuracy improved from %0.2f to %0.2f" %(model_metrics_before_retrain_dict["categorical accuracy"],trained_model_metrics_dict["categorical accuracy"])
                print(s)
                logging.info()
            else:
                s= "!! RETRAIN: accuracy did NOT improve from %0.2f to %0.2f; using anyway" %(model_metrics_before_retrain_dict["categorical accuracy"],trained_model_metrics_dict["categorical accuracy"])
                print(s)
                logging.warning()
                
        else:
            if trained_model_metrics_dict["mae"] < model_metrics_before_retrain_dict["mae"]:
                s= "!! RETRAIN: mae improved from %0.2f to %0.2f" %(model_metrics_before_retrain_dict["mae"],trained_model_metrics_dict["mae"])
                print(s)
                logging.info(s)
            else:
                s= "!! RETRAIN: mae did NOT improve from %0.2f to %0.2f; using anyway" %(model_metrics_before_retrain_dict["mae"],trained_model_metrics_dict["mae"])
                print(s)
                logging.warning(s)


            
    ##########################
    # .predict() after training
    ##########################

    # compute metrics "manually" using INFERENCE on dataset. 
    # metrics should match results from evaluate

    nb, nb_correct, nb_correct1, t = inference.predict_on_ds_with_labels(model, test_ds, s="predict() after training")
    
    if not categorical:
        errors = t[1] # error numpy
        mae = t[0]
        assert errors is not None

        # analyze distribution, and plot histograms
        print("analyzing error distribution for mae: %0.1f" %mae)
        train_and_assess.analyze_error_distribution(errors)

    else:
       # _ = t[0] # confidence
        _ = t # (b) is returned as b

    ##########################
    # various PLOT after training
    # confusion matrix(categorical) or error histograms (regression)
    # training history with pandas line and pabou
    # loss (plt)
    # print precision, recall from sklearn
    # save it to json
    ##########################

    p_r_dict = train_and_assess.plot_examine_training_results(model, test_ds, history, model_param_dict)

    # make sure the keys are not misspelled
    train_and_assess.metric_to_json(categorical, p_r_dict)


    ##########################
    # Update EVALUATION dataframe 
    ##########################
    # metrics on test and training set
    # dataframe used at the end to create plot
    row = [model.name, "test"] + metrics_test
    evaluate_df.loc[len(evaluate_df.index)] = row

    row = [model.name, "train"] + metrics_train
    evaluate_df.loc[len(evaluate_df.index)] = row

    #  baseline model (add 3 entries)
    if categorical:
        # use full ds, update evaluate_df
        (confusion_matrix_l, evaluate_df) = train_and_assess.baseline_confusion_matrix(ds, evaluate_df, majority_y_one_hot, metrics, hot_size, histo_prod) # return list of confusion matrices

        for i, c in enumerate(confusion_matrix_l):
            print("baseline confusion matrix: ", "baseline_%d" %i, c)

    print("updated evaluate dataframe", evaluate_df.head(5))


    ##########################
    # SAVE model
    ##########################

    # both tf and h5 format. save weigth as epoch 999, architecture as json
    # callback also save model or weigths

    # 'solar2ev' + '_full_model.h5'
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    print("\nsaving model %s in all formats (SaveModel, h5, weigths, json)" %model_name)

    pabou.save_full_model(model_name, model) # signature = False , dir created if needed

    print('h5 model size %0.1fMb' %pabou.get_h5_size(model_name))

    ##########################
    # RECORD training run
    ##########################

    # one entry per training run
    # header defined in record_run.record_param_hdr
    # use str(list) to avoid dim issues Shape of passed values is (18, 1), indices imply (18, 17)

    run = 0

    record_run_param = [str(categorical),
    len(prod_bins)-1, str(input_feature_list), days_in_seq, 
    ds_dict["samples"]["train"],
    seq_build_method, str(selection), stride,
    metrics_test[0],metrics_test[1], metrics_test[2], metrics_test[3],
    run, sampling, repeat, shuffle,
    nb_lstm_layer, nb_unit, nb_dense, config_model.use_attention,
    elapse, ep
    ]

    assert len(record_run_param) == len(record_run.record_param_hdr)

    print('record training run in: %s' %history_training_csv)
    if (record_run.record_training_run(history_training_csv, record_run_param)):
        print(pd.read_csv(history_training_csv).tail(1))
    else:
        print("ERROR recording run result")


    ##################################
    # do NOT update blynk model info in GUI
    # do NOT update text and metrics gauge
    #################################

    # starting blynk when training is an overkill when developping model
    # for all option except those who call inference_from_web_at_a_day(), only ONE model is loaded
    #   ie daily inference and postmortem
    # on the other end, the GUI model info page has gauges for BOTH categorical and regression metrics
    # simpler to update gauge when running inference (even if should not change often)
    #   ie run evaluate on both model 
        
    

    # end of TRAIN/RETRAIN
    # ie all cases involving training a model


##########################
# GOAL: LOAD an already trained model
##########################
else:

    try:
        x = 'tf' # directory
        x = 'h5' # single file, easier to transfert from Windows to raspberry Pi
        x = "v3" # latest, recommended. zip. not saved as zip on 2.10 ??? but as h5 ??

        x = 'tf'

        print('\nload model (format: %s): %s' %(x, app_name))
        model, h5_size, _ = pabou.load_model(model_name, x)
        if model ==  None:
            raise Exception ("!!!!!! CANNOT LOAD MODEL %s %s" %(app_name, x))

        ######################################
        # evaluate after loading
        #####################################

        print("\nmodel loaded")

        if not running_on_edge:
            print("running on desktop .evaluate(test dataset), ie remind me the accuracy:")
            # remind me, what is the accuracy of the current model

            # verbose = 1, trace 
            loaded_model_metrics_dict  = model.evaluate(test_ds, return_dict=True, verbose=0) # return dict = false returns a list, with loss as first element
            print("%s" %str(loaded_model_metrics_dict))

            # also check with .predict
            # for categorical, % correct prediction should match acc from .evaluate().
            # NOTE: for regression: correct uses either max acceptable error or use bins
            print("run .predict(test dataset)")
            nb, nb_correct, nb_correct1, _ = inference.predict_on_ds_with_labels(model, test_ds, s ="test loaded model")
            # NOTE: quentile print is done in function

        else:
            # speed up start on the PI
            print("running on edge. do not bother doing .evaluate()")


        ##########################################
        # update GUI with model accuracy
        ##########################################
        # done in daily inference/postmortem

    except Exception as e:
        raise Exception ("!!!!!! CANNOT LOAD MODEL ", str(e))


#################################################################
# model available (train/retrain/vault or load already trained)
#################################################################

print("\n")
print("############################################################")
print("model loaded, either the one just trained (train/vault/retrain) or a saved one")
print("now can do: inference, charge, postmortem ,bench ,unseen")
print("############################################################")


################################################
# everything we can do with a trained model:
################################################

#daily inference: (19h:30 CET)
#charge: (23h)
#postmortem (same as daily inference, or anytime)

#bench
#unseen (only if retrain is false)


##################################
# before exiting
##################################
#train/retrain/vault: plot evaluate dataframe
#GUI: sleep

if arg["test_blynk"]:
    s = "%s testing blynk. update GUI model tab" %datetime.datetime.now()
    print(s)
    logging.info(s)

    blynk_ev.blynk_v_write(vpins.v_terminal, s)

    inference.update_GUI_model_tab()

#########################################################
# GOAL: run inference to predict tomorrow's solar 
######################################################### model_param_dic

if arg["inference"]: 

    # meant to run late afternoon "today", when today's solar production is known (because model could be trained with input features conraining solar)
    
    # does all processing (update GUI, prediction history file)
    # can be invoqued from command line (eg CRON, systemd timer)

    # WARNING: both model (categorical, regression) should be available before

    #        same WTF: cloud updated late afternoon today (ie time of daily inference) for yesterday's solar
    #        !! not yet updated 18:50 CET
    #           it is OK to get today solar with telemetry api 

    print("\nrunning daily inference, ie predict solar tomorrow")

    flag, _ = inference.predict_solar_tomorrow(model, model_param_dict) # bool, dict

    if not flag:
        s = "error running daily inference, ie predict solar tomorrow"
        print(s)
        logging.error(s)
        # alarms already done

    # do not need dict (needed for charger)
        

###################################################
# GOAL: program EV charger for overnigth charge
###################################################

if arg["charge"]:

    # run late nite "today", to set charger to charge overnigth (off peak hours)
    # read GUI to get instruction (eg confidence high enough)

    # meant to be invoqued from command line (eg CRON)
    # predict to get proba. used as condition to set charge

    # the idea is the use will see the result of daily inference, and than can set the charge mode accordingly
    # so this cannot run at the same time as daily inference. give user some time to react

    flag, dict_result = inference.predict_solar_tomorrow(model, model_param_dict) # return list

    if not flag:
        s = "error running inference, needed for charger"
        print(s)
        logging.error(s)

    else:
       
        ret = charger.set_charge_tommorrow(dict_result)
        if not ret:
            s = "error setting charger"
            print(s)
            logging.error(s)
        else:
            s = "charger configured"
            print(s)
            logging.info(s)
        


###############################
# GOAL: validate if known production would have been correctly predicted
###############################
            
# can run multiple time (will execute inference and update GUI, will just not update the history json)

if arg["postmortem"]: 

    # json file
    # "date": date of update of last entry in history list
    # "history" list of int, more recent at the end

    # context:
    # ground truth is day n
    # day n is last day of sequence
    # need additional day_in_seq -1 to build sequence 
    #     eg day_in_seq = 4, day n is 10 => needs 3 days, ie 7,8,9,10 ,ie n-(day_in_seq-3) to n - 1
    # NOTE: also need an additional day if using averaging

    # if True: run postmortem in the following morning, ie day n+1
    #       NOTE: solar ground truth already available in enphase cloud, use daily API and get day n as well
    #       as_if is day before yesterday (seen from the time the module is called) , to predict yesterday
    #       a single enphase API call to build sequence and get truth
    #       less easy to use ?, I guess it is a benefit to see postmortem at the same time as daily prodiction
    #       easier to debug (can run anytime)
    #       WTF: cloud updated late afternoon next day !!! 
    #           until updated, the API call returns one less data
    #           so benefit of using the method vs running at daily inference time is low
    #             unless we use this to go 2 days back in time. This sucks

    # if False: run postmortem about the same time as daily inference
    #       NOTE: same constraint as daily inference, ie run when sun is set and production over (and know to telemetry)
    #       as_if is yesterday (seen from the time the module is called), to predict today
    #       use additional telemetry API enphase call to build day n (ie today) production
    #       available to user at the same time of tomorrow inference


    postmortem_next_day = config_features.post_mortem_next_day

    print("\npostmortem, ie run inference as if in the past to validate against known solar production")

    if postmortem_next_day:
        print("WARNING: run postmortem without any time limit (run anytime), BUT will go back an extra day in the past")
    else:
        print("WARNING: run postmortem at the same time as daily inference. time limit will be enforced")
        
        
    ret, result_dict = inference.post_mortem_known_solar(model, model_param_dict, postmortem_next_day)

    if not ret:
        s = "postmortem failed"
        print(s)
        logging.error(s)


#################################################################
# GOAL: BENCHMARK (performane and accuracy) 
#################################################################
    
if arg['bench']:

    print("\nbenchmark performance and accuracy (test dataset")

    # nb number of batches 
    nb = len(test_ds) # on full test dataset

    pabou.bench_full(model_name + ' full model, test ds', model, test_ds , pabou.get_h5_size(model_name), acceptable_error = acceptable_absolute_error, nb = nb, bins=prod_bins)

    # returns dict of metrics
    print('benchmark: .evaluate():')
    r = model.evaluate(test_ds, return_dict=True, verbose = 0)
    
    for k in r.keys():
        print ("%s : %0.2f" %(k,r[k]))

    # performs .predict() on dataset with labels. goes thru predictions one by one
    print('benchmark: .predict():')
    nb, nb_correct, nb_correct1,  _ = inference.predict_on_ds_with_labels(model, test_ds, s="benchmark")



#################################### 
# GOAL: test model accuracy on unseen (ie UNTRAINED) data
# until yesterday -1, to make sure enphase cloud is updated (so can run this "anytime")
####################################

if arg["unseen"] :

    # update unseen widget on GUI
    # UNTRAINED, but past data, so labels exists and can compute accuracy
    # mutually exclusive with retrain

    # feature untrained csv already created
    # use it as inference input.
    # get a few days from tail of df_model, to build sequences
    # and concatenate with untrained
    # build dataset
    # call  evaluate_acc_on_ds_with_labels

    if not arg["retrain"]:    

        #############################
        # compute accuracy on data not yet used for training
        # labels available
        #############################

        print("\ncheck model accuracy on unseen/untrained data (ie labels exists): %s" %features_input_untrained_csv)

        #################################
        # get all untrained data so far
        #################################
        if os.path.exists(features_input_untrained_csv):
            df_untrained = pd.read_csv(features_input_untrained_csv)

            # read timing and check integtrity
            dataframe.assert_df_model_integrity(df_untrained, s="loaded df_untrained")

            untrained_days = len(df_untrained) / len(retain)
            print("unseen (aka untrained): %d days" %untrained_days)

            ###################################
            # need to get a few previous days to build input sequences
            ###################################
            df_model = pd.read_csv(features_input_csv)

            # check df_model (trained data) and df_untrained can be concatenated
            dataframe.assert_df_model_concatenate(df_model, df_untrained, s="concatenate df_model(trained data) and df_untrained")

            # need last (day_in_seq) from existing model last 3 predict 4th
            # df_last is last couple of rows in df_model
            nb_row = days_in_seq * len(retain)
            df_last = df_model[-nb_row:] # get last 96 rows on features seen by model  , eg 6 to 9 march included
            assert len(df_last) == nb_row
            print("add last %d (days_in_seq) of df_model before building sequence" %days_in_seq)

            # check can be concatenated
            dataframe.assert_df_model_concatenate(df_last, df_untrained, s="concatenate df_last (last days from df_model) and df_untrained") 

            # concatenate last few days to untrained
            #axis{0/’index’, 1/’columns’}, default 0

            #######################
            # concatenate df_last and df_untrained
            #  ie what is needed to build sequences 
            #######################
            df_concat = pd.concat([df_last, df_untrained], ignore_index=True,axis = "index")

            # check in days
            assert len(df_concat) / len(retain) == untrained_days + days_in_seq
            assert len(df_concat) == len(df_last) + len(df_untrained)
            dataframe.assert_df_model_integrity(df_concat, s="concatenation of df_last and df_untrained")
            

            # save for debug
            print("save unseen plus few days as csv for later debug:", unseen_csv)
            df_concat.to_csv(unseen_csv, header=True, index=False)

            print("df ready for sequence building has now %d untrained days, %d days total (%d added to build dataset)" %(untrained_days, len(df_concat)/len(retain), days_in_seq))
        
            # we could build all sequences 
            # or only sequences which represent inference input, ie which starts around 0h, ie less sequences, but more relevant

            print("building dataset for unseen data. build method %d, stride %d" %(seq_build_method, stride))
            ds_unseen, nb_seq, nb_hours = dataset.build_dataset_from_df(df_concat, \
            input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
            stride=stride, sampling=sampling, shuffle=shuffle, labels=True, categorical=categorical)

            stride = config_model.stride
            seq_build_method = config_model.seq_build_method

            print("%d sequences created" %nb_seq)

            # 7 unseen days, 11 in df , ie 264 hours
            # call .predict() on the entire dataset (assumed to contains labels). compute number of correct prediction 
            # average confidence for OK cases

            print("\npredict() on unseen data")
            nb, nb_correct, nb_correct1, t = inference.predict_on_ds_with_labels(model, ds_unseen, verbose=True, s="unseen data")
            # nb is number of sequences (with stride)

            # regression
            # nb_correct:  use ABSOLUTE , compare with acceptable_error , interpreted as ae
            # nb_correct 1: ùap to categorical bins

            assert nb == nb_seq

            r  = model.evaluate(ds_unseen, return_dict=True, verbose=0)
            print("\nunseen data .evaluate()" ,r)

            ##########################
            # update unseen widget on GUI
            ##########################

            # persistent label. 
            # double % to escape %

            blynk_ev.blynk_color_and_label(vpins.v_unseen, blynk_ev.color_text, "%s: running model on days unseen by training" %datetime.date.today()) # not initialized. #FFFFFF is white
            
            if categorical:
                confidence = t[0]
                s = '%d unseen days. %0.0f%% accuracy on %d samples (conf %d%%)' %(untrained_days, 100*nb_correct/nb, nb, int(confidence*100))
            else:
                mae = t[0]
                # use nb_correct 1, ie map to bins. nb_correct 1 uses an abritrary 3Kwh maximum error
                s = '%d unseen days. %0.0f%% accuracy on %d samples (mae %0.1f)' %(untrained_days, 100*nb_correct1/nb, nb, mae)

            logging.info(s)
            print("GUI text", s)

            blynk_ev.blynk_v_write(vpins.v_unseen,s) 

            blynk_ev.blynk_v_write(vpins.v_terminal, s) # write also to terminal

        else:
            # untrained does not exist (did exit, but was concatenated to df_model for retraining. then retrying unseen too early (min 2 days)). wait a couple of days
            # NOTE: retrain delete untrained.csv
            s = "%s does not exist. likely was deleted as part of retrain, and too early. please try again in one day"
            logging.error(s)
            print(s)

            blynk_ev.blynk_v_write(vpins.v_terminal, s) # error message to terminal


        inference.update_GUI_model_tab() # label and metrics

    else:

        # retrain will contatenate unseen into df_model, and delete unseen
        print("cannot use unseen option if already using retrain")


#######################################
# GOAL: plot evaluate dataframe 
#######################################

if arg['train'] or arg['retrain'] :

    # created by all training cases

    #print("\nsaving training summary to %s" %eval_csv)
    #evaluate_df.to_csv(eval_csv) # not needed

    evaluate_df = evaluate_df.set_index(evaluate_df.columns[0])
    #evaluate_df = evaluate_df.drop(["loss"], axis=1) # not in same range as others 0 to 1 need to spec axis, drop can be used for row as well

    # create plot
    evaluate_df.plot(figsize=(8,5), kind='bar', subplots=False, title="metrics", grid=True)

    # plot 
    plt.savefig(os.path.join(various_plots.save_fig_dir, model_name + "_metrics.png"))


###################
# all blynk cases
##################

if use_GUI:

    print("\nBlynk (inference, postmortem, charge or retrain). some sleeping before exit, to make sure blynk runs")
    sleep(10)

plt.show()

s = "application end"
print(s)
logging.info(s)

### THE END
print('\nthis is the end')
logging.shutdown()

sys.exit(0)

