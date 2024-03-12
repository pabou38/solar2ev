
#!/usr/bin/env python3

################################
# configuration for FEATURES to be used
#   input features (data,column, heads) 
#   output features (bins for categoricals, max error for regression)
################################

# NOTE: categorical is defined in config_model

# use dict variable
# use single quote ' for strings

# feature_list is list of head(s). each head is a list of feature
# input_feature_list: LIST of one TUPLE (head) or multiple TUPLE separated by ,
#  eg [("temp", "pressure", "production", "sin_month", "cos_hour")]  or [(), (), ()]
# use solar production quantile (eg quartile) to get same number of samples in each bin
# last interval in unbound to avoid any issue with unexpected production (eg when using synthetic data)


# bins are used for both classification (categorical) and regression
# for classification, prediction is already on index into prod_bins_labels_str and charge_hours
# for regression, prediction is a scalar, mapped to prod_bins to test accuracy and get charge hours


###########################
# features, bins and charge
###########################

# NOTE: look at latest run to get bins boundaries (if you want balanced bins)

#"prod_bins":[0, 11.06 , 22.49, 1000000],
#"prod_bins":[0, 6.3, 13.7, 20.6, 26.1, 1000000],

_ = ('temp', 'production', 'sin_month', 'cos_month', 'pressure')

config = {

"input_feature_list": [

('temp', 'production', 'sin_month', 'cos_month', 'pressure')
  
],

"prod_bins":[0, 7.83, 17.37, 25.14, 1000000],
"prod_bins_labels_str" :['0-8Kwh', '8-17Kwh', '17-25Kwh', '>25Kwh'],
"charge_hours":[4,3,1,0],

}


assert len(config["charge_hours"]) == len(config["prod_bins_labels_str"]), "error defining charge and labels"
assert len(config["prod_bins_labels_str"]) == len(config["prod_bins"]) -1, "error defining production bins"

print("model uses: %d bins" % (len(config["prod_bins"]) -1)) 


# for regression. in kwh. ABSOLUTE error (vs RMS)
# used by pabou.bench_full() and inference.predict_on_ds_with_labels()
# other way is to map both prediction and target scalar to bins used for categorical
# set based on what is acceptable for the application and/or observed val_mae
regression_acceptable_absolute_error = 3.0

# number of 'today''s hours not needed. ie today (inference)/last day(training) only requires 24-n hours of meteo and solar
# allows to run inference late afternoon
# set for when today's solar productin is known, ie as soon as sun is set
# 6 means can run inference starting at 6 in the afternoon (call it 7 for various data source to get updated)
today_skipped_hours = 6 


# for daily inference, and not postmortem_next_day, cannot run inference_from_web_at_a_day() until this time
# both need solar production from previous day (as well as day before), and enphase cloud for daily production is not updated until 7:30 pm CET for previous day
# could avoid this problem by getting previous day solar with telemetry API (as for current day), well ...
not_until_h = 19
not_until_mn = 30

# False: run postmortem at the same time as daily inference, ie as if one day ago (ie) the day before, to predict today. time limit (to get today's production and yesterdat from cloud daily API)
#    Feb 7th in the evening. run as if 6th, to predict 7th.   5th solar is updated in cloud, 6th is not yet , but using telemetry.

# True: run postmortem the day after daily inference (no time limit). so as if THREE days ago, predicting TWO days ago.
#  Feb 8th in the morning. as if 5th, to predict 6th. 6th solar is already updated in cloud (and as ground truth, got the same time as other solar), 7th is not yet

# so even if running on the 8th, will get "less fresh" validation. the advantage is 1) run anytime 2) save a call to telemetry
post_mortem_next_day = True

# use to take action on inference result
# if GUI not available (set by GUI otherwize)
confidence_threshold = 0.6 # for classification


############################
# GUI and misc parameters
############################

# use GUI
blynk_GUI = True

# charging behavior if GUI not available (set by GUI otherwize)
# 0 = auto, ie use result of inference if confidence high enough, 1 = on regardless of inference, 2 = off regardless of inference
default_charge = 0


#max_expected_production = 33 
# all value above are consider wh (not kwh) and divided by 1000
# enphase reporting idiosyncrasie
# deprecated. parse string to spot "," which means kwh

# to use alternate ways to get solar/meteo
features_input_csv  = 'features_input.csv'


# meteo station being queried
station_id = 278

