#!/usr/bin/python

############################
# train model with synthetic data, and see if it improves
############################

import pandas as pd
import sys
import argparse
import logging
import os


import tensorflow as tf

import config_features
import config_model

import dataset
import model_solar
import enphase
import train_and_assess

# concatenate: retrain from scratch on concatenation of real and synthetic datasets. 
#   every batch will see both
#   individual datasets already shuffled

# continue: train on real data (ie load trained model), and continue training with synthetic data
#  option to freeze last layers (transfert learning)

# trained on synthetic only, used as app name
syn_model_name = "trained_on_syn_model"

tf.random.set_seed(1234)

root = './'
# debug, info, warning, error, critical
log_file = root + "synthetic_train.log"
print ("logging to:  " , log_file)

if os.path.exists(log_file) == False:
    with open(log_file, "w") as f:
        pass  # create empty file

# encoding not supported on ubuntu/jetson ?
try:
    logging.basicConfig(filename=log_file,  encoding='utf-8', format='%(levelname)s %(name)s %(asctime)s %(message)s',  level=logging.INFO, datefmt='%m/%d/%Y %I:%M')
except:
    try:
        logging.basicConfig(filename=log_file, format='%(levelname)s %(name)s %(asctime)s %(message)s',  level=logging.INFO, datefmt='%m/%d/%Y %I:%M')
    except Exception as e:
        print(str(e))
        sys.exit(1)


sys.path.insert(1, '../PABOU') # this is in above working dir 
try: 
    import pabou
except Exception as e:
    print('%s: cannot import modules in ..  check if it runs standalone. syntax error will fail the import' %__name__)
    exit(1)

categorical = config_model.categorical

app_name = config_model.app_name
input_feature_list = config_features.config["input_feature_list"] # select series for model input(s). handle multi head
today_skipped_hours = config_features.today_skipped_hours 
prod_bins=config_features.config["prod_bins"]
prod_bins_labels_str = config_features.config["prod_bins_labels_str"]
prod_bin_labels = [i for i in range(len(prod_bins)-1)]
retain = config_model.retain
epochs=config_model.epochs
batch_size = config_model.batch_size  # one dataset element = one batch of samples
nb_lstm_layer = config_model.nb_lstm_layer
nb_unit = config_model.nb_unit
nb_dense = config_model.nb_dense
use_attention = config_model.use_attention
dropout_value = config_model.dropout_value

acceptable_absolute_error = config_features.regression_acceptable_absolute_error
(watch, metrics, loss) = model_solar.get_metrics(categorical)

split=config_model.split
repeat = config_model.repeat
shuffle = config_model.shuffle 
days_in_seq = config_model.days_in_seq   
stride = config_model.stride
sampling = config_model.sampling 
seq_build_method = config_model.seq_build_method
selection = config_model.selection
hot_size = len(prod_bin_labels)  # len of softmax, aka one hot
seq_len = len(retain) * days_in_seq - today_skipped_hours # input is n days x temp, or [temp, humid] , etc .. last day ends early to be able to run inference late in afternoon for next day
seq_len = int(seq_len / config_model.sampling) # use for range, so must be int

model_name_cat = app_name + "_cat"
model_name_reg = app_name + "_reg"


features_input_csv = config_features.features_input_csv  # available from different modules



# synthetic data created. this app assumes it exists
synthetic_data_csv =config_model.synthetic_data_csv

print("\n######\ntest model with synthetic data %s and see if it improves\n######\n" %synthetic_data_csv)

#######################################################
# parse CLI arguments 
# set CLI in .vscode/launch.json for vscode
# set CLI with -xxx for colab
######################################################

def parse_arg(): # parameter not used yet
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-a", '--categorical', help='classification. default FALSE, ie regression', required=False, action="store_true")

    parser.add_argument("-r", '--retrain', help='retrain from scratch on larger dataset, ie the concatenation of real data (training set) and synthetic data. order of concatenation depends on -w default FALSE', required=False, action="store_true")
    parser.add_argument("-s", '--shuffle', help='when retraining, shuffle concatenated dataset. default FALSE', required=False, action="store_true")
    parser.add_argument("-w", '--swap', help='retraining: put real data before synthetic data when concatenating. contining training: train first on synthetic data, then continue on real data. default FALSE', required=False, action="store_true")

    parser.add_argument("-c", '--continue', help='continue training. order depends on -w. default FALSE', required=False, action="store_true")

    parser.add_argument("-u", '--unfreeze', help='when continuing training, freeze LSTM, existing classifier trainable. (note: when neither -u nor -n, entiere network is trainable). default FALSE', required=False, action="store_true")
    parser.add_argument("-n", '--new', help='when continuing training, freeze LSTM and replace existing classifier with new (trainable) one (ala transfert learning). (note: when neither -u nor -n, entiere network is trainable). default FALSE', required=False, action="store_true")

    # return from parsing is NOT a dict, a namespace . can access as parsed_args.new if stay as namespace
    parsed_args=parser.parse_args()
    
    parsed_args = vars(parsed_args) # convert object to dict. get __dict__ attribute
    #print('ARG: parsed argument is a  dictionary: ', parsed_args)
    
    """
    print('ARG: keys:' , end = ' ')
    for i in parsed_args.keys():
        print (i , end = ' ')
    print('\n')
    """
    
    return(parsed_args) # dict


##################
# SAVE model contained synthetic data (in case)
##################
def save_syn_model(model):
    if categorical:
        a = app_name + "_syn_cat"
    else:
        a = app_name + "_syn_reg"
        
    print("save model trained on real + synthetic data: %s" %a)
    pabou.save_full_model(a, model) # signature = False


########################
# concatenate synthetic and real data dataset into one larger dataset 
# only training part of real data
########################

def concatenate_ds(ds, ds_syn, shuffle=False):
    # think about order. syn, then real

    # retrain model from scratch with concatenation of real + syn
    # think about order
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset

    #### concatenates real data (training) dataset with synthetic data dataset
    # at this point, len(ds) and len(ds_real) are know (in batches)

    ###### flattening creates two problems
    # len is lost, need to set cardinality
    # .batch() creates extra dim    # TensorShape([1, 45, 5]) to # TensorShape([16, 1, 45, 5])
    
    # keep batch as it
    #ds = ds.unbatch().batch(1)
    #ds_real = ds_real.unbatch().batch(1)
    # len(ds), len(ds_real)  TypeError: The dataset length is unknown.

    nb_seq_real = len(list(ds.unbatch().batch(1).as_numpy_iterator()))

    print("vault: concatenate real data dataset (training only) with synthetic data dataset")
    print("vault: REAL before concatenation: input ", iter(ds).next()[0].shape)
    print("vault: REAL before concatenation: label ", iter(ds).next()[1].shape)
    print("vault: REAL ds element spec", ds.element_spec)
    print("vault: REAL ds %d sequences" %nb_seq_real)


    nb_seq_syn = len(list(ds_syn.unbatch().batch(1).as_numpy_iterator()))

    print("vault: SYNTHETIC before concatenation: input ", iter(ds_syn).next()[0].shape)
    print("vault: SYNTHETIC before concatenation: label ", iter(ds_syn).next()[1].shape)
    print("vault: SYNTHETIC ds element spec", ds_syn.element_spec)
    print("vault: SYNTHETIC ds %d sequences" %nb_seq_syn)


    # The input dataset and dataset to be concatenated should have compatible element specs
    assert ds.element_spec == ds_syn.element_spec

    # retrain model from scratch with concatenation of real + syn
    # think about order
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset


    #ds = train_ds_real.concatenate(ds_syn) # dataset as argument comes last

    ds = ds_syn.concatenate(ds)
    # ds is now concatenated (syn + real), ie  will see first syn data , then real data in ebery batch

    print("vault: after concatenation: input ",  iter(ds).next()[0].shape)
    print("vault: after concatenation: label ",  iter(ds).next()[1].shape)
    print("vault: element spec", ds.element_spec)
    assert ds.element_spec == ds_syn.element_spec
    # check did not loose any sequence
    nb_seq_total = len(list(ds.unbatch().batch(1).as_numpy_iterator()))
    
    assert nb_seq_total == nb_seq_real + nb_seq_syn

    print("concatenated: number on inputs: real data (training only): %d, synthetic data: %d. total: %d" %(nb_seq_real, nb_seq_syn, nb_seq_total))
    print("concatenated: %d batches. average batch size: %0.2f" %(len(ds), nb_seq_total/len(ds)))

    # days, nb_hours is not updated. leave as it. not important

    # set cardinality if len is lost (eg when flattening). as some code expect to do len(ds)
    #https://www.tensorflow.org/api_docs/python/tf/data/experimental/assert_cardinality
    #ds = ds.apply(tf.data.experimental.assert_cardinality(nb_seq_total))
    #assert ds.cardinality().numpy() == nb_seq_total
    #assert len(ds)==nb_seq_total

    ##### batch add a dimension if I try to rebatch after flattening to batch = 1
    # TensorShape([1, 45, 5])
    #ds = ds.batch(batch_size)
    # TensorShape([16, 1, 45, 5])
    
    # see if all batch are full
    print('check for incomplete batch')
    for i , (f,l) in enumerate(ds):
        if f.shape[0] != batch_size or l.shape[0] != batch_size:
            print("vault: dataset: batch #%d (of %d) is incomplete" %(i, len(ds)), f.shape, l.shape)
            print("vault: this batch misses %d sequences"  %(batch_size - f.shape[0]))

    # 2 incomplete batch, ie incomplete from the sources batches remains
    # eg 12948 sequences, 811 batches.  miss 1x14

    ################
    # rebatch
    # issues when leaving 2 incomplete batches , assert len(pred) == len(list(iter(ds.unbatch())))
    ################

    print("vault: rebatch to get rid of uncomplete batches")
    # get rid of multiple incomplete batches by flattening and rebatching
    # rebatch(N) is functionally equivalent to unbatch().batch(N), but is more efficient,
    ds = ds.unbatch().batch(batch_size)


    print('check again for incomplete batch')
    for i , (f,l) in enumerate(ds):
        if f.shape[0] != batch_size or l.shape[0] != batch_size:
            print("vault: rebatch: %d miss %d sequences" %(i, batch_size - f.shape[0]), f.shape, l.shape)
            
    # 810 batches, one incomplete = 12  

    assert nb_seq_total == len(list(ds.unbatch().batch(1).as_numpy_iterator())) 

    # for some reason, rebatch looses cardinality
    ds = ds.apply(tf.data.experimental.assert_cardinality(i+1))
    print("\ncombined synthetic and real data(training) dataset READY")


    # shuffle at the end ? , ie keep real and syn sequentially separated or not ?
    if shuffle:
        print("shuffling concatenated dataset")
        ds = ds.shuffle(nb_seq_total)
    else:
        print("DO NOT shuffle concatenated dataset")

    return(ds)

###################
# take a trained model as input (ie either trained on real data or synthetic data)
# returns a model with LSTM and classifier updated
#   LSTM frozen and new or existing classifier (both trainable)
#   unchanged (ie both LSTM and classifier still trainable)
###################

def update_model(model):

    if arg["unfreeze"]:

        ############################
        # freeze LSTM
        # unfreeze EXISTING classifier (dense + last layer)
        ###########################

        print("\nCONTINUE: kind of transfert learning, ie freeze LSTM but unfreeze EXISTING last classifier\n")
        # freeze all, and unfreeze existing dense classifier

        model.trainable = True
        for layer in model.layers[:classifier]:
            layer.trainable = False  

        # as soon as trainable is modified, need to recompile
        lr = 0.001
        lr = 0.0001 # smaller lr

        model.compile(optimizer=tf.keras.optimizers.Adam(lr),  loss=loss,  metrics=metrics)

    elif arg["new"]:

        ############################
        # freeze LSTM
        # replace existing classifier with new one (kind of fransfert learning)
        ###########################

        print("\nCONTINUE: kind of transfert learning, ie freeze LSTM but replace last classifier\n")

        # need to get rid of existing classifier

        nb_dense  = model.layers[-1].input.shape[-1] # TensorShape([None, 128])
        final =  model.layers[-1].output.shape[-1]

        # remove last 2
        x = model.layers[-3].output 

        ####################
        # create new classifier
        ####################

        x = tf.keras.layers.Dense(nb_dense, activation = "relu", name = 'new_classifier_dense') (x)

        if final == 1:
            # regression
            outputs = tf.keras.layers.Dense(1, name = 'new_classifier_final')(x)
            assert categorical is False
        
        else:
            outputs = tf.keras.layers.Dense(final, activation = 'softmax' , name ='softmax')(x) 
            assert categorical

        model = tf.keras.Model(inputs=model.layers[0].input, outputs=outputs, name = "transfert")

        # as soon as trainable is modified, need to recompile
        lr = 0.001
        lr = 0.0001 # smaller lr

        model.compile(optimizer=tf.keras.optimizers.Adam(lr),  loss=loss,  metrics=metrics)

    else:
        print("CONTINUE: entiere model is trainable\n")
        # no need to change anything, loaded model still trainable
        # no ned to compile

    return(model)


#####################################
# train a model on synthetic data ony
#####################################

def train_syn():
    ####################################
    # train a model on synthetic data only
    ###################################
    (train_ds_syn, val_ds_syn, test_ds_syn, ds_dict) = dataset.fixed_val_split(ds_syn, nb_seq, nb_hours, split, retain, batch_size)

    histo_prod, majority_class_percent, majority_y_one_hot = enphase.build_solar_histogram_from_ds_prior_training(ds_syn, prod_bins)

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
        "test_ds" : test_ds_syn
    }

    # no need to increase capacity

    # build model, incl compile
    model = model_solar.build_lstm_model(ds_syn, categorical,
    name='syn_base_lstm', units = nb_unit, dropout_value=dropout_value, num_layers=nb_lstm_layer, nb_dense=nb_dense, use_attention=config_model.use_attention)

    # train model on synthetic data
    model, history, elapse = train_and_assess.train(model, epochs, ds_syn, val_ds_syn, model_param_dict, checkpoint = False, verbose=0)
    print("model trained on synthetic data only")

    return(model)





##################################
# main
##################################

arg= parse_arg()

if arg["categorical"]:
    categorical = True
    model_name = model_name_cat
else:
    categorical = False
    model_name = model_name_reg


logging.info("start training with synthetic data. categorical: %s. arg: %s" %(categorical, arg))


#########################
# STEP 1: create real data dataset
#########################

print('\nload saved df model/feature input from csv %s' %features_input_csv)
df_model =  pd.read_csv(features_input_csv)

# "simulate" smaller dataset to see impact of synthetic data ?
## do I need to increase model capacity because larger training set ?

print("\nbuild REAL DATA sequence dataset from df_model") 
ds, nb_seq, nb_hours = \
dataset.build_dataset_from_df(df_model, \
input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
stride=stride, sampling=sampling, shuffle=shuffle, categorical=categorical)

# number of sequence in real dataset 
assert nb_seq == len(list(ds.unbatch().batch(1).as_numpy_iterator())) # this takes a few sec to run
# len(ds) is nb batch
print("\nreal data: dataset %d batches, %d sequences. ratio: %0.2f" %(len(ds), nb_seq, nb_seq/len(ds)))

# test_ds is real data
(train_ds, val_ds, test_ds, ds_dict) = dataset.fixed_val_split(ds, nb_seq, nb_hours, split, retain, batch_size)

histo_prod, majority_class_percent, majority_y_one_hot = enphase.build_solar_histogram_from_ds_prior_training(ds, prod_bins)

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

#########################
# STEP 2: load baseline model, trained on real data  (tf SavedModel)
#########################
print("\nload current model (trained on real data only")

# load model and get accuracy (and later check if/how improved)
try:
    model, h5_size, _ = pabou.load_model(model_name, 'tf')

    if model is  None :
        raise Exception ("!!!!!! CANNOT LOAD MODEL ")
except Exception as e:
    raise Exception ("!!!!!! CANNOT LOAD MODEL ", str(e))


#####################
# get layer number for dense classifier
#####################

classifier = None # will be int

# use real data model. is the same for model trained on synthetic data

for i, l in enumerate(model.layers):
    print(i, l.name)
    if l.name == "dense":
        classifier = i

print("\ndense classifier is layer number:", classifier)

(watch, metrics, loss) = model_solar.get_metrics(categorical)

######################################################
# baseline, ie metrics for real model. use to check if we improved.
##################################################### 
model_metrics_before_retrain_dict  = model.evaluate(test_ds, return_dict=True, verbose=0)
s= "\nabout to use synthetic data. metrics on real data (test set):%s" %model_metrics_before_retrain_dict
print(s)
logging.info(s)


###########################
# STEP 3: creates synthetic dataset
############################

# load synthetic data already created OFFLINE
# validation of syn data done at generation time
print("\nloading synthetic data from %s" %synthetic_data_csv)
try:
    df_model_syn =  pd.read_csv(synthetic_data_csv)
except Exception as e:
    print("cannot load synthetic data %s" %str(e))
    raise Exception ("make sure to run synthetic data generation before using this option")

# build dataset from syn data
print("\ncreate dataset from synthetic data")
ds_syn, nb_seq_syn, nb_hours_syn = \
dataset.build_dataset_from_df(df_model_syn, \
input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
stride=stride, sampling=sampling, shuffle=shuffle, categorical=categorical)

assert nb_seq_syn == len(list(ds_syn.unbatch().batch(1).as_numpy_iterator()))
print("syn data dataset %d batches, %d sequences; ratio %0.2f" %(len(ds_syn), nb_seq_syn, nb_seq_syn/len(ds_syn)))

# test validation split done if training. 

#########################
# STEP 4: load model trained on synthetic data only (tf SavedModel)
# if does not exist, train it and save
#########################
print("\nload model trained on synthetic data only, ie %s" %syn_model_name)

try:
    syn_model, h5_size, _ = pabou.load_model(syn_model_name, 'tf')
    # exception caugth in pabou.load_model, and returns none

    if syn_model is  None :  
        print ("\n!!! cannot load %s. train it" %syn_model_name)
        syn_model = train_syn()
        print("save model trained on synthetic data only: %s" %syn_model_name)
        pabou.save_full_model(syn_model_name, syn_model) # signature = False
       
except Exception as e:
    print ("\n!!!! cannot load %s. train it" %syn_model_name)
    syn_model = train_syn()
    print("save model saved on synthetic data only: %s" %syn_model_name)
    pabou.save_full_model(syn_model_name, syn_model) # signature = False

# syn_model is trained on synthetic data only


# we have now 2 datasets, and two trained models


###########################
# STEP 5: process retrain from scratch on larger dataset or continue training
############################


if arg["retrain"]:

    ###############################
    # retrain on larger (concatenated) dataset
    # order of concatenation matters
    # shuffle concatenated  ? (ie mix real and synthetic sequences in every batch)
    # increase network capacity
    ##############################

    shuffle = arg["shuffle"]

    print("\nconcatenate real data (training) and all synthetic data into larger dataset. shuffle %s" %shuffle)

    # get larger dataset
    # try both order
    if arg["swap"]:    
        larger_ds = concatenate_ds(ds_syn, train_ds, shuffle=shuffle)
    else:
        larger_ds = concatenate_ds(train_ds, ds_syn, shuffle=shuffle)

    # increase capacity
    nb_unit = 2 * nb_unit
    nb_dense = 2 * nb_hours

    # build model, incl compile
    model = model_solar.build_lstm_model(larger_ds, categorical,
    name='base_lstm', units = nb_unit, dropout_value=dropout_value, num_layers=nb_lstm_layer, nb_dense=nb_dense, use_attention=config_model.use_attention)

    # train on larger dataset
    # use real data(training) + entiere synthetic ds as training set
    # use real data (validation)
    # real data test set if set aside
    print("train larger model from scratch on larger dataset")
    model, history, elapse = train_and_assess.train(model, epochs, larger_ds, val_ds, model_param_dict, checkpoint = False, verbose=0)


if arg["continue"]:

    ######################################
    # train on one dataset, then continue training on the other one
    # option 1: use model trained on real data (ie loaded model), then continue training on synthetic data
    # option 2: train on synthetic data, then continue training on real data
    # when continuing
    #   freeze LSTM, and either swap classifier with new one (ala transfert lerning), or use existing classifier (ie make it trainable).
    #   or keep entiere model trainable 
    #######################################

    # loaded model is already trained on real data

    if arg["swap"]:

        ############################
        # start with synthetic data and continue with real data
        ############################
        print("\ncontinue training with real data (test set), on model already trained on synthetic data")

        # update classifier/LSTM of model already trained
        model = update_model(syn_model)

        # continue training with real data (training only)
        ds_to_continue_training = train_ds

    else:

        ############################
        # start with real data (already trained), and continue with synthetic data
        ############################

        print("\ncontinue training with all synthethic data, on model already trained on real data")

        # update classifier/LSTMof model already trained
        model = update_model(model)

        # continue training with all synthetic data
        ds_to_continue_training = ds_syn


    # continue training on model, with ds_to_continue_training
    # validation is always val_ds
        
    # model can be either already trained on real or on synthetic data
    # depending, ds_to_continue_training will be synthetic or real

    model, history, elapse = train_and_assess.train(model, epochs, ds_to_continue_training, val_ds, model_param_dict, checkpoint = False, verbose=0)
    


############################
# STEP 5: evaluate model trained with synthetic data
############################
# use test set from real data
        
s = "model trained with synthetic data. evaluate on real data test set"
print("\n%s" %s)
logging.info(s)


# evaluate on REAL data test set
#(metrics_test_syn, metrics_train_syn, ep) = train_and_assess.metrics_from_evaluate(model, test_ds, larger_ds, history)
#print("metrics on REAL DATA test set", metrics_test_syn)

ret = model.evaluate(test_ds, return_dict=True, verbose=0)
s= "with synthetic data, metrics on REAL DATA test set: %s" %ret
print(s)
logging.info(s)

s= "without synthetic data, metrics on REAL DATA test set: %s" %(model_metrics_before_retrain_dict)
print(s)
logging.info(s)

if categorical:
    syn_acc = ret["categorical accuracy"]
    real_acc =  model_metrics_before_retrain_dict["categorical accuracy"]

    if syn_acc > real_acc:
        s = "!!!!!!!! HOURRA. acc increased from %0.2f to %0.2f, ie %0.2f%% improvement" %(real_acc, syn_acc, 100*(syn_acc-real_acc)/real_acc)
        print("\n\n%s\n\n" %s)
        logging.info(s)
        logging.info(arg)

        #save(model)
    
    else:
        s= "))))))))) sorry no improvement. categorical: %s" %categorical
        print(s)
        logging.info(s)



else:
    syn_mae = ret["mae"]
    real_mae = model_metrics_before_retrain_dict["mae"]

    if syn_mae < real_mae:
        s = "!!!!! HOURRA. mae decreased from %0.2f to %0.2f, ie %0.2f%% improvement" %(real_mae, syn_mae, 100*(real_mae-syn_mae)/real_mae)
        print("\n\n%s\n\n" %s)
        logging.info(s)
        logging.info(arg)

        #save(model)
    
    else:
        s= "))))))))) sorry no improvement. categorical: %s" %categorical
        print(s)
        logging.info(s)

