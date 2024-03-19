#!/usr/bin/env python3

#Google Colab notebooks have an idle timeout of 90 minutes and absolute timeout of 12 hours. 
#This means, if user does not interact with his Google Colab notebook for more than 90 minutes, 
#its instance is automatically terminated. Also, maximum lifetime of a Colab instance is 12 hours

# rubber ducky USB device


###################################
# brutal (combinatorial) search of best inpout and output feature
# implmenet end to end, ie feature load, dataset build, split, train, evaluate
# move as a separate app, to keep solar.ev tidy
# hard to integrate in solar.ev, as this loop around train
###################################

import pandas as pd
import os
import sys
import datetime

import dataset
import enphase
import train_and_assess
import model_solar
import config_model
import config_features
import record_run

import brutal_force_space # "search space"

p = "../my_modules"
sys.path.insert(1, p)
try:
    from my_decorators import dec_elapse
    print("import from %s OK" %p)
except:
    print('%s: cannot import modules from %s. check if it runs standalone. syntax error will fail the import' %(__name__, p))
    exit(1)


#########
# brutal force search results
#########
search_result_csv = "result_search_%d_%d_%d_%d.csv" %(datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute)  # manual hypersearch. 
search_result_csv = os.path.join("search_result", search_result_csv)

# WARNING: CLI for solar2ev
features_input_csv = "features_input.csv"

################################################
# GOAL: brutal force search 
# input features and model param/hyperparam
#################################################

@dec_elapse
def brutal_force():

    # goes thru ENTIRE cycle, ie build dataset, model, train, evaluate, record run

    #  categorical, prod_bins, input_feature_list
    #  day_in_seq, sampling, seq_build_method, (stride or selection depending on method)
    #  nb_units, nb_lsmt_layers, nb_dense
    #  run, shuffle


    # other param (needed below in train cycle)
    split = config_model.split
    epochs = config_model.epochs
    batch_size = config_model.batch_size
    dropout_value = config_model.dropout_value
    repeat = config_model.repeat
    retain = config_model.retain
    today_skipped_hours = config_features.today_skipped_hours


    # lstm build uses this
    use_attention = config_model.use_attention


    print("\n\n##########\nBRUTAL FORCE SEARCH OF VARIOUS INPUT/OUTPUT FEATURES, MODEL TYPE and HYPERPARAMETERS\n##########\n")
    print('result in %s' %search_result_csv)

    # better run on colab

    df_model =  pd.read_csv(features_input_csv)

    best_metric_classification = 0   # accuracy for classification, 
    best_metric_regression = 1000000   # rmse for regression
    best_row_classification = []
    best_row_regression = []

    i = 0 # runs

    best_run = -1

    ######################
    # nested for loop for search space
    ######################

    for categorical in brutal_force_space.categorical_h:
        for prod_bins in brutal_force_space.prod_bins_h:
            for input_feature_list in brutal_force_space.feature_list_h:
                for days_in_seq in brutal_force_space.days_in_seq_h:

                    for sampling in brutal_force_space.sampling_h:

                        for nb_unit in brutal_force_space.nb_unit_h:
                            for nb_lstm_layer in brutal_force_space.nb_lstm_layer_h:
                                for nb_dense in brutal_force_space.nb_dense_h:

                                    for run in range(brutal_force_space.nb_run):
                                        for shuffle in brutal_force_space.shuffle_h:

                                            for seq_build in brutal_force_space.seq_build_h:
                                                seq_build_method = seq_build[0]
                                                if seq_build_method in [1]:
                                                    stride = seq_build[1]
                                                    selection = [] # need to be set as a calling parameter of function building dataset (even if not used in this case)
                                                else:
                                                    selection = seq_build[1]
                                                    stride = -1  # need to be set as a calling parameter of function building dataset
                                                
                                                for use_attention in brutal_force_space.use_attention_h:

                                                    # The nonlocal keyword is used to work with variables inside nested functions, where the variable should not belong to the inner function.
                                                    #nonlocal i, best_metric_classification,  best_metric_regression,  best_row_classification, best_row_regression
                                                                            
                                                    print("\n===>> SEARCH %d. categorical: %s. %d bins. %d heads. method: %d. (sampling %d / selection %s). %d days" %(i,categorical, len(prod_bins)-1, len(input_feature_list), seq_build_method, sampling , selection, days_in_seq))
                                                    i = i + 1

                                                    # seqlen depends on nb of days in sequence, and sampling
                                                    seq_len = len(retain) * days_in_seq - today_skipped_hours # input is n days x temp, or [temp, humid] , etc .. last day ends early to be able to run inference late in afternoon for next day
                                                    seq_len = int(seq_len / sampling) # use for range, so must be int

                                                    prod_bin_labels = [x for x in range(len(prod_bins)-1)]
                                                    # [0, 1, 2, 3] used in pandas 

                                                    hot_size = len(prod_bins) -1

                                                    ds, nb_seq, nb_hours = \
                                                    dataset.build_dataset_from_df(df_model, \
                                                    input_feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
                                                    stride=stride, sampling=sampling, shuffle=shuffle, labels=True, categorical=categorical)

                                                    histo_prod, majority_class_percent, majority_y_one_hot = enphase.build_solar_histogram_from_ds_prior_training(ds, prod_bins)

                                                    train_ds, val_ds, test_ds, ds_dict = dataset.fixed_val_split(ds, nb_seq, nb_hours, split, retain, batch_size)

                                                    model = model_solar.build_lstm_model(ds, categorical, name='base_stm', units = nb_unit, dropout_value=dropout_value, num_layers=nb_lstm_layer, nb_dense=nb_dense, use_attention = use_attention)

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
                                                            "stride": stride,
                                                            "sampling": sampling,
                                                            "shuffle": shuffle,
                                                            "categorical": categorical,
                                                            "majority_class_percent": majority_class_percent,
                                                            "majority_y": majority_y_one_hot,
                                                            "nb_hours" : nb_hours
                                                        }

                                                    print("\n==>> about to train %d out of total %d training" %(i, brutal_force_space.space_size))
                                                    model, history, elapse = train_and_assess.train(model, epochs, train_ds, val_ds, model_param_dict, checkpoint=False, verbose=0)

                                                    (metrics_test, metrics_train, ep) = train_and_assess.metrics_from_evaluate(model, test_ds, train_ds, history)

                                                    ##### record training run. one entry per training run
                                                    # header defined in record_run.record_param_hdr
                                                    # use str(list) to avoid dim issues Shape of passed values is (18, 1), indices imply (18, 17)

                                                    record_run_param = [str(categorical),
                                                        len(prod_bins)-1, str(input_feature_list), days_in_seq, 
                                                        ds_dict["samples"]["train"],
                                                        seq_build_method, str(selection), stride,
                                                        metrics_test[0],metrics_test[1], metrics_test[2], metrics_test[3],
                                                        run, sampling, repeat, shuffle,
                                                        nb_lstm_layer, nb_unit, nb_dense, use_attention,
                                                        elapse, ep
                                                        ]
                                                    
                                                    assert len(record_run_param) == len(record_run.record_param_hdr)

                                                    # save this run to csv
                                                    if (record_run.record_training_run(search_result_csv, record_run_param)):
                                                        print(pd.read_csv(search_result_csv).tail(1))
                                                    else:
                                                        print("ERROR recording search result; EXIT")
                                                        sys.exit(1)

                                                    if categorical:
                                                        improved  = metrics_test[0] > best_metric_classification   # acc increased
                                                    else:
                                                        improved  = metrics_test[0] < best_metric_regression   # rmse decreased

                                                    if improved: # ie best model so far

                                                        print(">>>>>>>>>>> run %d improved performance" %i) 

                                                        best_run = i
                                                        
                                                        if categorical:
                                                            print(">>>>>>>>>>> improved categorical metric from %0.5f to %0.5f" %(best_metric_classification, metrics_test[0]))
                                                            best_metric_classification  = metrics_test[0]
                                                            best_row_classification = metrics_test
                                                        else:
                                                            print(">>>>>>>>>>> improved regression metric from %0.5f to %0.5f" %(best_metric_regression, metrics_test[0]))
                                                            best_metric_regression  = metrics_test[0]
                                                            best_row_regression = metrics_test

                                                        r  = model.evaluate(test_ds, return_dict=True, verbose=0)
                                                        print(">>>>>>>>>> metrics from evaluate() on just improved, ie best model so far", r)

                                                        #print(">>>>>>>>>>>> saving best model so far")
                                                        #pabou.save_full_model(app, model) # signature = False
                                                    else:
                                                        print("run %d did not improved" %i)
                                                        # not improved

                                                    print("\n#############\nbest run so far %d: metrics on test ds: classification %s, regression %s\n" %(best_run, best_row_classification, best_row_regression))


    print("\n\nBrutal search ended")
    print("ran %d experiments. best run overall: classification: %s. regression %s" %(i, best_row_classification, best_row_regression))

if __name__ == "__main__":

    print("\n\n************** brutal force exploration of best input/output features **************\n")
    print("search space is described in %s" %"brutal_force_space")
    print("please use keras tuner to further explore hyperparameters")
    print("\nwhen returning, look at %s" %search_result_csv)
    print("nb of training: %d" %brutal_force_space.space_size) # 100 to 400 sec / training

    print("\n****** go and have a nap ******")

    brutal_force()

    print("this is the end. good buy")





"""
feature_list_h = [

    [['temp', 'pressure', 'cos_hour', 'sin_hour']],
    [['temp', 'pressure', 'production', 'cos_hour', 'sin_hour']],

    [['temp', 'pressure', 'production', 'cos_hour', 'sin_hour'], ['wind','direction', 'cos_hour', 'sin_hour']],

    [['temp', 'production', 'cos_hour', 'sin_hour'], ['pressure', 'production', 'cos_hour', 'sin_hour']], 
    [['temp', 'production', 'humid', 'cos_hour', 'sin_hour'], ['pressure', 'production', 'humid', 'cos_hour', 'sin_hour']],

    [['temp', 'production', 'humid', 'direction', 'cos_hour', 'sin_hour'], ['pressure', 'production', 'humid', 'direction', 'cos_hour', 'sin_hour']],

    [['temp', 'production', 'cos_hour', 'sin_hour', 'sin_month'], ['pressure', 'production', 'cos_hour', 'sin_hour', 'sin_month']], 

    [['temp', 'cos_hour', 'sin_hour'], ['pressure', 'cos_hour', 'sin_hour'], ['production', 'cos_hour', 'sin_hour'] ],
    [['temp', 'direction', 'cos_hour', 'sin_hour'], ['pressure', 'direction', 'cos_hour', 'sin_hour'], ['production', 'direction', 'cos_hour', 'sin_hour'] ],

    [['temp', 'cos_hour', 'sin_hour', 'sin_month'], ['pressure', 'cos_hour', 'sin_hour', 'sin_month'], ['production' ,'cos_hour', 'sin_hour', 'sin_month'] ],
    

    ]
"""