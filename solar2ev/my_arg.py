#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
#!/usr/bin/env python
"""

import argparse
#######################################################
# parse CLI arguments 
# set CLI in .vscode/launch.json for vscode
# set CLI with -xxx for colab
######################################################

def parse_arg(): # parameter not used yet
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #parser.add_argument("-p", '--process', type=str, choices=['load', 'transfert', 'maker'], help='optional either load and run model or generate one with transfert learning or with model maker. default load', required=False, default="load")
    #parser.add_argument("-a", '--app', type=str, help='optional. application name. used to store training artifacts. default meteo.', required=False, default='meteo')
    #parser.add_argument("-t", '--threshold', type=int, help='optional. threshold in %. default 60%.', required=False, default = 60)

    # The store_true option automatically creates a default value of False.
    # returned argument is boolean. action="store_true" means default is False
    # nothing, 'windowed':False -w  'windowed': True   
    parser.add_argument("-g", '--get_features', help='combine meteo and solar csv to create features. default FALSE, ie load feature csv', required=False, action="store_true")
    parser.add_argument("-t", '--train', help='train model. default FALSE', required=False, action="store_true")
    parser.add_argument("-b", '--bench', help='run benchmark. optional. default False (ie do not run benchmarks)', required=False, action="store_true")
    
    parser.add_argument("-i", '--inference', help='run inference for today. optional. default False', required=False, action="store_true")
    parser.add_argument("-m", '--postmortem', help='validate last inference against thruth and build rolling list for GUI. default False (ie do not set)', required=False, action="store_true")
    parser.add_argument("-c", '--charge', help='set up charger based on inference. default False (ie do not set)', required=False, action="store_true")

    #parser.add_argument("-f", '--features', type=str, help='optional. name of feature csv file used for training. default features_input.csv.', required=False, default='features_input.csv')
    
    parser.add_argument("-u", '--unseen', help='build incremental (unseen) input features and test model accuracy with unseen data. default False', required=False, action="store_true")
    parser.add_argument("-r", '--retrain', help='build and merge incremental (unseen) input features and retrain model FROM SCRACTH. default False (ie do not set)', required=False, action="store_true")
   
    # both unseen and retrain will start by creating unseen input
    # retrain from scratch (vs continue training)
    # unseen run inference and update GUI text widget

    # move as standalone
    #parser.add_argument("-s", '--search', help='brutal force search best features. default False (ie do not set)', required=False, action="store_true")
    
    parser.add_argument("-k", '--kerastuner', help='use keras tuner to search for hyperparameters. optional. default False (ie do not use)', required=False, action="store_true")
    parser.add_argument("-s", '--synthetic', help='create synthetic data (SDV). optional. default False (ie do not use)', required=False, action="store_true")

    parser.add_argument("-p", '--plot', help='plot many view of training data. optional. default False', required=False, action="store_true")

    parser.add_argument("-l", '--test_blynk', help='test Blynk. update model tab. optional. default False', required=False, action="store_true")


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


    """
    .vscode/launch.json
    "configurations": [
        {
            "args":[
            // those option are for fit and benchmark only
            //    "-pmaker",
                "-pload",
            //    "-ptransfert",
                "-b",

            // applies to both
            // -e runs in headless, ie will not img_show() , nothing: in windowed
            //    "-e",

            // those option are for run only

                "-t60", // threshold *100. for non int8 models only

            // model type
            //    "-mtf",
                "-mli", 

            // for TFlite, type of quantization

                "-qTPU_edgetpu",
            //    "-qfp32",
            //    "-qGPU",
            //    "-qTPU",

            ],
    """

"""

"configurations": [
        {

            "args":[
                    // get features.  ie scrap or use existing meteo csv, use downloaded solar production, create meteo/solar  df, combine into feature csv. default no, ie load existing feature csv
                    //"-g",

                    // run inference for today 
                    //"-i",

                    //post mortem for previous days
                    //"-m",

                    // create delta feature, and retrain model
                    //"-r",

                    // creates delta features and test on unseen data
                    //"-u",

                    // keras tuner, search model hyperparameters
                    //"-k",

                    // synthetic data vault: fit model on all features, sample model to create synthetic data
                    //"-v",

                    // build dataset from feature cvs, train and evaluate
                    "-t",

                    // benchmarck
                    //"-b",

                    // model name
                    "-asolar2ev",

                    // plot data 
                    //"-p",

            ],
    """


