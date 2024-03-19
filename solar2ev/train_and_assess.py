#!/usr/bin/env python3

import sys
import os

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report # precision, accuracy

from scipy.stats import shapiro# to test if distribution is gaussian

import matplotlib.pyplot as plt

import logging

from time import time
import json

from typing import Tuple

import model_solar
import create_baseline_models

import config_model

import shared
model_type = shared.model_type # enum, with values as str, to be used as key when creating metric json file  model_type.cat (enum) , model_type.cat.value ("cat")

sys.path.insert(1, '../PABOU') # this is in above working dir 
try: 
    import pabou
    import various_plots
except Exception as e:
    print('%s: cannot import modules in ..  check if it runs standalone. syntax error will fail the import' %__name__)
    exit(1)

save_fig_dir = various_plots.save_fig_dir 

sys.path.insert(1, '../my_modules') # this is in above working dir 
try: 
    from my_decorators import dec_elapse
except Exception as e:
    print('%s: cannot import modules in ..  check if it runs standalone. syntax error will fail the import' %__name__)
    exit(1)

################ 
# GOAL: ONE TRAINING. 
# return history, elapse
################

@dec_elapse
def train(model, epochs, train_ds , val_ds, model_param_dict, verbose = 1, checkpoint=True) -> Tuple[dict,float]:

    # call keras fit
    # validation split already done
    # can be called multiple time, because of different validation strategies, ie fixed, k fold etc ..
    # categorical used to query metric to watch (for callback) - could also use pabou.is_categorical() 
    # callback list created here

    # use of checkpoint callback optional
  
    s= "Start training on max %d epochs" %epochs
    print(s)
    logging.info(s)

    categorical = model_param_dict["categorical"]
    repeat = model_param_dict ["repeat"]
    ds_dict = model_param_dict["ds_dict"]

    (watch, metrics, loss) = model_solar.get_metrics(categorical)

    print("dataset dict:", ds_dict)
    print("metric for early stop and reduce lr: ", watch)
    print("configured metrics: ", metrics)
    # watch: metric used by early stopping and reduce lrn on plateau
    # rather monitor loss than accuracy. accuracy not meaningfull for inbalanced dataset ?

    # checkpoint bool: enable saving model or weigth (best only or all) during training
    # if enabled, will log to console status of improvement (saving if improved, or no improvement)
    # dir and file names managed in get_callback_list()

    # use of checkpoints callback defined here
    callbacks_list = pabou.get_callback_list(early_stop = watch, reduce_lr = watch, checkpoint=checkpoint, tensorboard_log_dir = './tensorboard_dir')

    # show validation metrics on same line at end of each epochs
    callbacks_list.append(model_solar.CustomCallback_show_val_metrics()) # ie end epochs

    print("callbacks list (incl custom):\n" , [x.__class__ for x in callbacks_list])


    if repeat != 1:
        # WARNING: ######## NOT A GOOD IDEA. OVERFIT
        # add more data, our training set is small ?????
        # REPEAT (shuffle done when buiding dataset)

        #buffer_size = repeat*len(train_ds)*batch_size # elements
        train_ds= train_ds.repeat(repeat)
        
        #buffer_size = repeat*len(val_ds)*batch_size # elements
        val_ds= val_ds.repeat(repeat)

    # should match trace of training 
    # WARNING: set globals, or variable changed outside will not PRINT correctly
    # train will be fine

    print('\ntraining dataset: %d batch, %d samples. repeat: %d.' 
    %(len(train_ds), ds_dict["samples"]["train"]*repeat, repeat))

    # if using dataset , should be a single one

    #A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
    #A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
    #A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
    #A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
    #A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights).

    print('\n===> fit with %d batches. max %d epochs\n' % (len(train_ds), epochs))

    logging.info("start training (model.fit)")

    start = time()

    # Keras fit is here
    
    # can shuffle in .fit() if not done before
    # custom callback will print metrics on the terminal (same line) . BEWARE: not best, just current epoch

    history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks_list, 
    verbose = verbose
    )

    # keras fit end
    """
    # printed by tf
    Restoring model weights from the end of the best epoch: 32., val_msle: 0.160, 
    Epoch 39: early stopping05, val_mse: 15.251, val_mae: 2.578, val_msle: 0.141, 
    Training end: val_rmse: 3.905, val_mse: 15.251, val_mae: 2.578, val_msle: 0.141,
    WARNING: Training end is LAST epoch, not BEST epoch
    """

    elapse = time() - start

    logging.info("training ended")

    # actual epoch are different from max epoch
    # also different from best epoch

    assert len(history.history["loss"]) == len(history.history["lr"])
    actual_epochs = len(history.history["loss"])

    s= "fitted in %0.2f sec, stop at epoch %d, ie %0.1f sec/epochs. max epoch: %d" %(elapse, actual_epochs, elapse/actual_epochs, epochs)
    print(s)
    logging.info(s)


    # history dict contains met + val_met + lr + loss
    # met are needed when compiling model. 
    # met = model_solar.get_metrics(categorical)
    print('training: history (keys): ', history.history.keys())

    # do I need to return the trained model ?

    return(model, history, round(elapse,1))



#############
# wrapper for .evaluate() - on test and train dataset
# return metrics list for test and training set, nb epochs 
#############

def metrics_from_evaluate(model, test_ds, train_ds, history):

    categorical = pabou.is_categorical_ds(test_ds)

    (metric_to_watch, metrics, loss) = model_solar.get_metrics(categorical)

    # Returns the loss value & metrics values for the model in test mode.
    # returns Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). 
    # The attribute model.metrics_names will give you the display labels for the scalar outputs.

    # evaluate on ANY dataset.
    # metrics returned : no val

    #model.metrics_names ['loss', 'categorical accuracy', 'precision', 'recall', 'prc']
    # metrics from get_metrics do not include loss

    # use same colums as the one used for evaluate dataframe definition
    columns=[m.name for m in metrics] # aka model.metrics_names

    print("will call evaluate(). configued metrics:", columns) 
    # evaluate. metrics: ['categorical accuracy', 'precision', 'recall', 'prc']

    actual_epochs = len(history.history['loss'])
    nb_samples = len(list(test_ds.unbatch().batch(1).as_numpy_iterator()))

    ##### EVALUATE model on test set
    r  = model.evaluate(test_ds, return_dict=True, verbose=0) # return loss and all metrics as either a dict or a list
    # metrics returned by evaluate (no loss, no val_)

    # returns list without any labels. assumes order is the same as in definition of metrics in config_model.py
    m_test = [round(r[m],3) for m in columns ]
    print('evaluate() on test ds: %d batches %d samples. metrics: %s' %(len(test_ds), nb_samples, m_test))

    # evaluate on train set to check overfit
    r  = model.evaluate(train_ds, return_dict=True, verbose=0) # return loss and all metrics as either a dict or a list
    m_train = [round(r[m],3) for m in columns ]
    print('evaluate() on train ds: %d batches %d samples. metrics: %s. !! LESS MEANINGFULL' %(len(test_ds), nb_samples, m_train))

    return(m_test, m_train, actual_epochs)

###########################
# GOAL: compute loss and accuracy 
# need prediction and ground thruth
# NOTE: does not seem used
###########################

def loss_and_all(test_ds, model, loss):

    for e,_ in test_ds:
        batch_size = e.shape[0]
        break

    # get ALL data from a dataset in one go, to avoid managing append in array etc ..
    #for e,l in train_ds.batch(8, drop_remainder=True):  # if drop_remainder False , fails as last is not same dim
    # does not work. add batch dimTensorShape([8, 16, 34, 1]
    #for e,l in train_ds.rebatch(100): #rebatch(N) is functionally equivalent to unbatch().batch(N), but is more efficient
    
    for e,l in test_ds.unbatch().batch(len(test_ds)*batch_size): #rebatch() does not exist for take dataset
        pred=model(e) # TensorShape([492, 34, 1]

    test_loss_0 = loss(pred, l).numpy() # loss returns tensor
    print('loss:' , test_loss_0) # loss before training: tf.Tensor(13.446404, shape=(), dtype=float32

    m = tf.keras.metrics.Accuracy()
    m.reset_state()
    m.update_state(pred,l)
    test_acc_0 = m.result().numpy()
    print('accuracy: ', test_acc_0)

    return(test_loss_0, test_acc_0)

    
###################### 
# GOAL: build confusion matrix using test dataset
# return matrix and truth/prediction
# called by solar2ev -t feature_plot.plot_training_result()
# only for categorical
# created with tf.math.confusion_matrix
######################

# ubuntu does not allow list[int]
def build_confusion_matrix_on_ds(model, test_ds)->Tuple[np.array, list, list]:

    # should only be used for categorical

    assert pabou.is_categorical_ds(test_ds) is True

    # typically used with test_ds
    # predict and build 2 list (1D)
    # tf.math.confusion  input 2x 1D  , return nxn
    # n derived from samples OR numclass  
    # numclass = None assumes ALL classes are present in dataset

    # need to generate 1D list of prediction and ground truth

    # get hot size
    for _,l in test_ds:
        hot_size = l.shape[1]   # TensorShape([16, 4])
        break

    # use .extend() to create flat list (vs list of list with .append())
    # list of int index
    predictions = [] 
    truth = [] # [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, ...]

    for e,l in test_ds:
        pred = model(e)  # .predict is for batches. for small number of input that fit in one batch, use model()
        
        # extend() adds all the elements of an iterable (list, tuple, string etc.) to the end of the list. DO NOT ADD inner list
        # add prediction index at the end
        predictions.extend(np.argmax(pred, axis=1)) # using .append create 2 dim array.
        # default axis = None, argmax on total (flattened) array
        # np.argmax(pred) 63
        # np.argmax(pred, axis=0) array([14,  3,  0,  0], dtype=int64)
        # np.argmax(pred, axis=1 )array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0], dtype=int64)

        # add ground truth index at the end
        truth.extend(np.argmax(l, axis=1))

    assert len(predictions) == len(truth)
    
    # NOT thue, last batch incomplete
    #assert len(predictions) == len(test_ds) * batch_size

    # prediction and labels must be 1-D arrays of the same shape in order for this function to work.
    # num_classes is None, then num_classes will be set to one plus the maximum value in either predictions or labels. 
    # Class labels are expected to start at 0

    #######################
    # create confusion matrix
    ########################

    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    # if num_classes not provided, is calculated
    # 1-D Tensor of real/pred labels for the classification task.
    # return Tensor of type dtype with shape [n, n]
    # specify numclass in case some class just not present (ie very high production in summer samples)
    confusion_matrix = tf.math.confusion_matrix(truth, predictions,num_classes=hot_size, weights=None) # <tf.Tensor: shape=(4, 4), dtype=int32, numpy=
 
    # The matrix columns represent the prediction labels and the rows represent the real labels
    confusion_matrix = confusion_matrix.numpy()

    # in case numclass=None
    assert confusion_matrix.shape[0] == hot_size

    assert len(predictions) == np.sum(confusion_matrix)
    print("total number of samples in confusion matrix", len(predictions))
          
    """
    [[537  54  17   0]
    [ 82 386  55  45]
    [ 28 103 454  39]
    [ 14   6  42 674]]
    """

    return(confusion_matrix, truth, predictions)


############################
# compute precision/recall with sklearn
# returns dict global/macro average, ie unweigthed average of per class (ok as class are balanced)
###########################
def build_precision_recall(truth:list, predictions:list, nb_classes:int) ->dict:

    ########################
    # WARNING: those are not supported by tf as metrics when using MULTICLASS classification (ok for binary) 
    # use sklearn
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    ########################

    print("use sklearn classification report to compute precision/recall")
    d = classification_report(truth, predictions, output_dict=True)

    # print per class
    for i in range(nb_classes): #
        print("class %d:  precision %0.3f, recall %0.3f" %(i, d[str(i)] ["precision"], d[str(i)] ["recall"]))

    print("sklearn macro average:  precision %0.3f, recall %0.3f " %(d["macro avg"] ["precision"], d["macro avg"] ["recall"]))

    print("sklearn accuracy %0.3f (should be the same as from tf metrics)" %d["accuracy"])

    # return values of interest, ie global/macro average

    p_r_dict= {
        "precision" : round(d["macro avg"] ["precision"],3),
        "recall" : round(d["macro avg"] ["recall"],3)
    }

    # p_r_dict.keys() ['0', '1', '2', '3', 'accuracy', 'macro avg', 'weighted avg'])
    # 0': {'precision': 0.8837209302325582, 'recall': 0.75, 'f1-score': 0.8113879003558719, 'support': 608.0},
    # 'accuracy': 0.6762618296529969,
    # macro average (averaging the unweighted mean per label), 
    # weighted average (averaging the support-weighted mean per label)

    return(p_r_dict)



##############################
# GOAL: build confusion matrix for 3 baseline models
# returns list of confusion matrix and updated evaluate dataframe
##############################

def baseline_confusion_matrix(ds, evaluate_df, majority_y, metrics, hot_size, histo_prod):

    # build baseline models with NO learn knowledge
    # use same loss and metrics

    # build baselines with ENTIERE ds, and evalute on entire ds (no training)
    # update evaluate dataframe

    # majority_y is used to create baseline model ALWAYS returning the same onehot prediction
    # histo_pro is used to create baseline random based on histograms

    print('creating 3 baseline models (ZeroZ, random, random weigthed)')

    loss=tf.keras.losses.CategoricalCrossentropy() # 'categorical_crossentropy'
    models_dir = pabou.models_dir


    # BASELINE 0: return constant = majority class
    baseline_0 = create_baseline_models.create_baseline_ZeroR(ds, majority_y, loss, metrics)
    tf.keras.utils.plot_model(baseline_0, to_file=os.path.join(models_dir,'baseline_0.png'), show_shapes=True)
    print(baseline_0.name)

    # BASELINE 1: return random class , param: number of classes
    baseline_1 = create_baseline_models.create_baseline_random(ds, hot_size, loss, metrics)
    print(baseline_1.name)
 
    # BASELINE 2: return random class weigthed on histogram
    baseline_2 = create_baseline_models.create_baseline_random_w(ds, histo_prod, loss, metrics)
    print(baseline_2.name)

    confusion_matrix_baseline = [] 

    columns=[m.name for m in metrics] # metrics, returned by .evaluate() , ie no loss, no val_

    for m in [baseline_0, baseline_1, baseline_2]: # keras models

        # build confusion matrix
        ma, truth, predictions = build_confusion_matrix_on_ds(m, ds)
        confusion_matrix_baseline.append(ma)
        
        ### evaluate on entiere ds. model was not trained anyway
        r  = m.evaluate(ds, return_dict=True, verbose=0) # return loss and all metrics as either a dict or a list
        r  = m.evaluate(ds, return_dict=True, verbose=0) # return loss and all metrics as either a dict or a list

        ####### WTF
        # two consecutive run FOR BASELINE 0 do not give the same result
        # {'loss': 11.815377235412598, 'categorical accuracy': 0.3700787425041199, 'precision': 0.35073742270469666, 'recall': 0.23273341357707977, 'prc': 0.2936703562736511}
        # {'loss': 11.815373420715332, 'categorical accuracy': 0.26695001125335693, 'precision': 0.26695001125335693, 'recall': 0.26695001125335693, 'prc': 0.25803861021995544}
       
        # update evaluate dataframe with model name and metrics
        m_test = [round(r[m],2) for m in columns ]  # metrics returned by evaluate (no loss, no val_)

        row = [m.name, "test"] # start with model name
        row = row + m_test
        
        evaluate_df.loc[len(evaluate_df.index)] = row
   
    return(confusion_matrix_baseline, evaluate_df) # list

 
#########################
# save a given metric dict to json
# update existing, or create new one
# update only metrics present in parameters, so can do partial updates
# read later to update GUI


# updated in solar2ev after train/evaluate (mae, rmse, accuracy) and after computing (precision, recall)
########################
def metric_to_json(categorical:bool, metrics: dict):


    # model_type is a shared variable (Enum)
    # see shared. use enum defined in central place, vs hardcoding "cat" all over the place
    #class model_type(Enum): # better than using str
    #cat = auto()
    #cat = "cat"  # used as key when storing metrics to json
    #reg = "reg"

    if categorical:
        k = model_type.cat.value  # str "cat"
    else:
        k = model_type.reg.value

    if os.path.exists(config_model.metrics_json):

        with open(config_model.metrics_json) as f:
            current_metrics_dict = json.load(f)

    else:
        #   # create new 
        s = "create new metrics json file: %s" %config_model.metrics_json
        print(s)
        logging.info(s)

        # empty dict

        current_metrics_dict = {
            model_type.cat.value : {},
            model_type.reg.value : {}
        }


    #####################
    # update metrics
    #####################

    # update metrics all metrics at once
    # do not do this anymore. metrics update can come from various place (accuracy and precision/recall)
    #metrics_dict[k] = metrict 

    # update only metrics present in dict passed as parameters 
    # iterate on all metrics present in dict parameters (eg mae/mse/rmse, or accuracy , or recall/precision)
    for i in metrics.keys():
        current_metrics_dict[k][i] = round(metrics[i],3)

    s = "update with: %s. updated: %s" %(metrics, current_metrics_dict)
    print(s)
    logging.info(s)


    ########################
    # save dict to json file
    ########################

    j = json.dumps(current_metrics_dict, indent = 4)
    with open(config_model.metrics_json, "w") as f:
        f.write(j)
        #or json.dump(post_mortem_dict, f)


#####################################
# plot confusion matrix(categorical) or error histograms (regression)
# plot training history
# plot loss
# compute precision/recall and save to metrics json
# called by solar2ev -t after training
####################################

def plot_examine_training_results(model, test_ds,history, model_param_dict)->dict:

    print("plotting confusion matrix/non absolute error histograms, precision/recall and training history")

    categorical = model_param_dict["categorical"]
    prod_bins_labels_str = model_param_dict["prod_bins_labels_str"]
    hot_size = model_param_dict["hot_size"]

    (watch, metrics, loss) = model_solar.get_metrics(categorical)

    ############################################################
    # confusion matrix, recall/precision or error distribution
    ############################################################

    if categorical:
        # build and plot confusion matrix for test set
        # columns represent the prediction labels and the rows represent the real labels

        confusion_matrix, truth, predictions = build_confusion_matrix_on_ds(model, test_ds)
        # The matrix columns represent the prediction labels and the rows represent the real labels

        """
        [[537  54  17   0]
        [ 82 386  55  45]
        [ 28 103 454  39]
        [ 14   6  42 674]]

        """
        ###############################
        # analyze confusion matrix by hand
        ###############################

        # look at extremes (make sense if classes are ordered and sorted)
        # also look at cases where reality is < prediction , ie sum all cases where reality is = or > predictions 

        # NOTE: I ended up re-computing precision, recall by hand
        a = confusion_matrix[0] [-1]/len(predictions)
        b = confusion_matrix[-1] [0]/len(predictions)

        print("confusion matrix:\n", confusion_matrix) 
        print("extremes: %0.3f%%, %0.3f%%" %(a, b))

        ###### BIG, BIG warning
        # sum(confusion_matrix) returns an array array([496, 635, 919, 486])
        # use np.sum (without axis)
        nb_samples = np.sum(confusion_matrix)

        nb_classes = confusion_matrix.shape[0]
        
        # The matrix columns represent the prediction labels and the rows represent the real labels
        # look at over, under optimistic

        all_under = 0
        all_over = 0
        all_correct = 0

        rec = 0 # to compute global average , ie average of precision% / nb classes

        ##################
        # go thru columns, ie thru predictions, ie precision
        ##################

        for nb in range(nb_classes):
            col = confusion_matrix[:,nb] # get one column

            col_size = sum(col) # same as np.sum BECAUSE one dim

            under = col[:nb]
            over = col[nb+1:]
            under = sum(under)
            over = sum(over)
            correct = col[nb]
            assert under + over + correct == col_size

            all_under = all_under + under
            all_over = all_over + over
            all_correct = all_correct + correct

            rec = rec + correct/col_size

            # correct for one class is actually precision per class
            # also compute OK or higher. for solar charger, the worst case is I do not charge overnite where I should have 
            print("Columns, ie precision: class %d: under %d %0.3f%%, over %d %0.3f%%, correct(precision per class) %d %0.3f%%" %(nb, under, 100*under/col_size,  over , 100*over/col_size, correct, 100*correct/col_size))
            print("correct or higher %0.3f%%" %((correct+over)/col_size))

        assert all_correct + all_over + all_under == nb_samples

        print("Columns, ie precision: under %d %0.3f%%, over %d %0.3f%%, correct(same as accuracy) %d %0.3f%%" %(all_under, 100*all_under/nb_samples, all_over, 100*all_over/nb_samples, all_correct, 100*all_correct/nb_samples))
        print("correct or higher %0.3f%%" %(100*(all_correct+all_over)/nb_samples))

        print ("average of precision / nb classes (ie macro average) %0.3f" %(rec/nb_classes)) # is the same as sklearn macro

        ##################
        # go thru rows, ie thru real, ie recall
        ##################

        all_under = 0
        all_over = 0
        all_correct = 0

        rec = 0 # to compute global average , ie average of recall% / nb classes

        for nb in range(nb_classes):
            col = confusion_matrix[nb,:] # get one row   ie [nb]

            col_size = sum(col) # same as np.sum BECAUSE one dim

            under = col[:nb]
            over = col[nb+1:]
            under = sum(under)
            over = sum(over)
            correct = col[nb]
            assert under + over + correct == col_size

            all_under = all_under + under
            all_over = all_over + over
            all_correct = all_correct + correct

            rec = rec + correct/col_size

            # correct for one class is actually precision
            print("Rows, ie recall: class %d: under %d %0.3f%%, over %d %0.3f%%, correct(recall per class) %d %0.3f%%" %(nb, under, 100*under/col_size,  over , 100*over/col_size, correct, 100*correct/col_size))
            print("correct or higher %0.3f%%" %((correct+over)/col_size))
        
        assert all_correct + all_over + all_under == nb_samples

        print("Rows, ie recall: under %d %0.3f%%, over %d %0.3f%%, correct(same as accuracy) %d %0.3f%%" %(all_under, 100*all_under/nb_samples, all_over, 100*all_over/nb_samples, all_correct, 100*all_correct/nb_samples))
        print("correct or higher %0.3f%%" %(100*(all_correct+all_over)/nb_samples))

        print ("average of recall / nb classes (ie macro average) %0.3f" %(rec/nb_classes))  # is the same as sklearn macro

        various_plots.plot_confusion_matrix_sns_heatmap(confusion_matrix, prod_bins_labels_str, title='confusion matrix for test set')


        #####################
        # correct or higher
        #####################


        #############################
        # compute precision / recall and update json
        #############################
        # p_r_dict.keys() ['0', '1', '2', '3', 'accuracy', 'macro avg', 'weighted avg'])
        # 0': {'precision': 0.8837209302325582, 'recall': 0.75, 'f1-score': 0.8113879003558719, 'support': 608.0},
        # 'accuracy': 0.6762618296529969,
        # macro average (averaging the unweighted mean per label), 
        # weighted average (averaging the support-weighted mean per label)

        # returned dict is what will go in json
        p_r_dict = build_precision_recall(truth, predictions, hot_size)

    else:

        ######################################
        # plot non absolute error distribution (histograms) 
        # NOTE: inference.predict_on_ds_with_labels() also plot regression error distribution (on any ds w/ labels)
        ######################################
        pred = model.predict(test_ds, verbose=0)
        # (816, 1)  <class 'numpy.ndarray'>  batch dim

        # if test_ds not unbatched, list is nested
        # len(labels) 51 
        # labels[0] <tf.Tensor: shape=(16,), dtype=float64, numpy= array([ 7.62,  7.62,  7.62, .. len(labels[0]) 16

        labels = test_ds.map(lambda i,l: l )
        labels = list(iter(labels.unbatch()))
        assert len(labels) == len(pred)

        labels = np.array(labels)
        # (816,)

        pred = np.squeeze(pred, axis =-1) # can only remove axis  of dimensions of length 1 , last axis by default
        assert pred.shape == labels.shape

        error = pred - labels

        # non absolute
        print("plot error distribution")
        various_plots.single_histogram(error, 25, title='training errors distribution for test set')

        p_r_dict = {}

    ##########################
    # plot training history
    ##########################
    # use panda plot:
    # dict contains val and train, incl loss and lr
    # only retains validation (exclusing loss), for better readability

    #  dict_keys(['loss', 'categorical accuracy', 'precision', 'recall', 'prc', 'val_loss', 'val_categorical accuracy', 'val_precision', 'val_recall', 'val_prc', 'lr']

    # plot all validation metrics (exludes loss)
    # creates a subset of the history dict
    # create dataframe from this dict
    d = {k:v for k, v in history.history.items() if k.split("_")[0] == "val" and k.split("_")[1] != "loss"}

    print("\nplot training history:")

    if categorical:
        title = "training history (classification)"
        ylabel = "classification metrics"
    else:
        title = "training history (regression)"
        ylabel = "regression metrics"

    print("plotting:", d.keys())
    title = "%s %s" %(config_model.app_name, title)
    various_plots.pd_line(pd.DataFrame(d), title = title, xlabel = "epoch", ylabel=ylabel)


    # plot training vs validation for one metric
    if categorical:
        d = {
            "accuracy": history.history["categorical accuracy"],
            "val_accuracy": history.history["val_categorical accuracy"]
            }
        title = "training classification"
        ylabel = "accuracy"
    else:
        d = {
            "mae": history.history["mae"],
            "val_mae": history.history["val_mae"]
            }
        title = "training regression"
        ylabel = "mean absolute error"
        
    print("plotting:", d.keys())
    title = "%s %s" %(config_model.app_name, title)
    various_plots.pd_line(pd.DataFrame(d), title = title,xlabel = "epoch", ylabel=ylabel )


    # plot loss using plt, name etc set in loss()
    various_plots.loss(history.history)


    # plot metric to watch for training, using module in PABOU
    
    #supposed to be defined as str 

    # watch may be defined as val_  and below will add "val_" to create both the training and validation version
    # so remove val_ if exists
    metric_to_plot = watch.replace("val_", "")

    # will plot loss/val_loss and metric/val_metric on ONE figure, TWO subplots

    # same dir 
    plot_dir = various_plots.save_fig_dir
    pabou.see_training_result(history.history, metric_to_plot,  os.path.join(plot_dir,'train_result.png'), fig_title = config_model.app_name )

    return(p_r_dict) 

#################################
# analyze and plot regression error distribution
# quentile, reverse quentile
# absolute and non absolute
# test non absolute gaussian
# called by predict_on_ds_with_labels(..., plot_regression=True). ie end of training 
#################################

def analyze_error_distribution(non_absolute_errors: np.array)->None:

    print("\nanalyzing error distribution. quentile, reverse quentile, gaussian, plot histograms")
    
    absolute_errors = abs(non_absolute_errors) # np.abs()
    #print("absolute errors", absolute_errors)
    
    ###############
    # quentile
    ###############
    # non absolute . mean should be close to 0
    print("error distribution: mean %0.2f, std %0.2f, median %0.2f.  max: %0.2f, min %0.2f" %(np.mean(non_absolute_errors), np.std(non_absolute_errors), np.median(non_absolute_errors), np.max(non_absolute_errors), np.min(non_absolute_errors)))
    
    # absolute 
    print("absolute error distribution: mean %0.2f, std %0.2f, median %0.2f.  max: %0.2f, min %0.2f" %(np.mean(absolute_errors), np.std(absolute_errors), np.median(absolute_errors), np.max(absolute_errors), np.min(absolute_errors)))

    # different quentile non absolute. should be kind of gaussian
    print("error, quantile: ", np.quantile(non_absolute_errors, 0.5), np.quantile(non_absolute_errors, [0.3333, 0.66666]), np.quantile(non_absolute_errors, [0.25,0.5, 0.75]))

    # 1st quantile is the median
    print("absolute error, quantile: ", np.quantile(absolute_errors, 0.5), np.quantile(absolute_errors, [0.3333, 0.66666]), np.quantile(absolute_errors, [0.25,0.5, 0.75]))


    # Kwh value separating above and below % of total samples
    # input %, output Kwh

    # 90% of total errors less than: 13.1777774810791
    # np.quantile(absolute_errors, 0.6) is a number, 1.994470367431640 and 0.6 is a %. ie same as 60% of absolute errors are less than 2
    for x in [0.5, 0.9]: # is a %
        print("%d%% of total errors less than: %0.2f Kwh" %(x*100, np.quantile(absolute_errors, x)))

    ################
    # reverse quentile
    ################
    
    # input Kwh, output % of total samples below this
        
    for threshold in [np.median(absolute_errors), 3.0, 2.0, 1.0]:
        x = sum(absolute_errors < threshold) / float(len(absolute_errors))
        print("%0.1f%% of total errors are less than %0.2fKwh" %(x*100, threshold))


    #################
    # gaussian
    #################
    # std is sqr of mean of squared errors
    # gaussian: 68% within 1 std, 95% within 2std

    # could check is error distribution is gaussian https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411
    # test normality on non absolute errors

    stat, p = shapiro(non_absolute_errors)
    if p > 0.05:
        print("error distribution is gaussian")
    else:
        print("error distribution IS NOT gaussian")


    #################
    # plot histograms
    # both absolute and non absolute
    #################
        
    suptitle = "error distribution"
    # facecolor = inside of fig
    # blue facecolor for figure is not very readable
    figure= plt.figure(figsize=(5,5), facecolor="#42f5e6", edgecolor = "red" ) # figsize = size of box/window
    plt.suptitle(suptitle) # on very top
    plt.ylabel('count')
    plt.xlabel('error kwh')

    # facecolor = bar color
    # edgecolor = bar's edge color
    nb_bins = 30
    (bin_values, bin_edges, _) =plt.hist(non_absolute_errors, bins=nb_bins, facecolor='yellow', alpha=0.9, edgecolor="black", color="green")
    # The values of the histogram bins
    # The edges of the bins.
    #print("values of histograms bins" , bin_values)  # the histograms, ie nb of sample
    #print("bins edge" , bin_edges)

    assert len(bin_values) == nb_bins
    assert len(bin_edges)  == nb_bins +1

    bin_size = abs(bin_edges[0] - bin_edges[1])

    #assertAlmostEqual(first, second, places=7, msg=None, delta=None) from unittest
    assert round(bin_size - abs(bin_edges[-1] - bin_edges[-2]),2) == 0

    print("bin size" , bin_size)

    plt.title('(bin size %0.1fkwh)' %bin_size) # between suptitle and graph


    plt.savefig(os.path.join(save_fig_dir, suptitle))
    plt.close()


    suptitle = "absolute error distribution"
    figure= plt.figure(figsize=(5,5), facecolor="#42f5e6", edgecolor = "red" ) # figsize = size of box/window
    plt.suptitle(suptitle) # on very top
    plt.ylabel('count')
    plt.xlabel('absolute error kwh')

    (bin_values, bin_edges, _) =plt.hist(absolute_errors, bins=nb_bins, facecolor='yellow', alpha=0.9, edgecolor="black", color="green")
    
    #print("values of histograms bins (abs)" , bin_values)
    #print("bins edge (abs)" , bin_edges)

    assert len(bin_values) == nb_bins
    assert len(bin_edges)  == nb_bins +1

    bin_size = abs(bin_edges[0] - bin_edges[1])

    assert round(bin_size - abs(bin_edges[-1] - bin_edges[-2]),2) == 0

    print("bin size" , bin_size)

    plt.title('(bin size %0.1fkwh)' %bin_size) # between suptitle and graph

    plt.savefig(os.path.join(save_fig_dir, suptitle))
    plt.close()