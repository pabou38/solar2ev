#!/usr/bin/env python3

import tensorflow as tf
import keras_tuner as kt

import sys

# https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f


p = "../PABOU"
sys.path.insert(1, p) # this is in above working dir 
try: 
    import pabou
except Exception as e:
    print('%s: cannot import modules in %s. check if it runs standalone. syntax error will fail the import'%(__name__, p))
    exit(1)

p = "../my_modules"
sys.path.insert(1, p) # this is in above working dir 
try: 
    from my_decorators import dec_elapse
except Exception as e:
    print('%s: cannot import modules in %s. check if it runs standalone. syntax error will fail the import'%(__name__, p))
    exit(1)

import model_solar

# https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=en

#######################
# GOAL: KERAS autotune
# hyper model (space) and tune (does training)
#######################

ds = None
cat = None

"""
the model builing function (hyper_model) defined in tuner is called with only one parameter, hp
the build_lstm_model required train_ds as a mandatory parameter. needed for input shape and norm.adapt()
but cannot pass train_ds from hyper_model (eg hyper_model(hp, train_ds))
so define a global variable, available when build_lstm is invoqued
this variable is set in tune(), which now resides in the same module as build_lstm
build_lstm can either get train_ds from parameter (in case invoqued from train) , or from this global (in case invoqued from tuner)
train_ds is now a keyword variable (optional)
"""

ds = None
catego = None

nb_run = -1 # will run once before starting

#######################
# GOAL: define search space and reuse existing build model
#######################
def build_model(hp):
    
    #nonlocal nb_run
    # no binding for nonlocal 'nb_run' found
    # If you use nonlocal, that means that Python will, at the start of the function, look for a variable with the same name from one scope above (and beyond).
    global nb_run

    nb_run = nb_run + 1
    
    # The Hyperparameters class is used to specify a set of hyperparameters and their values
    # called by keras_tuner.search multiple times, with one instance of hp. 
    # called when the tuner class is instantiated

    # returns a COMPILED model with a given set of hyper parameters

    # the way the model building/compile is written, it needs ds and categorical, 
    # but there is no way to pass this as argument. use gobal instead 

    global ds
    global catego # not a search parameter 

    ##################################################
    # KERAS TUNER SEARCH SPACE
    #  unit 3
    #  layer 2
    #  dense 2
    #  dropout 2

    # 24 runs
    ##################################################

    units = hp.Int("nb units", min_value=128, max_value=384, step=128)

    num_layers = hp.Int("nb layers", min_value=1, max_value=2, step=1)

    nb_dense = hp.Int("nb dense", min_value=0, max_value=128, step=128)

    #dropout = hp.Boolean("dropout")
    #dropout_value = hp.Float("dropout_value", min_value=0.2, max_value=0.4, sampling="log")
    dropout_value = hp.Choice("dropout_value", values = [0.0, 0.3])

    #use_attention = hp.Boolean("attention")
    use_attention = False
 
    #learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    #lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    #activation = hp.Choice("activation", ["relu", "tanh"])

    # loss left to default, or categorical cross entropy
    # objective part of tuner class definition

    print("\nbuild hypermodel #%d: units %d, layers %d, dropout %s, dense %d" %(nb_run, units, num_layers, dropout_value, nb_dense))

    model = model_solar.build_lstm_model(ds, catego, units=units, dropout_value=dropout_value, num_layers=num_layers, nb_dense=nb_dense, use_attention=use_attention)
    
    return(model)


# https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=en

@dec_elapse
def tune(train_ds, val_ds, categorical, kt_type="random"):

    # so that available in hyper_model(hp)
    # needed for lstm build (infer input structure, hot size, multi head from dataset)
    global ds
    global catego

    ds = train_ds
    catego = categorical
    
    # trying to use Fixed as an alternative to global to pass categorical and train_ds to hyper_model ?????
    # TO DO: investigate tune_new_entries, allow_new_entries:

    test_hp = kt.HyperParameters()
    s = "fixed hyper"
    test_hp.Fixed("fixed", s)
    assert s == test_hp.get("fixed")
 

    # Could not infer optimization direction ("min" or "max") for unknown metric "val_categorical accuracy". 
    # Please specify the objective  asa `keras_tuner.Objective`, 
    # for example `keras_tuner.Objective("val_categorical accuracy", direction="min")`.
    # The value should be "min" or "max" indicating whether the objective value should be minimized or maximized.

    (watch, met, loss) = model_solar.get_metrics(categorical)

    if categorical:
      print("keras tuner: categorical, MAXimize %s" %watch)
      objective = kt.Objective(watch, direction="max") # for accuracy
    else:
      print("keras tuner: regression, MINimize %s" %watch)
      objective = kt.Objective(watch, direction="min") # for error

    epochs = 80 # for hyperband


    # at search time hyper_model (ie model building function) is called with different hyperparameter combinations
    # During the search, the model-building function is called with different hyperparameter values in different trial.
 
    # After defining the search space, we need to select a tuner class to run the search. 
    # The Keras Tuner has four tuners available - RandomSearch, Hyperband, BayesianOptimization, and Sklearn
    # call hypermodel, with one argument = hp

    # https://keras.io/api/keras_tuner/tuners/random/
    # https://keras.io/api/keras_tuner/tuners/hyperband/

    """
    RANDOM:
    Random search may pick some values which are very obviously bad and will do full training and evaluation on it, 
    which is wasteful. Hyperband provides one way to solve this problem.

    HYPERBAND:
    Solution: Randomly sample all the combinations of hyperparameter and now instead of running full training 
    and evaluation on it, train the model for few epochs (less than max_epochs) with these combinations and select the best 
    candidates based on the results on these few epochs. It does this iteratively and finally runs full training and evaluation 
    on the final chosen candidates. The number of iterations done depends on parameter ‘hyperband_iterations’ and 
    number of epochs in each iteration are less than ‘max_epochs’.
    The Hyperband tuning algorithm uses adaptive resource allocation and early-stopping to quickly converge on a 
    high-performing model. This is done using a sports championship style bracket. 
    The algorithm trains a large number of models for a few epochs and carries forward only the top-performing 
    half of models to the next round. Hyperband determines the number of models to train in a bracket by computing 1 + logfactor(max_epochs) and rounding it up to the nearest integer.
    """

  
    if kt_type == "random":
      print("keras tuner: using random search")
      # hyperparameters = test_hp,

      # creating this class runs hyper_model:
      #  returning hp model: units 64, layers 1, dropout False, dense 0, attention False

      # max trial = nb of models tested, ie nb of training. could or could not exhaust search space

      # trials saved in tuner/solar2ev
      #    sub dir trial_00x, includes checkpoints. and trial.json
      #    as many sub dir as number of trial (ie seach space size)  ?? not clear

      tuner = kt.RandomSearch(
          hypermodel=build_model,
          max_trials=50,
          objective=objective,
          seed = None,
          allow_new_entries = True,
          tune_new_entries = True,
          executions_per_trial=1,
          overwrite=True,
          directory="tuner",
          project_name="solar2ev",
      )

    if kt_type == "hyperband":
      print("keras tuner: using hyperband")

      tuner = kt.Hyperband(
          hypermodel=build_model,
          executions_per_trial=1,
          objective=objective,
          max_epochs=epochs,
          overwrite=True,
          directory='tuner',
          project_name='solar2ev')
      
    if kt_type == "bayesian":
      print("keras tuner: using bayesian")

      tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=objective,
        max_trials=24,
        tune_new_entries=True,
        allow_new_entries=True,
        project_name='solar2ev')

    # print summary of search space
    #print("tuner, search space:\n", tuner.search_space_summary())

    # same callback as for training. to be able to compare results

    callbacks_list = pabou.get_callback_list(early_stop = watch, reduce_lr = watch, tensorboard_log_dir = './tensorboard_dir')
    callbacks_list.append(model_solar.CustomCallback_show_val_metrics()) # ie end epochs

    print("\ntuner: START THE SEARCH, ie call .fit(). watching: %s. callback %s\n" %(watch, [x.__class__ for x in callbacks_list]))

    # All the arguments passed to search is passed to model.fit() in each execution. 
    # metrics is used in .compile, so set as global , and available in build_lstm
    # detailed logs, checkpoints, etc, in the folder directory/project_name.

    tuner.search(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks_list, verbose=0)

    ######################
    # can also subclass the HyperModel class to tune the model building and training process respectively.
    # override HyperModel.build() and HyperModel.fit() (ie add shuffle as hyper) 
    # Tune model training https://keras.io/guides/keras_tuner/getting_started/
    #####################

    class MyHyperModel(kt.HyperModel):
       def build(self, hp):
          image_size = hp.Int("image_size", 10, 28)
          model = None
          return(model)
       
       def fit(self, hp, model, ds, *args, **kwargs):
          
          image_size = hp.get("image_size")

          if hp.Boolean("normalize"):
             pass
             # can access dataset if passed as param
          return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs)
       
    #tuner1 = kt.RandomSearch( hypermodel=MyHyperModel(),
    #tuner1.search(param to fit)
    #####################
          
  

    n = 3
    print("\n########\ntuner: SEARCH COMPLETED using %s\n########\nshow best %d results:" %(kt_type, n))

    # summary of the search results including the hyperparameter values and evaluation results for each trial.
    # print, default best 10
    tuner.results_summary(n)


    #######################################
    # best trained model, as determined by the tuner's objective.
    ######################################
    #The model is saved at its best performing epoch evaluated on the validation_data.
    #The models are loaded with the weights corresponding to their best checkpoint (at the end of the best epoch of best trial).

    print("\ntuner==>: retrieve best model, trained by Keras Tuner during search")
    best_models = tuner.get_best_models(num_models=1) # List of trained model instances sorted from the best to the worst.
    best_kt_trained_model = best_models[0]
 
    print("\ntuner==>: best kt trained model:")
    best_kt_trained_model.summary()

    
    ##########################
    # build model to be retrained from scratch
    #########################


    # For best performance, it is recommended to retrain your Model on the full dataset 
    # using the best hyperparameters found during search, which can be obtained using tuner.get_best_hyperparameters().
    # This method can be used to reinstantiate the (untrained) best model found during the search process.

    best_hps = tuner.get_best_hyperparameters(1) # return list of hp, from best to wors

    print("\ntuner==>: build new model with best hyper parameters")
    best_model_to_train = build_model(best_hps[0])
    #best_model_to_train.summary()

    # can then retrain model with full dataset
    #history = model.fit( train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks_list,  verbose = 2, )
    #r  = model.evaluate(test_ds, return_dict=True) # return loss and all metrics as either a dict or a list
    #print ('metrics: ', model.metrics_names) # ['loss', 'mean_absolute_error', 'accuracy']
    #print('evaluate keras tuner', r)
    
    # return best model to be retrained and kt trained best model
    return(best_model_to_train, best_kt_trained_model)

