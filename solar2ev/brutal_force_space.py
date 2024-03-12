#!/usr/bin/env python3






#################
# input feature
# tuple of what would be configured in config_features, ie [()] or [(), (), ()]
#################

"""
[('temp', 'pressure', 'production', 'sin_month', 'cos_month')],
    [('temp', 'pressure', 'production', 'sin_month', 'cos_month', 'humid')],
    [('temp', 'pressure', 'production', 'sin_month', 'cos_month', 'wind')],
    [('temp', 'pressure', 'production', 'sin_month', 'cos_month', 'wind', 'humid')],
    [('temp', 'pressure', 'production', 'sin_month', 'cos_month', 'wind', 'direction_sin', 'direction_cos')],
"""

feature_list_h = (
  
    [('temp', 'pressure', 'production', 'sin_month', 'cos_month')],
)


#################
# output feature
#################
categorical_h = [True]

#"prod_bins":[0, 11.06 , 22.49, 1000000],
#"prod_bins":[0, 6.3, 13.7, 20.6, 26.1, 1000000],
# [0, 7.83, 17.37, 25.14, 1000000],

prod_bins_h = [
    [0, 7.83, 17.37, 25.14, 1000000],
    [0, 11.06 , 22.49, 1000000], 
    [0, 6.3, 13.7, 20.6, 26.1, 1000000]
    ]

########################
# sequences
########################

# days_in_seq and sampling drives the sequence length
days_in_seq_h = [4]
sampling_h = [1]

# seq_build_method and stride or selection drives the number of sequences

# combine seq_build_method and either stride or selection
# tuple of seq_build_method , either selection (build 3) or stride (build 1)
# selection: starting hours

# (1,24), (1,2), (1,1)  
# (3,[17,18,19,20,21,22,23,0,1,2,3,4,5,6,7]),
seq_build_h=[
   (3,[17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,8]),
   (3,[0,1,2,3,4,5,6,7,8]),
   (1,1),
   (1,2)
    ]

###################
# other
###################

shuffle_h = [True]
nb_run = 1 # to repeat SAME run twice

###########################
# model hyper parameters
###########################

use_attention_h = [False]

# those can be fixed, and use keras tuner to search hyper parameters
nb_unit_h = [256]
nb_lstm_layer_h = [2]
nb_dense_h = [64]


space_size = nb_run * len(nb_dense_h) * len(nb_lstm_layer_h) * len(nb_unit_h) * len(shuffle_h)
space_size = space_size * len(seq_build_h) * len(sampling_h) * len(days_in_seq_h)
space_size = space_size * len(categorical_h) * len(use_attention_h) * len(prod_bins_h)
space_size = space_size * len(feature_list_h)

#print ("==> brutal search space size (ie nb of training) %d" %space_size)

