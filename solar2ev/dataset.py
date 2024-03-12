
#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd


#################################
# various method to create dataset of sequences,label from X,Y numpy array
#################################


# sampling allow hours subsampling
# stride allow to select start of sequence, every day (stride = 24), every hours (stride = 1) 
# predict will be done end of day n to predict solar day n+1, 
# so better if networks sees mostly sequence (which are fixed len) which start around 0h


def seq_from_array_1(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical) -> tuple:

    ########
    # simplest, but no way to only select sequences which start around midnigth
    # no need to set cardinality
    # stride = 24 returns one sequence per day (starting at 0h)
    # stride = 2, seq start every two hours
    # X and Y aligned, ie production in Y is future (day(X) + n) 
    ########


    """creates td.data dataset from numpy array

    Args:
        X (ndarray): samples
        Y (ndarray)): labels, onehot
        seq_len (int): len of sequence
        sampling (int): interval between timestep
        stride (int): interval between start of sequence
        batch_size (int): for dataset
        selection (list[int], optional): hours to retain . Defaults to None.

    Returns:
        tuple: dataset and number of elements
    """

    print("\nmethod1: create dataset using .timeseries_dataset_from_array() from X: %s " %(str(X.shape)))
    if labels:
        print("Y: ", Y.shape) # if was None, ie labels = false
    else:
        Y=None
        print("Y: ", Y)
       
    # data: Numpy array or eager tensor containing consecutive data points (timesteps). Axis 0 is expected to be the time dimension.
    # target: targets corresponding to timesteps in data. targets[i] should be the target corresponding to the window that starts at index i (see example 2 below). Pass None if you don't have target data (in this case the dataset will only yield the input data).
    ds = tf.keras.utils.timeseries_dataset_from_array(
            data = X,
            targets = Y,
            sequence_length=seq_len,
            sampling_rate=sampling,
            sequence_stride = stride,
            batch_size=batch_size,
            shuffle = False
        )

    # i or (i,l) if Y is there

    ################### THIS IS EXPENSIVE TO IT ONCE
    # compute number of sequences 
    nb_seq = len(list(iter(ds.unbatch().batch(1))))

    return (ds, nb_seq) # 2nd parameter is nb of seq (samples) if known.  
    



def seq_from_array_2(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical):

    ########
    # use tf data .window()
    # access to ds of individual sequences, on which some processing can be done
    # to select sequences of interest, eg sequences starting "around" 0h
    # uses selection
    ########

    print("create sequence and label dataset using .window() from X %s and Y %s" %(X.shape, Y.shape))

    assert stride == 1, "stride must be 1 for this method"

    ###### retain only the sequence corresponding to early hours
    print(selection)

    # convert X and Y to dataset, 
    # .window() creates dataset of datse
    # flat_map convert to dataset of tensors 
    #     - use batch to get dataset of tensor sequences (FLATTED has unknown cardinality)
    # define generator which iterate on all sequences, select some, and yield some for X, and label for Y
    # create x and y dataset from this generator
    # zip

    feature_dim = X.shape[1]  # nb of features
    hot_size = Y.shape[1]

    ############
    # sequences
    ############
    
    ds_x = tf.data.Dataset.from_tensor_slices(X) 
    # Creates a Dataset whose elements are slices of the given tensors.
    # removing the first dimension of each tensor and using it as the dataset dimension
    # All input tensors must have the same size in their first dimensions
    # X (14880, 2) =>  ds.element_spec TensorSpec(shape=(2,), dtype=tf.float64, name=None)
    e = iter(ds_x).next() #  TensorShape([2])


    # shift: period between start, ie number of input elements to shift between the start of each window
    # stride: stride between input elements within a window.
    # True to only get complete sequences (fixed len)

    windows_ds = ds_x.window(seq_len, shift = stride, stride = sampling, drop_remainder=True) 
    # Returns a dataset of "windows". ie of seq dataset

    for seq_ds in windows_ds: # seq_ds is a dataset  yielding each element in one sequence
        e = iter(seq_ds).next() # temp, pressure
        print("windows ds len, ie nb of seq: %d." %(len(windows_ds)))
        print("element of above (ie seq): len: %d. element : shape %s value %s" %(len(seq_ds), e.shape, e))

        # windows ds len, ie nb of seq: 14813.
        #element of above (ie seq): len: 68. element : shape (2,) value tf.Tensor([   1.3 1023.9], shape=(2,), dtype=float64)
        break

    # convert dataset (sequence) to array np.asarry(ist(iter)), create list of array, converted to np.asarray 
    # at the end is SLOWWW (20sec)

    #ds = windows_ds.flat_map(lambda x:x) not used, seems to do nothing
    # function that takes an element from the dataset and returns a Dataset. flat_map chains together the resulting datasets sequentially.
    # elements in windows_ds are datasets. each of thoose is another dataset, whose element are individual timestep
    # flap_map convert those elements into a single batch , ie make a TENSOR

    def f(e):
        return e.batch(seq_len, drop_remainder=True) # batch all timestep of one sequence

    # windows dataset  DatasetSpec(TensorSpec(shape=(2,), dtype=tf.float64, name=None), TensorShape([]))
    flat_ds = windows_ds.flat_map(f)
    # this dataset now returns TENSOR
    # flatten dataset  TensorSpec(shape=(68, 2), dtype=tf.float64, name=None)
    # for some reason, this dataset has unknown cardinality, and I use a lot of assert len(ds) ==   later
    assert flat_ds.cardinality().numpy() == tf.data.UNKNOWN_CARDINALITY

    # set cardinality. is the same as unflatten
    flat_ds = flat_ds.apply(tf.data.experimental.assert_cardinality(len(windows_ds))) 

    ######### create dataset from iterator
    # allows to select sequences to include, ie around midnigth every day
    # WAYYYY faster
    # select from flat_ds to create ds_x (a subset). should have same element_spec
    # len(ds_x) < len(flat_ds) ratio based on what is selected

    # generator select sequences to retain in dataset. assumes no stride, so one sequence per hour
    def gen_x():
        for i,seq in enumerate(flat_ds):
            if i % 24 in selection:
                yield(seq) # tensor  (68, 2)

    # need output signature or output type
    output_signature=( tf.TensorSpec(shape=(seq_len, feature_dim), dtype='float'))

    output_signature=(flat_ds.element_spec)
    # output signature should be consistent with what is yielded by generator

    ds_x = tf.data.Dataset.from_generator(gen_x, output_signature=output_signature)
    # cardinality unknow
    assert ds_x.cardinality().numpy() == tf.data.UNKNOWN_CARDINALITY

    # ds_x is a "subset" of flat_ds
    assert ds_x.element_spec == flat_ds.element_spec

    # compute cardinality by running the generator .. any other way ??
    ################### THIS IS EXPENSIVE TO IT ONCE
    nb_seq = 0 
    for e in gen_x():
        nb_seq = nb_seq + 1 # how many sub samples sequences
   
   # https://www.tensorflow.org/api_docs/python/tf/data/experimental/assert_cardinality
    ds_x = ds_x.apply(tf.data.experimental.assert_cardinality(nb_seq))

    # len() unknown , unless set cardinality manually
    o = flat_ds.cardinality().numpy()
    d = ds_x.cardinality().numpy()
    print("subsampling from %d. retain %d creates %d ratio %0.2f"  %(o, len(selection), d, o/d))
    # with if i % 24 in [22,23,0,1,2,3], ratio should be exactly 4.00

    # 3.9970318402590395
    np.testing.assert_almost_equal(o/d, 24/len(selection), decimal=2, err_msg='', verbose=True)

    e = iter(ds_x).next() # tensor
    assert e.shape[0] == seq_len
    assert e.shape[1] == feature_dim

    x = ds_x.as_numpy_iterator().next() #  numpy
    print("ds_x dataset yield ", type(x), x.shape)

    for e in ds_x.take(1):
        #print(e)
        pass

    #[ 2.9] [ 1.9] [ 1.1]], shape=(68, 1), dtype=float64)

    

    ############
    # labels
    ############

    # same for label, except gen returns hot (vs a seq)
    ##### WTF , reusing windows_ds and flat_ds created a problem when accessing ds_x 
    ##### after creating ds_y
    ##### I guess related to use of generator
    # TypeError: `generator` yielded an element of shape (1, 2) where an element of shape (68, 1) was expected.

    ds_y = tf.data.Dataset.from_tensor_slices(Y) 
    w = ds_y.window(seq_len, shift = stride, stride = sampling, drop_remainder=True)
    for seq_ds in w: # 
        e = iter(seq_ds).next() #
        print("windows ds len, ie nb of seq: %d." %(len(windows_ds)))
        print("element of above (ie seq): len: %d. element : shape %s value %s" %(len(seq_ds), e.shape, e))
        # windows ds len, ie nb of seq: 14813.
        # element of above (ie seq): len: 68. element : shape (2,) value tf.Tensor([1. 0.], shape=(2,), dtype=float32)
        break


    def f1(e):
        return e.batch(seq_len, drop_remainder=True) # batch all timestep of one sequence
    flat_ds_y = w.flat_map(f1) 
    
    # .batch(1) TensorSpec(shape=(1, 2), dtype=tf.float32, name=None)
    # batch(seqlen) TensorSpec(shape=(68, 2), dtype=tf.float32, name=None

    assert flat_ds_y.cardinality().numpy() == tf.data.UNKNOWN_CARDINALITY
    flat_ds_y = flat_ds_y.apply(tf.data.experimental.assert_cardinality(len(windows_ds)))

    def gen_y():
        for i,seq in enumerate(flat_ds_y):
            if i % 24 in selection:
                yield(seq[0]) # 

    nb_seq_1 = 0 
    for e in gen_y():
        nb_seq_1 = nb_seq_1 + 1 # how many sub samples sequences
        print(e) # tf.Tensor([1. 0.], shape=(2,), dtype=float32)
        break        #
        

    #output_signature=flat_ds.element_spec # (68,2)
    output_signature=tf.TensorSpec(shape=(hot_size, ), dtype='float')

    ds_y = tf.data.Dataset.from_generator(gen_y, output_signature=output_signature)

    assert ds_y.cardinality().numpy() == tf.data.UNKNOWN_CARDINALITY
    ds_y = ds_y.apply(tf.data.experimental.assert_cardinality(nb_seq))

    e = iter(ds_y).next() # tensor
    assert e.shape[0] == hot_size

    y = ds_y.as_numpy_iterator().next()
    print("ds_y dataset yield ", type(y), y.shape)

    for e in ds_y.take(1):
        print(e)

    for e in ds_x.take(1): # check 
        #print(e)
        pass

    ############
    # combine sequence and label
    ############

    ds = tf.data.Dataset.zip((ds_x, ds_y))
    assert ds.cardinality().numpy() == ds_x.cardinality().numpy()
    assert ds.cardinality().numpy() == nb_seq

    for e,l in ds.take(1): 
        pass

    ds = ds.batch(batch_size)

    i,l = ds.as_numpy_iterator().next()
    print("ds (i,l) dataset yield ", type(i), i.shape, type(l), l.shape)
    # ds (i,l) dataset yield  <class 'numpy.ndarray'> (16, 68, 1) <class 'numpy.ndarray'> (16, 2)

    return(ds, nb_seq)



def seq_from_array_3(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical):

    ########
    # manual, 
    # has to fit in memory, but not a problem
    # assumes stride 1
    # use selection

    # called several time if multi head

    # labels = False for inference (daily/postortem)
    ########

    print("\nmethod 3: create dataset from numpy X (ie for one head): %s" %(str(X.shape)))  # if not str, decoded as tuple

    # one head
    assert len(X.shape) == 2

    # stride ignored, use selection  instead
    #assert stride == 1 , "stride must be 1 for manual sequence building"

    ###### retain only the sequence corresponding to selected (eg early) hours
    print("method 3: selection ",selection)

    if labels:
        print("with labels", Y.shape)
        # shape(33333,2) or (33333,)

    else:
        print("without labels, use fake labels 1 to satisfy processing", Y)
        # building dataset for inference 
        # created fake Y to satisfy processing. will not be used in dataset 
        Y = [1] * X.shape[0]
        Y = np.asarray(Y)


    ############################
    # x list of numpy array (seq_np)
    #   created from slicing X  i:i+seq_len
    #   ends when slicing does not return what we asked for
    ############################

    x = [] # list of numpy array representing one sequences,  x = np.asarray(x), tf.data.Dataset.from_tensor_slices(x)
    y = [] # list of labels

    for i, hour in enumerate(X): # hour: array of features

        ##################################
        # since Y is shorter than X, check first if we exausted Y (will run out before X)
        ##################################

        try:
            Y[i]
        except Exception as e:
            print("%s. Y is running out at index %d" %(str(e), i))
            assert i == len(Y)
            print("exit loop")
            break

        if i % 24 in selection: # retains hour starts around 0h

            try:
 
                # creates ONE sequence, considering sampling
                # [start:end:step]  default [0:len(array):1]

                seq_np = X[i:i+seq_len*sampling:sampling] # numpy

                # out of bound slicing does not generates exception, just returns what is available
                # whereas out of bound indexing does

                if len(seq_np) == seq_len: # is not the same at the end of X   could also use seq.shape[0]
                
                    assert (seq_np[0] == X[i]).all(), "seq_build 3: error seq[0]" # stride is 1      array([1.0239e+03, 1.3000e+00, 9.9000e-01])
                    assert (seq_np[-1] == X[i+(seq_len-1)*sampling]).all() , "seq build 3: error seq[-1]"

                    # prepare one sequence
                    x.append(seq_np) # x list. list of numpy array (one array is one sequence)

                    # prepare associated label
                    # this is different from using tf.keras.utils.timeseries_dataset_from_array
                    if i>12:
                        #y.append(Y[i+1]) # eg sequence starting at 22h 
                        y.append(Y[i]) # eg sequence starting at 22h 
                    else:
                        y.append(Y[i]) # X and Y are synched  Y[i] is the label for a sequence starting at X[i] numpy.ndarray
                
                else:
                    # end of slicing X, indicates we are done
                    # note should not happen, as we run out of Y before

                    print("seq_build_method 3: row %i. seq_len expected %d, seq_len %d. indicates we are done" %(i, seq_len, len(seq_np)))
                    break

            except Exception as e: 
                # this correspond to a real problem
                print('\n\n!! seq_build_method 3. exception %s %d. exit' %( str(e), i))
                assert False

        else:
            pass # not in selection

    print("%d sequences , %d labels" %(len(x), len(y)))

    assert len(x) == len(y)
    x = np.asarray(x) # (3088, 68, 2) 
    y = np.asarray(y) # (3088, 2)

    # defined from np array
    nb_seq = x.shape[0]

    assert x.shape[0] == y.shape[0], "dataset: sequence %s and labels %s not aligned" %(x.shape, y.shape) # sequence and labels  are aligned
    assert x.shape[1] == seq_len
    assert x.shape[-1] == X.shape[-1] # same number of dim , ie nb of features


    # create ZIPPED dataset of ds_x and ds_y if labels
    # only ds_x if not labels
    ds_x = tf.data.Dataset.from_tensor_slices(x)

    if labels:
        if categorical:
            assert y.shape[-1] == Y.shape[-1] # same number of dim, ie nb of bins

        ds_y = tf.data.Dataset.from_tensor_slices(y) 
        assert ds_y.cardinality().numpy() == ds_x.cardinality().numpy()
        # ZIP ds_x and ds_y
        ds = tf.data.Dataset.zip((ds_x, ds_y))
    else:
        ds= ds_x

    # not batched yet
    # ds has same number as original numpy
    assert len(ds) == nb_seq

    print('dataset created: %s, nb_seq %d' %(ds.element_spec, nb_seq)) # could yield (i,l) or i
    # (TensorSpec(shape=(90, 5), dtype=tf.float64, name=None), TensorSpec(shape=(3,), dtype=tf.float32, name=None))
    
    # various check before returning
    assert ds.cardinality().numpy() == ds_x.cardinality().numpy()
    assert ds.cardinality().numpy() == nb_seq 

    # check dataset elements shape
    element = list(iter(ds))[0] # 1st element of dataset.
    e = iter(ds).next() # tensor

    if labels:
        i = e[0]
        l = e[1]
        assert i.shape[0] == seq_len # TensorShape([68, 2])
        assert i.shape[1] == X.shape[1]
        if categorical:
            assert l.shape[0] == Y.shape[1] # TensorShape([2])

    else:
        i = e
        assert i.shape[0] == seq_len # TensorShape([68, 2])
        assert i.shape[1] == X.shape[1]
       

    # check dataset element content. too complex, depend on selection
    #for i, (seq, lab) in enumerate(ds.take(10)):
       
    # batch dataset
    ds = ds.batch(batch_size)
    print("dataset batched, %d batch, %d sequences" %(len(ds), nb_seq))

    if labels:
        print("iter(ds).next() returns ", iter(ds).next()[0].shape, iter(ds).next()[1].shape )
    else:
        print("iter(ds).next() returns ", iter(ds).next().shape)


    return(ds, nb_seq)



########################################### 
# build sequence tf.data dataset from pandas dataframe

# input: df_model 

# X selected serie , converted to numpy (normalization happens later, in model)
# Y created using pandas.cut, which returns labels, converted to one hot
# shuffle, cache, prefetch
# returns dataset
#############################################

"""
    day-2   day-1   day0    day+1
    meteo   meteo   meteo
    nh      nh      ph
                            solar

for multi head:
    create multiple sequence dataset (one per head)    ds1 (i1,l), ds2 (i2,l) , ...
    zip all ds  ( (i1,l), (i2,l), ..) 
    map on zipped to create ds with input onlt (i1,i2, ..)
    map on zipped to create ds with label only (l)
    zip above two to create final ds ((i1,i2, ..) , l)

"""

def build_dataset_from_df(df_model, \
feature_list, days_in_seq, seq_len, retain, selection, seq_build_method, hot_size, batch_size, prod_bins, prod_bin_labels, 
stride=1, sampling=1, shuffle=False, labels=True, categorical=True):

    #################################################
    # can be used for TRAINING dataset, ie:
    #   need label, ie Y
    #   disgard first Y 
    #   align X and Y 
    #   return dataset with X and Y 

    # also (labels=False) for inference
    #   do not disgard or align
    #   return dataset with X only (fake Y)

    # =====> first create X, Y np array(s), from dataframe. one X per head
    # =====> second call seq_from_array_x(X,Y). returns ds
    #################################################

    # hot size use for creating one hot (force number of classes)

    #assert seq_len == days_in_seq * len(retain)/sampling  #  not true. last days in incomplete for inference

    # for inference, not an even number of days
    nb_days = len(df_model)/len(retain)

    print('\nbuild sequence dataset\n')
    print('features: %s'  %( feature_list))
    print('%d hours. %0.2f days'  %(len(df_model), nb_days))
    print('categorical %s, labels: %s' %(categorical, labels))
    print('sampling %d, days in seq %d. seqlen %d' %(sampling, days_in_seq, seq_len))
    print("build method %d, stride %d, selection %s" %(seq_build_method, stride, selection))
    print("shuffle %s" %shuffle)



    ############################################
    # switch, ie all method to build dataset from NUMPY ARRAY of hours
    # based on varialble seq_build_method
    ############################################
    # called later

    def seq_build_one_head(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical):
        if seq_build_method == 1:
            return seq_from_array_1(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical) # keras.utils
        elif seq_build_method == 2:
            stride=1
            return seq_from_array_2(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical) # ds.window()
        elif seq_build_method == 3:
            stride =1
            return seq_from_array_3(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical) # manual with array
        else:
            return(None)
        

    ###############################
    # first create X(s), Y numpy array from df. one X per head
    # X array of hours
    # Y array of production (same for each days)
    ###############################

    ######### convert df to X np array
    # build X_l list of X numpy array. one array per head. 
    # if single head , list has one element

    print('\nCREATE DATASET: step 1: converting pandas to numpy: 1) X_l, list of X and 2) Y (possible one hot):')


    # build list of X (numpy), one per head
    X_l = [] 

    if len(feature_list) == 1:  
        #######################################################
        # single head model [['temp']] or [['temp', 'humid']]
        #######################################################

        l = [f for f in feature_list[0]]
        pd_features = df_model[l] # select colums  df_model[['temp']] df_model[['temp', 'humid']]
        X = pd_features.to_numpy() #selected serie(s) for input, convert to ndarray
        # [['temp']] (7476, 1) X[0] array([-1.4])
        # [['temp', 'humid']] (7476, 2) X[0] array([-1.4, 96. ])
        X_l = [X] # create a list of only one head

    else: 
        #######################################################
        # multi head 
        #######################################################

        # use list, instead of having everything in same array, as num features for each head can be different
        # could not broadcast input array from shape (7500,2) into shape (7500,)

        X_l = [] # list of head
        # multi heads [['temp'], ['humid']] or [['temp', 'month'], ['humid']]
        for h in feature_list:
            l = [f for f in h] # not needed !!! what was I thinking
            pd_features = df_model[l]  # select columns for each head
            X_l.append(pd_features.to_numpy())
    

    print('list of X(s) (numpy array of hours) used to build dataset, before alignment: ', [X.shape for X in X_l])  # shape[1] is number of columns
    print('in days %0.2f' %(X_l[0].shape[0]/len(retain)))
        

    # output

    # output is from original df_model dataframe, last columns
    # last columns is solar production for that day, ie same value for day's hours

    if labels:

        print("create dataset with labels(ie for training, vs for inference")

        ################################################
        # THINKING ABOUT IT
        # for a long time, I had dropped the last X entries, (ie last day_in_seq days), because they have no FUTURE PRODUCTION associated
        # It is true I cannot start a sequence with those days, but I can still finish one, ie they can be used by sequences starting earlier
        # it does not hurt, but will ignore unecessarly a couple of valid sequences
        #   not a big deal for training, we have plenty of days already , 
        #   an issue for analyzing unseen data:
        #     days_in_seq = 4, 4 unseen, selection = [...., 0,1,2,3,4,5,6,7]
        #     only 7 sequences will be built

        # building sequences with slicing will end because getting to the "end" of X (detected because returned slice do not have the required length) 
        # SO:
        #   still need to ignore the first entries of Y. those a useless as I do not have the previous meteo days (for which y is a prediction)
        #   aligning the resulting Y to X is still needed. at the same index, we have meteo for a day, and solar for day +days_in_seq, ie solar for sequences STARTING at that day
        #   do not cut last X

        #print("ignore last day_in_seq X and first day_in_seq Y, and shift to align")

        print("ignore first day_in_seq Y, and shift Y to align to X")

        # labels required for dataset used for TRAINING, not dataset used for today inference
     

        # for all heads
        for i, X in enumerate(X_l):   # this does not do anything anymore if we do not ignore last entries of X

            # initial nb of days
            days_ = X.shape[0] / len(retain) # number of days's worth of hourly sample  7476 623

            #X = df_model[['temp']].to_numpy()  # ['temp'] shape(7000,)  [[]] shape(7000,1)
            # X should match original data frame 
            
            nb_samples = len(df_model) # nb of hours
            assert nb_samples == X.shape[0] # one sample = one hour's reading (7476, 2)

            #####################################
            # ignore remove last n days of X. not done anymore
            #X = X[:-days_in_seq*len(retain)] # ignore last 96 samples, last 4 days, which have no solar prediction (have same day solar available)
            ##################################### 

            # NOTE -days_in_seq*len(retain), is also used to remove FIRST entries in Y, to align

            # X should be only full days 
            assert X.shape[0] % len(retain) == 0, "number of hours is not full days %d %0.1f" %(X.shape[0], X.shape[0]%len(retain))

            # numbers of days with associated solar production
            days = X.shape[0] / len(retain)
            print('head %d: X input to create sequences correspond to %0.1f days. %d hours' %(i, days, X.shape[0]))
            #assert (days_ - days) == days_in_seq , "X drop last entries error" # 623 to 620 days
            assert (days_ - days) == 0 , "X drop last entries error" # 623 to 620 days

            # do not forget to put back
            X_l[i] = X

        # last X entries dropped
        # X_l created, one or more array, each row is one hour:  X_l[0].shape (17400, 5) X_l[1].shape (17400, 2)

        """
        # normalize input here or in keras preprocessing layers
        # if done here, use all samples for selected colums, ie train , val, test
        X_mean = X.mean()
        X_std = X.std()
        X = (X-X_mean)/X_std
        print('X normalized ', X.mean(), X.std())
        """

        ############ CREATE Y 
        # >>>>>> ALIGN, ie skip first n days to align Y as day+n production for X (ie start of sequence) meteo 
        # this is a requirement for tf.keras.utils.timeseries_dataset_from_array
        #        target: targets corresponding to timesteps in data. targets[i] should be the target corresponding to the window that starts at index i (see example 2 below). Pass None if you don't have target data (in this case the dataset will only yield the input data).
        # for synthetic data, colums order is not enforced, and last colums may not be production. so use ["production"] vs [-1]
        
        # first Y entries dropped
        Y = df_model["production"].iloc[days_in_seq*len(retain):]

        assert Y.isnull().sum() == 0 # nan should be removed  Y is serie. Y.isnull() is serie of bool. isnull().sum() scalar

        ##################################
        assert len(X)/24 == len(Y)/24 + days_in_seq, "len X %d does not match len Y %d" %(len(X), len(Y))
        
        # now X for day n is aligned with Y solar production day n+3
        # WARNING: X is longer than Y. 


        if categorical:

            # convert production float into bin index
            # create bucket for float production value. assumes easier to train than trying to predict scalar (less output values)
            # ie prod between 0 to 5 mapped to bucket index 0
            # index later transformed to one hot

            # this would create a new columns 
            #df_model['bins'] = pd.cut(x=df_model[df_model.columns[-1]], 
            #Y = df_model['bin'].to_numpy()

            print("categorical: convert Y to one hot")

            # use retbin to avoid adding a new colums in df_model
            (serie_indices, bracket)= pd.cut(x=Y, retbins=True, bins=prod_bins, labels = prod_bin_labels, include_lowest=True)
            assert list(bracket) == prod_bins 
            # use -1 as some prod are 0, or play with include_lowest 
            # serie is list of indices 0 to 5 from prod_bina_labels
            assert len(Y) ==  len(serie_indices)

            # if date beyond last bin, series included nan, and to_categorical fails
            assert serie_indices.isnull().sum() == 0, "nan values in bin serie" 

            Y = serie_indices.to_list() #  to_list(), to_numpy() for serie and frame. tolist() seems to work as well
            # keep as list to use later in to_categorical.  or convert to_numpy() and use Y.to_list()   

            assert all(Y) in range(hot_size)

            # convert bin index to one hot
            # tf.keras.utils.to_categorical is for integer labels
            Y = tf.keras.utils.to_categorical(Y, num_classes=hot_size, dtype='float32')
            # force number of classes. some input may contains few sample (for delta), which do not contains all examples of hot size
        
        else:

            print("regression: leave Y as it")
            # Y still a dataframe, convert to numpy
            Y = Y.to_numpy()


        # X and Y have compatible len. 
        assert X.shape[0] == Y.shape[0] + 24*days_in_seq
        assert len(X) == X.shape[0]

        print("Y labels", Y.shape) # Y labels (17400,) for scalar (mse)

    else:

        print("no labels. do not modify X and Y=None")
        Y = None
        print("Y: ", Y) 



    #############
    # X_l (list of X) and Y (numpy array) available

    # at this point one day's of X (12 rows) have day(X)+ 3 solar production
    # ie sequence starting at X will be trained to predict day(X) + 3 solar, day, day+1, day+2 is one seqlen, day+3 is target 

    # create dataset of sequences for RNN 
    # targets[i] should be the target corresponding to the window that starts at index i 

    # X_l list of X, Y ready
    # Y = None for inference
    # creates one dataset (one head)
    #  or creates list of dataset
    #  then merge them into a single dataset
    ###################

    

    ####### 
    # stride : interval between start of sequences
    #       sequence starts at 0h or not
    # sampling rate: interval between individual timesteps
    #       subsampling hours
    #######

    nb_heads = len(X_l) # nb of heads

    print("CREATE DATASET. step 2: create dataset from numpy")
    
    if len(X_l) == 1:

        # single head

        print("\ncreate dataset from SINGLE head X %s, ie %0.2f days" %(X.shape, X.shape[0]/len(retain)))
      
        ds, nb_seq = seq_build_one_head(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical)

        print("single head dataset created. nb_seq: %d, seq len %d" %(nb_seq, seq_len)) 
        # ds final dataset. do not use spec[1], does not exist for inference dataset (no labels)
        # X.shape (7476, 1) len(_ds) 39   39*12*16 = 7488  some batch not complete 

    else:

        # multi head
        print("\ncreate multiple dataset from each head, then merge them into ONE dataset")
        
        # list of dataset (one per head)
        # each dataset created with labels (to reuse existing code). Y should be identical in each
        # combining dataset is done later

        ds_l = [] 

        for X in X_l:

            # create dataset for one head
            print("create dataset from head X %s ie %0.2f days" %(X.shape, X.shape[0]/len(retain)))

            ds_ , nb_seq = seq_build_one_head(X,Y, seq_len, sampling, stride, batch_size, labels, selection, categorical)

            print("one head dataset created: nb_seq: %d, seq len %d" %(nb_seq, seq_len))
            
            ds_l.append(ds_) 
            
        print("dataset corresponding to all %d heads created. now merge them" %len(X_l))

        # ds_l is a list of 2 or more dataset
        assert len(ds_l) >= 2
        assert len(ds_l) == nb_heads
    
        for i in range(len(ds_l)):
            assert len(ds_l[0]) == len(ds_l[i]) # check, all dataset should have same len

        #############
        # create final dataset by merging them all
        #############

        ########
        # STEP 1
        # merge (zip) all datasets 
        # (i,l) zip (i1,l1) becomes ( (i,l), (i1,l1) )
        # one element of ZIPPED is a tuple of multiple (i,l) 
        ########

        l = tuple(ds_l) # cannot pass list, use tuple instead
        merge_ds = tf.data.Dataset.zip(l)   # A (nested) structure of datasets.
        
        # element_spec ((i1,l) , (i2,l))
        assert len(merge_ds.element_spec) == nb_heads #ie len(ds_l)

        #### checking
        tu = iter(merge_ds).next()
        # t is a tuple with as many element as merged dataset (ie heads). 
        # each element is (i,l) , at least 2
        #  ie ( (i,l), (i,l) , ..)
        assert len(tu) == nb_heads # ie nb of heads

        # 0: (<tf.Tensor: shape=(1...0e-01]]])>, <tf.Tensor: shape=(1...=float32)>)
        # 1: (<tf.Tensor: shape=(1..., 16.]]])>, <tf.Tensor: shape=(1...=float32)>)

        #for e, e1, _ in merge_ds.take(1): # does not work for 2 dataset, expected 3 got 2

        take = merge_ds.take(1)
        for el in take:  #run only once. could also use tu = iter(merge_ds).next()
  
            # (
            # (<tf.Tensor: shape=(1...0e-01]]])>, <tf.Tensor: shape=(1...=float32)>), 
            # (<tf.Tensor: shape=(1..., 16.]]])>, <tf.Tensor: shape=(1...=float32)>)
            # )
            assert type(el) in [tuple]
            i = el[0][0] # TensorShape([16, 34, 2])
            l = el[0][1] # TensorShape([16, 6]
            i1 = el[1][0] # TensorShape([16, 34, 1])
            l1 = el[1][1] # we have at least 2  TensorShape([16, 6])
            assert l.numpy().all() == l1.numpy().all() # same labels. need numpy to use .all
            assert i.shape[0] == i1.shape[0] # same batch and seqlen
            assert i.shape[1] == i1.shape[1] # 3rd dimension can be different [['temp', 'month'], ['pressure'] ]

        ########
        # STEP 2
        # create ds1 dataset with inputs heads only , element is tuple (input1, input2 , ..)
        ########
        

        def m(*args):  # variable number of parameters # get list of input only dataset , ie all e's[0]
            # args: list of (e,l)
            l=[x[0] for x in args] # list of input only, ie disgard labels
            return(tuple(l)) # tuple and dict are OK for nested dataset

        #ds1 = merge_ds.map(lambda e1,e2: (e1[0],e2[0])) # does not work. need to work with variable number of heads

        # input only, labels disgarded
        ds1 = merge_ds.map(m)

        assert len(ds1.element_spec) == nb_heads 

        # ds1.element_spec[0] TensorSpec(shape=(None, 90, 5), dtype=tf.float64, name=None)
        # ds1.element_spec[1] TensorSpec(shape=(None, 90, 2), dtype=tf.float64, name=None)

        ###########
        # STEP 3:
        # create ds2 dataset with labels only
        ###########


        def m1(*args):  # variable number of parameters  # e1 (i,l) e2 (i1,l1) ...
            # args: list of (e,l)
            for e in args:
                return(e[1])
                break   # all labels are the same
            
        #ds2 = merge_ds.map(lambda e1,e2: e1[1]) only works for 2 heads
        ds2 = merge_ds.map(m1)

       # ds2.element_spec TensorSpec(shape=(None, 3), dtype=tf.float32, name=None)
        
        assert len(ds1) == len(ds2)

        #########
        # step 4:
        # create dataset with both ie ((in, in1, in...) , l)
        #########

        # list, tuple and dict are OK for model input
        # tuple and dict are OK for nested dataset
        
        ds = tf.data.Dataset.zip((ds1, ds2)) 
        
        assert len(ds) == len(ds1)

        e,l = iter(ds).next() 
        assert len(e) == nb_heads
        assert type(e) in [tuple]

        # e[0].shape TensorShape([16, 90, 5])
        # e[1].shape TensorShape([16, 90, 2])
        # l.shape TensorShape([16, 3])
    

    ##############################################
    # ds is final dataset
    ##############################################
    #### len(ds) * batch_size may not be the number of samples, as last batch is not complete 
    # nb_seq set above (expensive, compute only once)

    # len(ds) is number of batches  
    #nb_batch =len(list(ds.as_numpy_iterator())) # nb of element if iterator
    #assert nb_batch == len(ds) 

    # get number of individual samples (ie sequences)
    #nb_seq = len(list(iter(ds.unbatch()))) # another simpler way to get nb of element regardless of batches

    # nb_seq already computed for last head. 
    
    print('\nDATASET CREATED. %d heads. %d batches, %d seq. from X %d hours %d days' %(nb_heads, len(ds), nb_seq, X.shape[0], X.shape[0]/len(retain))) 
    

    ##################################
    ####### look at dataset structure
    ##################################
    print("DATASET: look at it")

    print("average batch: %0.2f"  %(nb_seq/len(ds)))

    # incomplete batches ?
    # goes thru all elements of dataset

    # dataset can have one or many head
    # dataset can contain labels (used for training, unseen), or not (predict tomorrow, post mortem)

    if nb_heads == 1:

        if type(ds.element_spec) in [tuple]:
            # tuple of tensorspec, ie dataset yield labels
            # (i,l)
            assert labels
            print('!!! single head dataset with labels. element spec: ', ds.element_spec) # tuple of tensor spec (labels = True) or single tensorspec
            
            i,l = ds.as_numpy_iterator().next() # get one element . one line equivalent to for e in ds.take(1):
            print("input shape:", i.shape)
            
            # uncomplete batches 
            for i , (f,l) in enumerate(ds):
                if f.shape[0] != batch_size or l.shape[0] != batch_size:
                    print("dataset: %d of %d incomplete batch " %(i, len(ds)), f.shape, l.shape)
                    print("batch misses  %d sequences"  %(batch_size - f.shape[0]))

        else:
            # i
            # dataset created with labels = False ,for inference
            # element_spec is a single tensorspec
            assert not labels
            print('!!! single head dataset WITHOUT labels . element spec: ', ds.element_spec) # tuple of tensor spec (labels = True) or single tensorspec

            i = ds.as_numpy_iterator().next() # get one element . one line equivalent to for e in ds.take(1):
            print("input shape:", i.shape)

            # uncomplete batches 
            for i , f in enumerate(ds):
                if f.shape[0] != batch_size :
                    print("dataset: %d of %d incomplete batch " %(i, len(ds)), f.shape)
                    print("batch misses  %d sequences"  %(batch_size - f.shape[0]))

    else:

        #e,l = iter(ds).next() 
        # e[0].shape TensorShape([16, 90, 5])
        # e[1].shape TensorShape([16, 90, 2])
        # l.shape TensorShape([16, 3])
       
        # ( (i1,i2..) ,l)
        # (i1,i2..)

        if type(ds.element_spec[0]) in [tuple]:
            # labels
            assert labels
            print('!!! %d head dataset WITH labels . element spec: ' %nb_heads, ds.element_spec)

            i_,l = ds.as_numpy_iterator().next() # get one element . one line equivalent to for e in ds.take(1):

            for i__ in i_:
                print("head:" , i__.shape)

            for i , (f,l) in enumerate(ds):
                # just check labels
                assert type(f) in [tuple]
                if l.shape[0] != batch_size: 
                    print("dataset: %d of %d incomplete batch " %(i, len(ds)), l.shape)
                    print("batch misses  %d sequences"  %(batch_size - l.shape[0]))

        else:
            assert not labels
            print('!!! %d head dataset WITHOUT labels . element spec: ' %nb_heads, ds.element_spec)

            i_ = ds.as_numpy_iterator().next() # get one element . one line equivalent to for e in ds.take(1):

            for i__ in i_:
                print("head:" , i__.shape)

            for i , (f) in enumerate(ds):
                assert type(f) in [tuple]
                if f[0].shape[0] != batch_size:
                    print("dataset: %d of %d incomplete batch " %(i, len(ds)), f[0].shape)
                    print("batch misses  %d sequences"  %(batch_size - f[0].shape[0]))
   
    
    # nb batch * batch size is not equel to nb sequence, some batches not complete
    
    m = len(ds) * batch_size - nb_seq # uncomplete batch ?
    print("last batch may be incomplete, would miss %d seq" %m)  # 4 39*16 = 624   
    

    # check nb of sequence vs nb of rows (stride)
    # DROP this test. a bit too moving parts with strides
    #assert nb_seq == X.shape[0]/len(retain) - (days_in_seq-1) # 620 row gives 618 seq of 3 days. 4 rows 2 seq


   
    ################
    # various check
    ################

    # take(2) is new data set containing at most 2 elements 
    # element could be either a tensor, or a tuple of tensor
    # type <class 'tensorflow.python.framework.ops.EagerTensor'>

    print("DATASET. exception ahead if anything goes wrong")

    if labels:
        for x, y in ds.take(1): # loop runs ONCE per element/batch and yield ONE batch. take 1, loop run only once, take 2, runs twice
            # x could be a tensor or a tuple of tensors
            if len(ds) >1:
                assert y.shape[0] == batch_size   
                
            if len(ds) == 1:
                assert y.shape[0] == batch_size -m  # if only one batch, could be less than 16
                
            if categorical:
                assert y.shape[1] == hot_size
            
            try:
                x.shape # a tensor
            except:
                x = x[0] # a tuple of tensor

            if len(ds) >1:
                assert x.shape[0] == batch_size
            if len(ds) == 1:
                assert x.shape[0] == batch_size -m 

            assert x.shape[1] == seq_len
            #assert x.shape[2] == num_features    #  not defined

    else:
        for x in ds.take(1): 
            # average = False TensorShape([1, 45, 5]) else, TensorShape([13, 45, 5])
            try:
                x.shape # a tensor  , single head
            except:
                x = x[0] # a tuple of tensor

            assert x.shape[1] == seq_len
            
            #assert x.shape[2] == num_features   
            # x.shape[0] = nb of sequences , eg 13 (build 1, stride 2, so 24/2 + 1)
            
    ####################
    # SHUFFLE
    ####################

    if shuffle:

        print("DATASET: shuffling. fix seed to get consistent results")
        #tf.random.set_seed(1234)
        ds = ds.shuffle(seed = 1234, buffer_size = len(ds)*batch_size) # buffer_size elements,

    else:
        print("DATASET: NOT shuffling")

    ####################
    # CACHE
    ####################

    ds = ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    nb_hours = X.shape[0] # before sampling, stride

    return(ds, nb_seq, nb_hours)

###########################
# GOAL: fixed SPLIT
###########################
# other sklearn randim split, kfold, timeseriessplit, but need to use array, not dataset

def fixed_val_split(ds, nb_seq, nb_hours, split, retain, batch_size):

    #l = len(list(iter(ds)))

    l = len(ds)

    print("\nsplitting dataset, len in batches", l, split)

    # split dataset. all heads split the same way
    train_size = int(split[0]*l) # in nb of batches  
    val_size = int(split[1]*l) # 39    31, 3  int() effect    test 5

    train_ds = ds.take(train_size)
    temp= ds.skip(train_size) # val + test
    val_ds = temp.take(val_size)
    test_ds= temp.skip(val_size)

    print("building size dict")

    # batch only applies to tensorflow dataset.
    # if using sklearn, need to use arrays

    print("total batch %d. train %d, validation %d, test %d" %(len(ds), len(train_ds), len(val_ds), len(test_ds)))
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(ds)
    assert train_size == len(train_ds)
    assert val_size == len(val_ds)


    """
    this is WAY TOO EXPENSIVE for large dataset.
    so just multiply by batch size and call it a day

    # len(ds) ok, len(ds.unbatch()) not ok. list(iter) returns list of all elements
    s = len(list(ds.unbatch().batch(1).as_numpy_iterator())  ) # 618
    #assert s == len(list(iter(ds.unbatch().batch(1)))) , "iter vs .as_numpy_iterator"
    tr = len(list(iter(train_ds.unbatch().batch(1))))   # 362
    va = len(list(iter(val_ds.unbatch().batch(1))))   # 48
    te = len(list(iter(test_ds.unbatch().batch(1))))   # 202  

    # WTF !!!!! generates exception on assert below  
    #assert s == tr+va+te, "mismatch sample total %s, train %d, val %d, test %d" %(s,tr,va,te)  # 
    print("total seq %d. train %d, validation %d, test %d. delta ? %d" %(s, tr, va, te, s-(tr+va+te)))
    """    

    ds_dict = {

    "days":  nb_hours/len(retain), 
    "hours": nb_hours,
    "seq":nb_seq,

    "batch": {
        "total": len(ds),
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds)
    },

    "samples": {
        "total": nb_seq,
        "train": len(train_ds) * batch_size,
        "val": len(val_ds) * batch_size,
        "test":len(test_ds) * batch_size
    }

    }


    # normally, train and val are whole number of batches, ie sample % batch_size ==0
    # test sample / batch_size may be decimal. last batch not complete

    return(train_ds, val_ds, test_ds, ds_dict)

