#!/usr/bin/env python3

import tensorflow as tf

from tensorflow import keras
from keras import backend as K
import keras_tuner as kt

import numpy as np


######################################
# simple attention custom layer
# build LSTM lodel
# tuner
# training custom call back
# get metric_to_watch, metric and loss 
######################################


# https://www.tensorflow.org/guide/keras/preprocessing_layers?hl=en

######################
# GOAL: define custom layer for attention
#####################

class simple_attention (tf.keras.layers.Layer):

    # https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
    # Variables set as attributes of a layer are tracked as weights of the layers (in layer.weights)
    #A layer encapsulates both a state (the layer's "weights")  and a transformation from inputs to outputs (a "call", the layer's forward pass).

    # unit not used, just for example
    # head just used to make name unique (for multi head)
    # avoid spaces
    def __init__(self, units=32, head=1, **kwargs):   # addition arguments
        self.name_at = "attention_" + str(head)
        super(simple_attention, self).__init__(name=self.name_at, **kwargs)
        self.units = units


    # Create the state of the layer (weights)
    # we recommend creating layer weights in the build(self, inputs_shape) method
    # will be called at 1st call(), when input dim is known

    # called when creating model

    def build(self,input_shape):
        # input_shape TensorShape([None, 66, 128])     seqlen, nb of unit in LSTM

        # input shape is Tx * 2d (Tx: number of input token, 2d is dim of token embedding + dim of decoder hidden state)
        # d is number of hidden unit in RNN
        # W convert is dim 2d, 1  to convert to energy vector of dim Tx * 1

        # W TensorShape([128, 1])  
        # b TensorShape([66, 1])

        # add_weight() offers a shortcut to create weights:
        # initializer='random_normal'
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal", trainable=True)
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros", trainable=True)

        super(simple_attention, self).build(input_shape)

    # Defines the computation from inputs to outputs
    # The __call__() method of your layer will automatically run build the first time it is called. 

    # WTF. breakpoint in call() does not seem to work

    def call(self, inputs):

        # input: all encoder hidden states (one per input token)
        # in case of seq2seq, each encoder hidden state (annotation, a1, a2 , atx) is CONCATENATED with decoder hidden state for current word to decode, ie d = 2d

        # input dim: Tx * d (RNN nb units)

        # multiply (a1, a2, atx)* d input by W+b , yield e1, e2, etx  , each scalar
        x = tf.keras.backend.dot(inputs,self.W)+self.b
        x = tf.keras.backend.tanh(x)

        # get rid of one dim
        et=tf.keras.backend.squeeze(x,axis=-1)

        # convert energy in weigths
        at=tf.keras.backend.softmax(et)

        at=tf.keras.backend.expand_dims(at,axis=-1)

        # weigth annotation with computed weigths
        output=inputs*at  # dim Tx * d

        # ct is sum. dim d * 1
        # replace a fixed context vector, ie last hidden state (w/o attention)
        context_vector = tf.keras.backend.sum(output,axis=1)

        # will be passed thru classification (dense, softmax)
        return context_vector

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    # get_config() method collects the input shape and other information about the model.
    # If you need your custom layers to be serializable as part of a Functional model, you can optionally implement a get_config() method:

    # layer = Linear(64)
    #config = layer.get_config()
    #print(config)
    #new_layer = Linear.from_config(config)

    def get_config(self):
        config = super(simple_attention, self).get_config() # 'name': 'linear_8', 'trainable': True, 'dtype': 'float32'
        config.update({"units": self.units})
        return config
        


###########################
# GOAL: build LSTM model
###########################

def build_lstm_model(ds, categorical, name='base_lstm', units=64, dropout_value=0.2, num_layers=1, nb_dense=0, use_attention = False):

    # model parameters passed as argument, so can be used with kerastuner
    # dataset used to determine input shape, 

    # WARNING .... also used for normalization adapt, normally only use train dataset, so that the model does never sees anything about test dataset
    # BUT simpler not to use train dataset, no dependancy on validation strategy

    # single head (one sequence) or multi head
    # categorical used to return metrics and loss (model.compile)

    print("\nbuild model: use attention ? %s" %use_attention)

    # different metric and loss for regression and classification
    # metrics used by compile()
    # met and val_met will be in history
    # met returned by evaluate()
    (watch, metrics, loss) = get_metrics(categorical)

    # !!! make sure layer names are unique
    
    # list, tuple and dict are OK for model input
    # tuple and dict are OK for nested dataset

    # ds used to infer input shape, and adapt normalization layer (so avoid test dataset)

    # create list containing all Input layers
    all_inputs=[] # list of all input layers, one per head
    all_norms = [] # one norm layer per input head
    
    # get input shape from training set
    # build model inputs as list 
    # num_features could be different for each Input (aka head)
   
    e,l = iter(ds).next() # get one batch to hot size

    if categorical: 
        hot_size = l.shape[1]
        assert len(l[0]) == l.shape[1]

    multi_head = type(e) in [tuple, dict] # list is not used for nested dataset
    # input could be tensor or tuple of tensors

    print("\nbuild lstm: multi head: %s. categorical %s" %(multi_head, categorical))


    if multi_head: 

        ################################
        # multi head
        ################################

        nb_head = len(e)
        print('%d heads' %nb_head)

        global xx # needed as index to inputs and norms list 

        for xx, e1 in enumerate (e): # for each dataset , ie each head TensorShape([16, 90, 5])


            ###################
            # input layer
            ###################
            seq_len = e1.shape[1] 
            num_features = e1.shape[2]

            # create input layer and store them in list, for later use
            all_inputs.append(tf.keras.layers.Input(shape=(seq_len, num_features), name = 'input_%d' %xx)) # use input layer name to record model name
            

            ###################
            # normalization layer
            ###################

            # create normalize layer and store them in list, for later use
            axis = -1 # Integer, tuple of integers, or None.
            norm = tf.keras.layers.Normalization(axis=axis, name = 'norm_%d' %xx) # feature-wise normalization to numerical features # invert = True, denormalize
        

            ### need to adapt THIS norm layer on only ONE head, ie only ONE dataset
            # adapt for each input head , so need to get associated input, ie ith dataset in tuple
           
            """
            not a solution. Creates a dataset that deterministically chooses elements from datasets
            need a list of dataset and pick elements from them
            choice_dataset = tf.data.Dataset.from_tensors(i)
            assert choice_dataset.as_numpy_iterator().next() == i
            choice_dataset.map(lambda x: (tf.cast(x, tf.int64)))
            choice_dataset = tf.data.Dataset.from_tensors(tf.Variable(i, dtype=tf.int64))
            """

            def get_nth_ds(*args): # receive variable nb of argument (not a tuple)
                global xx # use global, as cannot pass this within .map below
                return args[xx]

            nth_ds = ds.map(lambda i, l: i) # return inputs only, exclude labels. # cannot use adapt(feature_ds) , this is a tuple , not a dataset
            nth_ds = nth_ds.map(get_nth_ds)

            print("adapting norm %d on dataset %s "%(xx, iter(nth_ds).next().shape))

            # adapt normalization layer on dataset (avoid test)
            norm.adapt(nth_ds)

            o = iter(nth_ds).next()
            print("before norm. one batch: std %d, mean %d" %(o.numpy().std(), o.numpy().mean()))
            o = norm(o) 
            print("after norm. one batch: std %d, mean %d" %(o.numpy().std(), o.numpy().mean()))

            # store norm layer per head
            all_norms.append(norm)

        # multi head
        
    else: 
        ################################
        # single head
        ################################


        # create INPUT and NORM layer. adapt norm on dataset. store in list
        
        print('single head. batch:', e.shape)
        seq_len = e.shape[1]
        num_features = e.shape[2]
        nb_head = 1

        all_inputs = [tf.keras.layers.Input(shape=(seq_len, num_features))] # use input layer name to record model name

        ##################
        # normalize
        ##################
        # axis
        #Integer, tuple of integers, or None. The axis or axes that should have a separate mean and variance for each index in the shape. 
        # For example, if shape is (None, 5) and axis=1, the layer will track 5 separate mean and variance values for the last axis. 
        
        #If axis is set to None, the layer will normalize all elements in the input by a scalar mean and variance. 
        # When -1 the last axis of the input is assumed to be a feature dimension and is normalized per index. 
        #Note that in the specific case of batched scalar inputs where the only axis is the batch axis, the default will normalize each index in the batch separately. In this case, consider passing axis=None. Defaults to -1.

        # Defaults to -1
     
        axis = -1 
        # This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1. 
        # It accomplishes this by precomputing the mean and variance of the data, and calling (input - mean) / sqrt(var)
        # The mean and variance values for the layer must be either supplied on construction or learned via adapt(). adapt() will compute the mean and variance of the data and store them as the layer's weights. 
        # adapt() should be called before fit(), evaluate(), or predict().
        norm = tf.keras.layers.Normalization(axis=axis, name = 'norm') # feature-wise normalization to numerical features # invert = True, denormalize
        
        # adapt on training set only. model should NEVER see test test
        # The mean and variance values for the layer must be either supplied on construction or learned via adapt(). 
        # adapt() will compute the mean and variance of the data and store them as the layer's weights.

        # During adapt(), the layer will compute a mean and variance separately for each position in each axis specified by the axis argument. 
        # To calculate a single mean and variance over the input data, simply pass axis=None.

        feature_ds = ds.map(lambda i, l: i) # return input only, exclude labels .element_spec: TensorSpec(shape=(None, 90, 6), dtype=tf.float64, name=None)
    
        # https://www.tensorflow.org/guide/keras/preprocessing_layers?hl=en
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization

        print("adapting norm on input dataset, shape:%s, %d batch" %(iter(feature_ds).next().shape, len(feature_ds)))
        norm.adapt(feature_ds) # either as a tf.data.Dataset, or as a numpy array

        ##################
        # just look at what Norm/adapt does , (ie after/before)
        ##################

        ## looking at one batch only is misleading
        # WARNING: norm(dataset) does not work
        o = iter(feature_ds).next() # TensorShape([16, 90, 6]).  o[0].shape TensorShape([90, 6])

        # normalize one batch
        print ("norm layer adapted ? %s" %norm.is_adapted)
        o1 = norm(o) # returns tensor

        print("before norm. one batch ONLY. NOT that meaningfull : std %0.1f, mean %0.1f" %(o.numpy().std(), o.numpy().mean()))
        print("after norm. one batch ONLY. NOT that meaningfull: std %0.1f, mean %0.1f" %(o1.numpy().std(), o1.numpy().mean()))

        ## look at norm effect on entiere dataset

        # approch 1.  not sure if valid for std
        mm = 0
        ss = 0
        i=0
        for i, a in enumerate(feature_ds):
            a1 = norm(a)
            mm = mm + a1.numpy().mean() # mean of mean OK
            ss = ss + a1.numpy().std() # how to compute std ?

        print("after norm. %d iteration on full dataset: std %0.1f, mean %0.1f" %(i, ss/i,mm/i))

        # approch 2:  convert a tf.data dataset to numpy or tensor. cannot do norm(dataset)
 
        a = list(iter(feature_ds.unbatch()))  # list of tensor. a[0].shape TensorShape([90, 6]). len(feature_ds) 1051. 
        b = np.asarray(a)  #b.shape (16816, 90, 6)  16816/16 = 1051

        n1 = tf.keras.layers.Normalization(axis=axis, invert=False)
        n1.adapt(b) # learn
        b1= n1(b) # normalize 


        # try to denormalize
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
        # https://stackoverflow.com/questions/73742308/looks-like-keras-normalization-layer-doesnt-denormalize-properly
        # was a bug in 2.10 ??? and F.. I use 2.10 on windows as last version with GPU 
        inv_n1 = tf.keras.layers.Normalization(axis=axis, invert=True)
        inv_n1.adapt(b) 
        b2 = inv_n1(b1) # denormalize

        # is b2 and b the same ?
        print("before: std %0.1f, mean %0.1f" %(b.std(), b.mean())) # std 372.8, mean 187.6
        print("normalized: std %0.1f, mean %0.1f" %(b1.numpy().std(), b1.numpy().mean())) # std 1.0, mean -0.0

        x = tf.__version__.split(".") # ['2', '10', '0']

        if int(x[0]) >=2 and  int(x[1]) > 10: 
            print("std %0.1f, mean %0.1f" %(b2.numpy().std(), b2.numpy().mean()))
        else:
            print("denormalization bug on tf 2.10")

        # !!!!! std = 0 means all samples are the same

        all_norms= [norm]

    #all_inputs and all_norms are list of one or more layers

    
    # where to apply normalization ?

    # OPTION 2: 
    # apply preprocessing to dataset, ie in pipeline
    # preprocessing will happen on CPU, asynchronously, and will be buffered before going into the model
    # preprocessing will happen efficiently in parallel with training

    #train_ds1 = train_ds.map(lambda x,y: (norm(x), y)) # return tuple
    #train_ds1 = train_ds1.prefetch(tf.data.AUTOTUNE)
    # and fit using above
    # use above for TPU, except for rescaling and normalization, which are ok for TPU

    # even if using OPTION 2, can create an inference only model which includes preprocessing
    #inp = tf.keras.layers.Input(shape=(seq_len, num_features))
    #x = norm(inp)
    #out = outputs = training_model(x) # already trained without norm layer
    #inference_model = tf.keras.Model(inp,out)

    
    # OPTION 1: 
    # include preprocessing layers in model
    # reduce the training/serving skew.
    # preprocessing will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration
    # best option for the Normalization layer, and for all image preprocessing and data augmentation layers.


    # one norm layer, and one or more stacked LSTM layers 
    # input index into input layers list , returns output of LSTM

    # one instance of this per head

    # default activation. activation="tanh"

    """
    Based on available runtime hardware and constraints, this layer will choose different implementations 
    (cuDNN-based or pure-TensorFlow) to maximize the performance. If a GPU is available and all the arguments to the layer 
    meet the requirement of the cuDNN kernel (see below for details), the layer will use a fast cuDNN implementation.

    activation='tanh' (default) required to use cudnn
    """

    # https://www.tensorflow.org/guide/keras/rnn?hl=en

    def one_lstm_stack(i:int):   # ith head
        print("building lstm stack %d" %i)
        x = all_inputs[i]  # takes input
        x = all_norms[i](x) # applies norm on input
        
        if num_layers == 1:

            #output = tf.keras.layers.Bidirectional (tf.keras.layers.LSTM(units))(x)

            if use_attention:
                # add attention. need all hidden states (return_sequence)
                output = tf.keras.layers.LSTM(units, return_sequences=True)(x)

                # head is used to make layer name unique
                context_vector = simple_attention(head=1)(output)
            else:
                # no attention. fixed context vector
                context_vector = tf.keras.layers.LSTM(units)(x)

            return(context_vector)

        else: # multiple stacked LTSM layers

            for _ in range(num_layers -1): # layers with return_sequences = True # returns a sequence of vectors
                #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
                x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
                # TensorShape([None, 66, 128]) for return_sequence = True

            # final LSTM
            #output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)) (x) # returns a vectors

            if use_attention:
                # add attention. need all hidden states (return_sequence)
                output = tf.keras.layers.LSTM(units, return_sequences=True)(x)

                # head is used to make layer name unique
                context_vector = simple_attention(head=i)(output)
            else:
                # no attention. fixed context vector
                context_vector = tf.keras.layers.LSTM(units)(x)

            return(context_vector) # TensorShape([None, 128])


    # list of output from each (possibly only) LSTM
    outs = []

    # connect each inputs+ norm to one (stacked) LSTM
    for i in range(nb_head): #for each head
        lstm_output = one_lstm_stack(i) 
        outs.append(lstm_output)


    # as many layer in outs as heads
    assert (len(outs) >1) == multi_head
    assert len(outs) == nb_head
     
    # multi heads. output of "parallels" LSTM are combined . 
    if len(outs) >1:
        x = tf.keras.layers.Concatenate()(outs)  # default axis=-1
        # It takes as input a list of tensors, all of the same shape except for the concatenation axis, 
        # and returns a single tensor that is the concatenation of all inputs.
        # reshape(2, 2, 5) reshape(2, 1, 5) Concatenate(axis=1)([x, y])  (2,3,5)

    else:
        x = outs[0] # else having problem at load time with concatenate


    ######### batch norm
    # work on activation output vs network input
    # normalize to mean=0 and std=1 , then rescale with LEARNED gamma and beta
    # work differently in training for() and inference evaluate(), predict()
    # model(x) default to model(x, training=False)

    # faster training. more "round" for gradient descent

    # Integer, the axis that should be normalized (typically the features axis), default -1
    #x = tf.keras.layers.BatchNormalization() (x)
    

    ############ DROPOUT
    if dropout_value != 0:
        x = tf.keras.layers.Dropout(dropout_value)(x)


    ############# DENSE
    if nb_dense !=0:
        x = tf.keras.layers.Dense(nb_dense, activation = "relu", name = 'dense') (x)

    """
    ############# DENSE with BAD:  batch norm, activation, dropout
    if nb_dense !=0:
        x = tf.keras.layers.Dense(nb_dense, activation = None, name = 'dense') (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Activation("relu") (x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout_value)(x)
    """ 

    ############# last lateyer , SOFTMAX or dense(1)
    if categorical:
        outputs = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax')(x) # for one hot
    else:
        outputs = tf.keras.layers.Dense(1, name = 'last_dance')(x) # for scalar


    model = tf.keras.Model(inputs=all_inputs, outputs=outputs, name = name)

    # metrics and loss are based on categorical
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  loss=loss,  metrics=metrics) 

    print("nb of params: %d" %model.count_params())

    return(model)
 

###################################### 
# GOAL: custom epoch call back
#######################################

class CustomCallback_show_val_metrics(tf.keras.callbacks.Callback):

    # callbacks=[CustomCallback()]
    # The logs dict contains the loss value, and all the metrics at the end of a batch or epoch.
    # The logs dictionary that callback methods take as argument will contain keys for quantities relevant to the current batch or epoch (see method-specific docstrings).
    # `val_acc` which is not available. Available metrics are: loss,categorical accuracy,precision,recall,prc,val_loss,
    # val_categorical accuracy,val_precision,val_recall,val_prc

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        # ['loss', 'mae', 'mape', 'mse', 'rmse', 'val_loss', 'val_mae', 'val_mape', 'val_mse', 'val_rmse', 'lr']

        # show only validation 
        s = ""
        for k in keys: 
            if k.find("val") != -1 and k.find("loss") == -1:
                s = s + "%s: %0.3f" %(k, logs[k]) + ", "

        print("epoch: %d, %s"  %(epoch, s), end = "\r" )


    def on_train_end(self, logs=None):
        keys = list(logs.keys())

        s = ""
        for k in keys:
            if k.find("val") != -1 and k.find("loss") == -1:
                s = s + "%s: %0.3f" %(k, logs[k]) + ", "

        print("Training end: %s"  %s)
        



######################################
# GOAL: defines metric_to_watch, metric to compute and loss function
######################################

def get_metrics(categorical):

    # metrics used by model.compile()
    # met and val_met will be in history
    # met returned by evaluate()
    # typically on val_met used by early stop and reduce lr callbacks

    # name is used as keys 

    if categorical:

        # val_categorical accuracy,val_precision,val_recall,val_prc
        metric_to_watch = 'val_categorical accuracy' # used for early stop and reduce lr callbacks
        loss = tf.keras.losses.CategoricalCrossentropy()

        # WARNING: tf does not support precision, accuracy for multi class 
        # included anyway, but will be computed with sklearn
        # do not change order before thinking about it. can be returned as list

        metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='categorical accuracy'), 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='prc', curve='PR')
        ]

        
    else:

        # use str format.

        # using rmse vs mae : 5 x error of 1 vs 1 x error of 5

        # rather use mae

        #metric_to_watch = 'val_rmse' # used for early stop and reduce lr callbacks. use str, used as key in history_dict
        #loss = tf.keras.losses.MeanSquaredError()
        
        metric_to_watch = 'val_mae'
        loss = tf.keras.losses.MeanAbsoluteError()
    
        #metric_to_watch = 'val_mae' # used for early stop and reduce lr callbacks
        #loss = tf.keras.losses.MeanAbsoluteError()
        # loss='mean_absolute_error'

        # tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),   gets very large


        # looks order is important, will be the same used in evaluate()
        metrics = [
            tf.keras.metrics.RootMeanSquaredError(name = "rmse"),
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredLogarithmicError(name="msle")
        ]

        # tf.keras.metrics.MSE() is function, need Y and Y1
        # tf.keras.metrics.MeanSquaredError() is a class. can pass name

    return(metric_to_watch, metrics, loss)


if __name__ == "__main__":
    hp = kt.HyperParameters()
    print(hp.Int("units", min_value=32, max_value=512, step=32)) # 32
    # test 
else:
    pass