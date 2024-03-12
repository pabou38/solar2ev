#!/usr/bin/env python3

import sys,os
import time, datetime
import tensorflow as tf
import numpy as np
import random

sys.path.insert(1, '../PABOU') # this is in above working dir 
try:
    print ('importing pabou helper')
    import pabou # various helpers module, one dir up
except:
    print('cannot import pabou. check if it runs standalone. syntax error there will fail the import')
    exit(1)

"""
Tensor object has no attribute 'astype'.
If you are looking for numpy-related methods, please run the following:
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
"""


# https://www.tensorflow.org/tutorials/customization/custom_layers?hl=en

#########################
# creates 3 baseline model
# zeroR, random, random W
#########################

#################################
# define custom layers

# __init__ , where you can do all input-independent initialization
# build, where you know the shapes of the input tensors and can do the rest of the initialization
# call, where you do the forward computation
#     call is invoked at model instanciation with NONE in batch

# return constant value, typically majority class
class ZeroR(tf.keras.layers.Layer):
    def __init__(self, constant): # to be returned, as one hot
        super(ZeroR, self).__init__()
        self.constant = constant # array([0., 0., 1., 0.], dtype=float32)
        #print('init ZeroR layer. constant: ', self.constant)
        self.indice = np.argmax(self.constant) # int64  2
        self.max = len(constant) # hot size

    # called with batch = None when invoquing layer(x) in model definition
    # called with batch = 16 and actual data when invoquing model(ds) , eg in evaluate

    # calling ZeroR layer. input shape: (None, 34, 1) 
    # calling ZeroR layer. input shape: (16, 34, 1) with actual data
    # WTF !!!!!! !!!! with vscode, cannot set breakpoint  in 1st call batch = None
    # only in 2nd call with batch set 

    # build output REGARDLESS of whether batch is None or set
    
    
    def call(self, inputs, training=False):  # Defines the computation from inputs to outputs
        # inputs: TensorShape([16, 34, 3])    actual data or None, ..

        #print('ZeroR call() is invoked with:', inputs.shape)

        self.batch_size = inputs.shape[0] # None or 16
        self.seqlen = inputs.shape[1]
        self.feature_dim = inputs.shape[2]

        self.z = tf.zeros([self.seqlen,1])  # TensorShape([34, 1])
        # need [,1] otherwize matmul fails:  Shape must be rank 2 but is rank 1 input shapes: [?,34], [34]
        # tf.zeros(5) tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 0., 0., 0., 0.], dtype=float32
        # tf.zeros([1,5]) tf.Tensor: shape=(1, 5), dtype=float32, numpy=array([[0., 0., 0., 0., 0.]], dtype=float32)
        
        #print(inputs.shape, self.z.shape) # (16, 34, 1) (34, 1)

        # remove last dim. slicing magic
        inputs = inputs[:,:,0] # TensorShape([16, 34])

        # make sure tensor can be mathmul:  TensorShape([16, 34]) x TensorShape([34, 1]) , 
        assert len(inputs.shape) == len(self.z.shape)
        assert inputs.shape[1] == self.z.shape[0]

        #
        
        try:
            self.x = tf.matmul(inputs, self.z) # build output REGARDLESS of whether batch is None or set zero. TensorShape([16, 1]) 
            self.x = self.x + self.indice # add int64
            # by default, cannot use np syntax on tensor , eg .astype(int)
            self.x = tf.cast(self.x, tf.int32) # one hot need int, x was float32 TensorShape([16, 1])
            self.x = tf.one_hot(self.x, self.max) # TensorShape([16, 1, 4])
            self.x = self.x[:,0,:] # Fuck, one dimension too many TensorShape([16, 4])

            # TensorShape([16, 4])

            #assert self.x.shape[0] == self.batch_size, "shape"    called first with batch size = None
            assert self.x.shape[1] == self.max, "shape" # specify, otherwize str(e) is ""
        

            return(self.x) 

        except Exception as e:
            print('exception call() ',str(e)) # In[0] and In[1] has different ndims: [16,34] vs. [34] [Op:MatMul]

    
        """
        SPENT A LOTTTTTTT of time on this
        the problem is that call() is called twice when layer is used with layer(input)
        first with batch = None, then batch = 16
        but cannot return value with batch being set explicitly 'either 1 or 16) inside call. will fail at model.evaluate
        output shape must be (None,6)
        so use computation from custom layer example, and make sure it returns constant


        the following DOES NOT work

        b = inputs.shape[0] # None at layer definition, ie model building. batch when calling model to predict
        if b == None: # input batch size available
            b = 1
        # return b x fixed tensor. there is likely a better way
        t = [self.constant for i in range(b)]
        t = tf.stack(t) # eager tensor TensorShape([16, 6])
        # at this point, this layer.output is KerasTensor: type_spec=NoneTensorSpec() (created by layer 'zero_r')
        # evaluate, fit will fail, but calling the model is OK
        return(t)
        """

    # called at layer(input)
    def build(self, input_shape):  # Create the state of the layer (weights)
        # TensorShape([None, 34, 3])

        # not needed, as there are no stated.  
        # just playing with tf.variable
        try:
            z = tf.zeros_initializer()
            z1= z(shape=(input_shape[-1],input_shape[1]), dtype='float32') # cannot use input_shape[0]
            #print(z1.shape) # (1, 34)
            z2 = tf.Variable(initial_value= z1, trainable=True)
            #print(z2.shape) # (1, 34)
        except Exception as e:
            print(str(e))

        
        

# return random value weigthed on class histograms
class random_w(tf.keras.layers.Layer):
    def __init__(self, l):
        super(random_w, self).__init__()
        self.l = l # list of weigths
        self.max = len(l)

    def call(self, inputs):  # Defines the computation from inputs to outputs

        self.seqlen = inputs.shape[1]
        self.num_features = inputs.shape[2]
        #print('seqlen %d, num features %d' %(self.seqlen, self.num_features))

        self.r = random.choices(*[range(0,self.max)],weights=self.l, k=1)
        self.r = random.randint(0, self.max-1) # Return a random integer N such that a <= N <= b

        self.z = tf.zeros([self.seqlen,1]) 
        inputs = inputs[:,:,0]
        
        try:
            self.x = tf.matmul(inputs, self.z) # build output REGARDLESS of wether batch is None or set zero [16, 1])
            self.x = self.x + self.r # add int64
            # by default, cannot use np syntax on tensor , eg .astype(int)
            self.x = tf.cast(self.x, tf.int32) # one hot need int, x was float32
            self.x = tf.one_hot(self.x, self.max) # (None, 1, 6) (16, 1, 6)
            self.x = self.x[:,0,:] # Fuck, one dimension too many

            #print('returned by call: ', self.x.shape) #  (16, 6)
            return(self.x) 

        except Exception as e:
            print(str(e)) # In[0] and In[1] has different ndims: [16,34] vs. [34] [Op:MatMul]
        
        
        
    def build(self, input_shape):  # Create the state of the layer (weights)
        pass




# return random value
# max integer, ie number of classes
class random_l(tf.keras.layers.Layer):
    def __init__(self, max):
        super(random_l, self).__init__()
        self.max = max

    def call(self, inputs):  # Defines the computation from inputs to outputs

        self.seqlen = inputs.shape[1]
        self.num_features = inputs.shape[2]
        #print('seqlen %d, num features %d' %(self.seqlen, self.num_features))

        self.r = random.choice(*[range(0,self.max)])   #[range(0,1)] is [range(0,1)] use unpacking *
        self.z = tf.zeros([self.seqlen,1]) 
        inputs = inputs[:,:,0]
        
        try:
            self.x = tf.matmul(inputs, self.z) # build output REGARDLESS of wether batch is None or set zero [16, 1])
            self.x = self.x + self.r # add int64
            # by default, cannot use np syntax on tensor , eg .astype(int)
            self.x = tf.cast(self.x, tf.int32) # one hot need int, x was float32
            self.x = tf.one_hot(self.x, self.max) # (None, 1, 6) (16, 1, 6)
            self.x = self.x[:,0,:] # Fuck, one dimension too many

            #print('returned by call: ', self.x.shape) #  (16, 6)
            return(self.x) 

        except Exception as e:
            print(str(e)) # In[0] and In[1] has different ndims: [16,34] vs. [34] [Op:MatMul]
        
        

    def build(self, input_shape):  # Create the state of the layer (weights)
        pass

#################################
# create fake models
# for fun, baseline_0 has inputs concatenated
# need to know input shape. could get this from model or dataset
#################################

#input shape: Shape tuple (not including the batch axis), or TensorShape instance (not including the batch axis).
# get input shape(s) from dataset, in a format compatible with Input layer, ie without batch dim
# covers case of multi head

def create_baseline_ZeroR(ds, constant, loss, metrics):
    ZeroR_layer = ZeroR(constant) # call init         

    # get list of input shape from dataset 
    input_shape_list = pabou.get_input_shape_from_dataset(ds) # [(34, 3)]


    # define baseline models to compare trained model with
    # fake model still concatenates input
    inputs= []
    for shape in input_shape_list:
        inputs.append(tf.keras.layers.Input(shape= shape))
    
    x = tf.keras.layers.Concatenate()(inputs) # no strictly required as model returns value independant from inputs

    outputs = ZeroR_layer(x) #  CUSTOM LAYER. IGNORE x. returns constant
    # calls class build(TensorShape([None, 34, 3])) . do not do anything there, as there are no state to set
    # calls class call(TensorShape([None, 34, 3])

    # WARNING: call(TensorShape([16, 34, 3]) with actual data is at inference time, model(data)

    baseline = tf.keras.Model(inputs=inputs, outputs=outputs, name = 'baseline_constant')

    # NEEDED ???? likely yes, as it define metrics used for evaluate
    baseline.compile(optimizer=tf.keras.optimizers.Adam(),  loss=loss,  metrics=metrics) 

    return(baseline)


def create_baseline_random(ds, max, loss, metrics):
    random_layer = random_l(max) # call init

    # get input shape from dataset 
    input_shape = pabou.get_input_shape_from_dataset(ds)

    inputs= []
    for shape in input_shape:
        inputs.append(tf.keras.layers.Input(shape= shape))
    
    x = tf.keras.layers.Concatenate()(inputs)

    outputs = random_layer(x) # call build and call
    baseline = tf.keras.Model(inputs=inputs, outputs=outputs, name='baseline_random')

    baseline.compile(optimizer=tf.keras.optimizers.Adam(),  loss=loss,  metrics=metrics) 

    return(baseline)


#####################
# return ramdom, weigthed with class frequency
#####################
def compute_w (histo_prod):# to compute random based on class weigths. histo_prod ndarray
    weigth = []
    nb_samples = histo_prod.sum()
    for h in histo_prod:
        weigth.append(h/nb_samples)
    #assert sum(weigth) == 1  # 0.9999999999999999  or use almostequal
    assert 1- sum(weigth) < 0.00000009
    return(weigth)

# random based on class histogram
def create_baseline_random_w(ds, histo_prod, loss, metrics):
    weigth = compute_w(histo_prod)
    random_layer = random_w(weigth) # call init
    # get input shape from dataset 
    input_shape = pabou.get_input_shape_from_dataset(ds)

    inputs= []
    for shape in input_shape:
        inputs.append(tf.keras.layers.Input(shape= shape))
    
    x = tf.keras.layers.Concatenate()(inputs)

    outputs = random_layer(x) # call build and call
    baseline = tf.keras.Model(inputs=inputs, outputs=outputs, name='baseline_random_w')

    baseline.compile(optimizer=tf.keras.optimizers.Adam(),  loss=loss,  metrics=metrics) 


    return(baseline)

#############################################
# EXAMPLE
#############################################

class SimpleDense(tf.keras.layers.Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    print('build ', input_shape) # build  (2, 3)
    # here , the batch dim, 2 is available , but NOT USED, NOT REQUIRED in computation

    w_init = tf.random_normal_initializer()

    # W 3X4
    # 4 is internal variable
    # 3 is input last dim; ie NOT BATCH

    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
        trainable=True)

    print("W ", self.w.shape)  # W  (3, 4)

    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      print('call ', inputs)
      return tf.matmul(inputs, self.w) + self.b



if __name__ == "__main__":
    print('create custom baseline layer')

    # input 2X3    
    # W 3x4       4 internal variable, 3 is extracted from inputs. computation does not need batch 
    
    # output 2X4

    # Instantiates the layer. 
    # call init
    nb = 4
    linear_layer = SimpleDense(nb)

    # This will also call `build(input_shape)` and create the weights.
    inputs = tf.ones((2, 3))
    print(inputs, inputs.shape)
    # [[1. 1. 1.]
    # [1. 1. 1.]], shape=(2, 3), dtype=float32) (2, 3)

    y = linear_layer(inputs)
    print(y, y.shape)

    # Y has batch dim and dim of 
    assert y.shape[0] == inputs.shape[0]
    assert y.shape[1] == nb

    #print(linear_layer.weights) # W
    # These weights are trainable, so they're listed in `trainable_weights`:
    #print(linear_layer.trainable_weights) 

else:
    pass