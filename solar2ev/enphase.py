#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import os, sys


import config_model

from typing import Tuple
# Alternatively I think since 3.9 or 3.10 you can just outright say tuple[int, int] without importing anything.

sys.path.insert(1, '../PABOU') # this is in above working dir 
try:
    print ('importing pabou helper')
    import pabou # various helpers module, one dir up
    import various_plots
    #print('pabou path: ', pabou.__file__)
    #print(pabou.__dict__)
except Exception as e:
    print('cannot import modules in ..  check if it runs standalone. syntax error will fail the import')
    exit(1)

installation = datetime.datetime(2021,3,12) 
# installed 11th but ignore that day (partial production). 1st day of full production
# 4x355w

##### WARNING: make sure downloaded csv start on March 12th

# max for normalization 20Kwh witgh 4x355 + 4x360
# 06/16/2021,"20,050"

# incremental installation . starts with 4x355
d1= datetime.datetime(2021,5,31)  # add 4x360w  May 2021
d2= datetime.datetime(2021,10,28) # add 2x400w  Oct 2021
d3= datetime.datetime(2022,5,21)  # add 2x400W  May 2022
d4= datetime.datetime(2023,10,2)  # add 1x400W  Oct 2023. observes 3.9kw max prod on enphase app for 1st time

# datetime.datetime.strptime(last_seen_date, "%Y-%m-%d")

capacity_1 = 4*355 # installed 11 mars 2021
capacity_2 = 4*360 + capacity_1# 30 may 2021
capacity_3 = 2*400 + capacity_2 # 27 oct 2021
capacity_4 = 2*400 + capacity_3 # 20 mai 2022
capacity_5 = 1*400 + capacity_4 # Oct 2023

total_capacity = capacity_5

factor_1 = total_capacity / capacity_1  # multiply production until 30 may 2021 included 3.4
factor_2 = total_capacity / capacity_2
factor_3 = total_capacity / capacity_3 
factor_4 = total_capacity / capacity_4  # 
factor_5 = total_capacity / capacity_5 # 1
assert factor_5 == 1

# solar production pannel normalization: 3.4, 1.7, 1.3, 1.1, 1.0

# to account for the fact that first batch of pannels have sligthly less exposition, powerfull micro-ondulers 
exposition_factor = 1

production_file = "enphase_downloaded.csv" # downloaded from enphase

# global to persist, max normalized production so far
max_energy = [0.0, 0.0, None]

production_column = "production" # added when creating input feature. used in config as well

###########################################################################
# GOAL: modify (normalize) production to cope for incremental pannel installation
###########################################################################

def normalize_energy(energy: float, date: datetime) -> Tuple[float, float]:
    # as if we always had the latest pannel configuration, ie multiply old value by >1
    # return modified energy and max energy seen so far
    # always returning one object; using return one, two simply returns a tuple.
    # -> Union[int, Tuple[int, int]]   can returns different type

    # keep track of max normalized energy so far
    global max_energy

    energy_to_normalize = energy # Kwh BEFORE normalize

    if date >= installation and date < d1:
        energy = energy * factor_1 *  exposition_factor
    elif date >= d1 and date < d2:
        energy = energy * factor_2 * exposition_factor
    elif date >= d2 and date < d3:
        energy = energy * factor_3 * exposition_factor
    elif date >= d3 and date < d4:
        energy = energy * factor_4 * exposition_factor
    elif date >= d4:
        pass
        #energy = energy* factor_5
    else:
        raise Exception
    
    # max 31 '2021-06-16 00:00:00'  20.05

    # keep track of max normalized energy so far
    if energy > max_energy[0]:
        max_energy[0] = energy # normalized
        max_energy[1] = energy_to_normalize
        max_energy[2] = date

    return(energy, max_energy) 


##########
# GOAL: clean dowloaded csv
# outlier, normalize (for incremental pannels)
# histograms
# used for initial downloaded AND delta (retrain, unseen)
# return df, max_prod
##########

def clean_solar_df_from_csv(production_file:str, plot_histograms=True) -> Tuple [pd.DataFrame, float]:

    # handle outlier (NOTE: use max production, but could also leverage format , ie use of "")
    # set date as index (to later use meteo's date to look up this data):  CAREFULL IF THIS FILE IS LESS RECENT THAN METEO
    # ditch last line (summary)
    # normalize daily production (use installation date passed as parameter)
    # plot histograms before and after normalization
    # return dataframe


    # looks like quote + , implies Kwh 

    #Date/Time,Energy Produced (Wh)
    #03/11/2021,92
    #03/12/2021,"4,233"

    #03/16/2021,"4,213"
    #03/17/2021,921

    print("\nclean solar csv (get data or unseen): %s" %(production_file))
    # plot = true, will plot histograms

    print('incremental installed base: %d, %d, %d, %d, %d' %(capacity_1, capacity_2, capacity_3, capacity_4, capacity_5))
    print('solar production pannel normalization: %0.1f, %0.1f, %0.1f, %0.1f, %0.1f' %(factor_1, factor_2, factor_3, factor_4, factor_5))
    # solar production pannel normalization: 3.4, 1.7, 1.3, 1.1, 1.0

    ########################
    # how to read csv
    ########################
   
    # NOTE could also use the type ("") of data to figure out entries in w vs kw
    #03/16/2021,"4,213"
    #03/17/2021,921

    # Specify the data type for each column
    my_dtype = {"Date/Time": str, "Energy Produced (Wh)": str}
    
    # read everything as str
    #my_dtype=str

    # object and str the same ?? almost https://stackoverflow.com/questions/34881079/pandas-distinction-between-str-and-object-types
    

    try:
        prod =  pd.read_csv(production_file)
        #prod =  pd.read_csv(production_file, dtype=str)
        #prod =  pd.read_csv(production_file, dtype=my_dtype) # force to read 
        #prod =  pd.read_csv(production_file, converters={"Energy Produced (Wh)":str}) 
    except Exception as e:
        print("cannot read enphase production csv file %s" %production_file)
        sys.exit(1)

    #print(prod.dtypes) # <class 'pandas.core.frame.DataFrame'>
    print("solar csv columns: ", prod.columns)

    #prod[prod.columns[0]]  = prod[prod.columns[0]].astype('str') NO NEEDED. already object
    # prod[prod.columns[1]].dtypes dtype('O')

    # data actually stored as str 
    # (prod[prod.columns[1]][3])
    #'938' <class 'str'>
    #(prod[prod.columns[1]][5])
    #'4,213'

    # remove last line , ie sum 
    prod = prod.iloc[:-1]

    
    prod.info() # print to console already
    print("solar csv stats\n", prod.describe()) # returns str count, unique top, freq


    ################## convert (time, float)
    # convert first columns to panda datetime, to be consistent with meteo dataframe
    # date stored as onject, ie string 03/11/2021
    
    # date, datetime, and time objects all support a strftime(format) method, to create a string representing the time under the control of an explicit format string.
    # % Day of the month as a zero-padded decimal number.
    # %m Month as a zero-padded decimal number.
    # %Y Year with century as a decimal number.
    # but columns 0 is not such an object

    # Convert Strings to Datetime in Pandas DataFrame from 03/11/2021
    prod[prod.columns[0]] =  pd.to_datetime(prod[prod.columns[0]], format='%m/%d/%Y')  
    # Date/Time             611 non-null    datetime64[ns]

    ##############################
    # manage Watt vs Kwh
    ##############################
   

    # in csv downloaded from enphase cloud (initial), some value seems to be reported in w vs kw, for less that 1kwh, 
    # or real production issue, eg snow, very very cloudy, or system down (grid down) 
    # do not interpret 92 as 92kw but as w

    # if set index to datetime, need to use columns 0 vs 1
    #03/11/2021,92
    #03/12/2021,"4,233"

    # option 1: use max production
    # option 2: detect use of " or ,

    ###############
    #OPTION 1
    # convert to float and replace with filter and lambda. convoluted
    ###############

    """

    # convert to float, (, to .)
    # prod.columns is list of colums
    prod[prod.columns[1]] = prod[prod.columns[1]].apply(lambda  x : float(x.replace(',','.')))

    # max value for production. beware there are some in watt vs kwh
    max_prod = prod[prod.columns[1]].max(skipna=True) # by default columns  axis=1 by row
    print('max value in production before outliers', max_prod)

    # use max expected production, ie 33
    max_expected_production = config_features.max_expected_production
    print("!!! everything larger than %d is considered outlier (ie reported in watt). PLEASE SET BASED ON INSTALLATION" %max_expected_production)

    filter = prod[prod.columns[1]] > max_expected_production # create filter , ie list of boolean
    outliers = prod[filter] # applie filter to df to create new df. 
    #  outlier is a dataframe  Columns: [date, energy]. shape (0, 2) if empty
    #print('outliers\n' , outliers)
    #print(outliers.info())
    
    print('%d outliers (ie larger than %d kwatt, and therefore considered watt)' % (outliers.shape[0], max_expected_production))
    if outliers.shape[0] >0: # max, min: exception if empty
        print('max %0.1f min %0.1f' % (max(outliers[outliers.columns[1]]), min(outliers[outliers.columns[1]]) ))
    
    # remove outliers OR devide by 1000
    # smaller outlier is 37.0
    # consider 37.0 to 987.0 as wh vs kwh in enphase energy report

    # this was watt, convert to kw
    prod[prod.columns[1]] = prod[prod.columns[1]].apply(lambda x : x/1000.0 if x > config_features.max_expected_production else x)

    """

    ###############
    #OPTION 2
    ###############

    def manage_watt(val):
        # when cleaning csv downloaded from enphase cloud , val is still str. having a comma means Kw, otherwize w
        # when csv created by unseen , ie querying web, val is a float 

        # to make sure; manages at this end 

        if isinstance(val, str): # downloaded from enphase

            if val.find(",") == -1:
                #print(val)
                return(float(val)/1000.0)
            else:
                val = val.replace(',','.')
                return(float(val))
            
        else: # likely csv was created by unseen/retrain and read back as float
            return(val)
        

    prod[prod.columns[1]] = prod[prod.columns[1]].apply(manage_watt)

    print("after managing outliers max %0.1f, min %0.1f" %(max(prod[prod.columns[1]]), min(prod[prod.columns[1]])))



    ############# check for zero production, (could be a sign of internet being down for several days, or snow on the pannels for several days)
    # fix it or not ? interpolate ?

    filter = prod[prod.columns[1]] == 0
    zero_prod = prod[filter]
    print("production csv has %d zero values" %len(zero_prod))

    """
         Date/Time  Energy Produced (Wh)
        4   2021-03-15                   0.0
        262 2021-11-28                   0.0
        263 2021-11-29                   0.0
        264 2021-11-30                   0.0
        268 2021-12-04                   0.0
        996 2023-12-02                   0.0
        """

    """
    11/27/2021,"1,076"
    11/28/2021,0
    11/29/2021,0
    11/30/2021,0
    12/01/2021,20
    12/02/2021,37
    12/03/2021,2
    12/04/2021,0
    12/05/2021,37
    12/06/2021,"1,128"
    """

    min_fix = config_model.enphase_min_fix
    filter = prod[prod.columns[1]] < min_fix
    print("production csv has %d values below %0.1f" %(len(prod[filter]), min_fix))

    """
         Date/Time  Energy Produced (Wh)
    4   2021-03-15                   NaN
    262 2021-11-28                   NaN
    263 2021-11-29                   NaN
    264 2021-11-30                   NaN
    268 2021-12-04                   NaN
    996 2023-12-02                   NaN
    """

    if config_model.enphase_interpolate:
        print("fix (replace with nan, then interpolate) enphase data below %0.1f" %min_fix)
        col = prod.columns[1]

        # replace all values below threshold with nan (to later interpolate)
        prod.loc[prod[col] < min_fix,  col] = np.nan # change based on condition. change in place
        # could also use np.where or .apply(lambda)
        assert prod.isna().sum().sum() == len(prod[filter])

        prod[col] = prod[col].interpolate(method = "linear")
        assert prod.isna().sum().sum() == 0




    ############### set index to date column
    # set index to datetime columns
    prod.set_index(prod.columns[0], append=False, inplace=True)

    # from now on, [0] is production 
    

    ################ get max prod
    max_prod = prod[prod.columns[0]].max(skipna=True) # by default columns  axis=1 by row
    print('max production after outliers', max_prod)
    min_prod = prod[prod.columns[0]].min(skipna=True)
    print('min production after outliers', min_prod)



    ############### histograms
    # histogram and max prod before normalization
    # just to get a sense of the distribution shift before and after normalization
    # range is relative to THIS csv
    # different than config.prod_bins "prod_bins":[0, 6.9, 15.7, 21.9, 1000000], used for ENTIERE dataset
    # need to use list and not ndarray

    ############## histogram before pannel normalization, ie take care of incremental pannel installation
    # number of occurence in each bin
    # FIXED WIDTH: The bins are equal width and determined by the arguments value_range and nbins
    
    # get value (prod) 
    values_1 = prod[prod.columns[-1]].to_list()  # if str(e) != 'nan'] # e == np.nan is False for e = np.nan 
    values_1 = [e for e in values_1 if str(e) != 'nan']

    # compute value range is [x,y], nbins is int


    ##### DO NOT USE prod_bins at this stage to configure histograms
    # v_range = [0.0, max(config.config["prod_bins"])+1] 
    # "prod_bins":[0, 6.9, 15.7, 21.9, 1000000],
    # this leads to all sample in first bin, as equal width and upper bound 100000 . tf.Tensor([219   0   0   0], shape=(4,), dtype=int32)
    # nbins = len(config.config["prod_bins"]) -1

    # rather set the range with maximum value for THIS csv, and choose some nb of bins

    #### the idea is (was) analyze distribution shift before and after pannel normalization
    # range related to only to the current csv (not entiere dataset)
    # WARNING: max_prod may increase after pannel normalization as it applies a factor >1
    # so the histograms will not be the same after and before !!

    # make it simple. only show histograms after normalization

    ### NOTE: histograms for ENTIERE dataset is done elsewehere

    ############# normalize production for incremental solar panels installation
    # cannot use lambda as I need index to operate on value
    nb = prod.shape[0] # nb of row , ie of production value

    for index, row in prod.iterrows(): #   for range in nb and iloc seems to mess around with index

        # index is Timestamp('2023-04-19 00:00:00')
        # row <class 'pandas.core.series.Series'>  (1,) , row[0] is value

        assert prod.loc[index] [0] == row[0] 

        n_energy, maxi = normalize_energy(row[0],index)

        # Similar to loc, in that both provide label-based lookups. Use at if you only need to get or set a single value in a DataFrame or Series.
        #prod.at[index,prod.columns[1]] = n_power

        prod.loc[index] [0] = n_energy

    print('max prod after normalization %0.2f applied factor: %0.1f date: %s' %(maxi[0] , maxi[0]/maxi[1], maxi[2])) # prod, ratio, date
    # max prod after normalization 34.07 applied factor: 1.7 date: 2021-06-16 00:00:00  06/16/2021,"20,050"
    # 20KWh with 4x355 + 4x360
    max_prod = prod[prod.columns[0]].max(skipna=True) # by default columns  axis=1 by row
    #print('max prod after after normalization %0.2f'  %max_prod)
    assert max_prod == maxi[0]

    v_range = [0.0, max_prod+1]
    nbins = 4  # fixed width bins, does not enforce same number of sample per bins


    ############## histogram after pannel normalization
    # WARNING: fixed width
    
    # tf.histogram: need to use list and not ndarray
    values_2 = prod[prod.columns[-1]].to_list()  # if str(e) != 'nan'] # e == np.nan is False for e = np.nan 
    values_2 = [e for e in values_2 if str(e) != 'nan']

    """
    # histogram, in which bucket is the value ?
    #  array with len = len(values), content = indices 0 to .. bin-1, ie in which bucket is the value
    # indices of bin for every data in values, 
    indices = tf.histogram_fixed_width_bins(values, v_range, nbins=nbins) # returns A 1-D Tensor holding histogram of values.
    indices = indices.numpy() 
    assert len(indices) == len(values)  # 
    """

    # number of occurence in each bin
    histo = tf.histogram_fixed_width(values_2, v_range, nbins=nbins)

    assert len(histo) == nbins
    assert histo.numpy().sum() == len(values_2)

    print("production histogram after normalization (%d fixed width bins, ie do NOT enforce same number per bins): %s" %(nbins, histo.numpy()))

    # plot histograms
    if plot_histograms:
        various_plots.histograms(values_1, values_2, nbins, "production histograms (fixed width)")

    ################ final check 
    assert prod.isnull().sum().all() == 0

    print("==> solar cleaned")

    return(prod, max_prod)


####################
# quentile for energy
####################

def solar_quentile_from_df(energy:pd.DataFrame):

    # from download csv or from df_model
    # both should be very similar to distribution using dataset, ie build_prod_bins_solar_histogram_from_ds()

    max_prod = energy.max(skipna=True)
    min_prod = energy.min(skipna=True)
    median_prod = energy.median() # 16.8
    mean_prod = energy.mean() # 15.2

    print("Solar production: max: %0.1f, min: %0.1f, mean: %0.1f, median %0.1f" %(max_prod, min_prod, mean_prod, median_prod))
 
    x = [.25, .5, .75] # is quartile
    quartile = energy.quantile(x) # serie
    #0.25     8.409500
    #0.50    16.681959
    #0.75    22.636876
    print("energy quartile (4 bins):\n" , quartile.to_list())

    x = 0.5
    quantile_median = energy.quantile(x)
    assert quantile_median == median_prod

    x = [.33, .66] 
    quartile = energy.quantile(x) # serie
    print("energy quartile (3 bins):\n" , quartile.to_list())



#####################################
# analyze production dataframe
# only called when building initial features
#####################################

def plot_downloaded_solar(prod:pd.DataFrame):

    # use df_prod (will use date index to add month)
    # print quartile
    # plot histogram
    # add month and plot hex bin production/month

    energy = prod[prod.columns[0]]  

    solar_quentile_from_df(energy)

    ###################
    # plot
    ###################

    ###### box plot quartile, ie box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2)
    various_plots.pd_box(energy, title= "solar production quartile")
    
    ####### histograms
    various_plots.pd_histogram(energy, bins= 4, title ="solar production histogram")
    
    ####### hexagonal histograms
    # need to add month , to plot production vs month
    # .apply() is for columns or row , not index. use serie.map
    m =energy.index.map(lambda d: int(d.strftime('%m'))) # get month serie

    #m = energy.index.apply(lambda d: int(d.strftime('%m'))) # extract month serie

    energy_copy = energy.copy(deep=True)  # independant copy
    df= energy_copy.to_frame() # cannot insert on serie, so convert to frame
    _ = df.insert(1,"month", m , False ) # in place    cannot use -1. 1 is after 1st columns (ie columns 0)         
    #2021-03-11              0.260062      3

    # x, y, c
    various_plots.pd_hex_histogram(df, df.columns[0], "month", c=None, title='production vs month')



###########################################
# build production histograms from sequence dataset (assume distribution for sequence and csv is the same)
# call PRIOR TRAINING ONLY
# mostly as reference. 
#  Check that if bins configured using quartile, histogram should be balanced (same number of samples) 
#  also compute majority class % and onehot. should be 1/nbins if dataset balanced
###########################################

def build_solar_histogram_from_ds_prior_training(ds, prod_bins:list) -> Tuple[list, float, list]:

    print("\nbuild solar production histograms from sequence dataset. validate if balanced using current bin setting")
    # prod_bins [0, 7.83, 17.37, 25.14, 1000000]
    nb_bins = len(prod_bins) - 1

    batch_size, _ = pabou.get_ds_batch_size(ds)
    categorical  = pabou.is_categorical_ds(ds) 

    # This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`.  You should use `dataset.take(k).cache().repeat()` instead.

    # if bins befined with solar quartile, the below histograms should be even, ie array([153, 146, 159, 160]
     
    # use below to only go once thru the loop TO get all labels in one go, l = ALL LABELS
    for _, l in ds.unbatch().batch(len(ds)*batch_size):  # ask for 7568, returns what is in dataset, ie 7554
        
        #assert len(l) == len(train_ds) * batch_size # NOT TRUE, last batch not complete 
        # l  TensorShape([8656, 4]) 
        # TensorShape([7937]) for regression

        if categorical:

            # labels one hot, converted to int, need numeric tensor for histograms
            # l.shape[0], aka len(l) (ie number of labels) should be equal to number of sequences in data set
            assert np.sum(l) == len(l) # for one hot

            value = np.argmax(l, axis=1)  # array of int (index) # (8656,)  array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
            assert value.shape[0] == l.shape[0] # same number
            assert value.ndim == 1 

            # value is an array of 0,1,2
            # A simple way: use list.count to get number of 1, of 2 etc ..
            l = value.tolist()
            h = [l.count(i) for i in range(max(l)+1)]
  
            # tf.histogram_fixed_width: 
            # NOTE: EQUAL WIDTH BINS 
            # counting the number of entries in values that fell into every bin.
            # The bins are equal width and determined by the arguments value_range and nbins
            # value range values <= value_range[0] will be mapped to hist[0], values >= value_range[1] will be mapped to hist[-1].

            # another more convoluted way, tf.histogram_fixed_width
            # value range [0,3], nbins = 4
            # hist[0] is for less or equal than 0
            # hist[-1] is for larger or equal to 3
            # hist[-1] = hist[3] as they are 4 bins 
            histo_prod = tf.histogram_fixed_width(value, value_range=[0,nb_bins], nbins=nb_bins)
            assert histo_prod.shape[0] == nb_bins 
             # <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1259, 1258, 1270, 1264])>
            # return tensor of len nbins , ie number of sample in each category
            # bin width adjusted but FIXED WIDTH

            # NOTE: another histo, ie tf.histogram_fixed_width_bins returns INDICES (between 0 and nbins), same len as value (vs same len as bins)

            histo_prod = histo_prod.numpy()

            assert h == histo_prod.tolist()
        
        else:
            value = l # l are scalar

            # tf.histogram_fixed_width. FIXED witdh, not using quartile

            # np.histograms: bins can be scalar (equal bins) or list of edges, for non uniform bin width
            # range: The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored
            histo_prod, _ = np.histogram(value, bins=prod_bins, range=None, density=None, weights=None)

        assert histo_prod.sum() == len(l) # all labels are there  histo_prod is in number of samples
        assert histo_prod.shape[0] == nb_bins # TensorShape([4])
        assert histo_prod.ndim == 1

    print('histogram of solar productions in number of samples (sequences)' , histo_prod.tolist()) 
    

    majority_index =  np.argmax(histo_prod)

    # onehot version of majority class needed for baseline which returns constant value (ie the majority)
    # NOTE: if dataset is balanced, the majority is meaningless 
    majority_y_one_hot = tf.one_hot(majority_index, nb_bins).numpy() 

    print('solar majority class: index: %d, y onehot value: %s. NOTE: meaningless is dataset balanced' %(majority_index, majority_y_one_hot))
 
    # trained model should beat a naive model with uses random
    # if dataset balanced, random is majority class %, which is also other class % and also 1/nbins

    majority_class_percent  = histo_prod[majority_index] / len(l)

    assert len(l) == histo_prod.sum()

    print('\n>>>>>> BEAT THIS: majority class %0.2f%% of total (%d bins)' %(majority_class_percent, nb_bins))
    
    return(histo_prod, round(majority_class_percent,2), majority_y_one_hot)



