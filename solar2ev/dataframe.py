#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import time

import config_model
import meteo # wind info is define there
import enphase_API_v4
import enphase
import shared

retain = config_model.retain
header_csv = meteo.header_csv 


# initial
#meteo.csv (initial scrapping) + system_energy.csv (download) => features_input.csv

# delta
#delta_meteo.csv (API) + delta_energy.csv (API) => features_input_untrained.csv (concat)

# merge
#features_input.csv + feature_input_untrained.csv  => feature_input.csv



"""
meteo csv format:
date,       hour,   temp,    humid,  direction   ,wind,  pressure
2022-11-24, 0,       0.7,    96%,    Nord,        0,      1018.8

solar format:
solar start on 11th. first day partial. no " seems to implies wh 
Date/Time,Energy Produced (Wh)
03/11/2021,92
03/12/2021,"4,233"

"""
 
################################
# GOAL: a bunch of stats on dataframe
################################

def df_info(df, nb=25, s="info on dataframe"):
    print(s)
    df.info() 
    print(df.describe())
    print(df.head(nb))
    print(df.tail(nb))

################################
# GOAL: create input feature df by combining (cleaned) meteo and solar df
#  ie add production, month, cos
# used in unseen, inference and initial feature input building
################################
def create_feature_df_from_df(df_meteo:pd.DataFrame, df_prod:pd.DataFrame, c_name) -> pd.DataFrame:

    # use df_meteo as base to create aggregate feature dataframe (with solar production)
    # add production columns (name as argument)
    # query solar for each date in meteo
    # if solar data does not exist, solar = nan (ie meteo is more up to date). should not happen as asserting len
    # set date format

    # return input feature dataframe (modified version of meteo)

    #t = df_meteo.iloc[1000] [0] # one date in meteo table
    #solar= df_prod.loc[t].values[0] # class 'numpy.float64  CAN use date as index into solar table to get daily production

    # check prod is up to date vs meteo, and starting date correspond
    assert len(df_prod) >= len(df_meteo) / 24 , "not enough production values %d for meteo %d" %(len(df_prod) ,len(df_meteo)/24)
    start_date_meteo = df_meteo.iloc[0] ["date"] # Timestamp('2021-03-11 00:00:00')
    start_date_solar = df_prod.index[0]  # Timestamp('2021-03-11 00:00:00')

    assert start_date_solar == start_date_meteo, "meteo and solar do not start at same date"

    ######## create production columns
    # add empty columns to meteo df add at the end. use df.insert to specify position
    df_meteo[c_name] = -1 

    # fill production colums in meteo table with corresponding solar production for same day
    # len(retain) row will have same production value
    def get_power(t):  # make sure index of df_prod is timestamp. use meteo 1st colums (timestamp) as index into production table
        try:
            prod = df_prod.loc[t].values[0] # or df_prod.loc[t] [0]
            return round(float(prod),2)
        except Exception as e:
            return np.nan 

    # fill new columns with production data , use meteo timestamp as index into production table
    df_meteo[c_name] = df_meteo[df_meteo.columns[0]].apply(get_power) # input timestamp, output production

    ####### convert date
    # convert date (timestamp) to known format string 
    df_meteo['date'] = df_meteo['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    ######## check
    # in case meteo file is more recent than downloaded solar production, exclude row without production
    # in theory, meteo file should be more recent, as it is easier to create (scrapping, vs manual download)
    # cut table to include only initialized power (in case)
    df_meteo = df_meteo[df_meteo[c_name] != -1]

    # if meteo file is more recent than solar file, production is nan
    filter =  pd.notnull(df_meteo[c_name]) 
    df_meteo = df_meteo[filter]

    
    # is.null() is array of bool
    # isnull().sum() is a serie , nb of null per columns
    assert df_meteo.isnull().sum()[df_meteo.columns[1]] == 0 # no temp missing
    assert df_meteo.isnull().sum()[df_meteo.columns[-1]] == 0 # no solar missing

    assert df_meteo.isnull().sum().sum() == 0 # sum of all null is 0 , ie no null anywhere 
    assert df_meteo.isnull().sum().all() == 0 # each colums has zero null

    ############### set index to date column
    # WHY ???.  return(df_meteo) is df_model, which has date as a 1st columns (an not an index)
    #df_meteo.set_index(df_meteo.columns[0], append=False, inplace=True)

    return(df_meteo) 


###############################
# assert all timing data is OK, ie scrapping did its job 
# use on meteo_df or df_model (with production/month). automatically detected
# if config_model.assert_timing, deep check on timing 
###############################
def assert_df_timing(df: pd.DataFrame, s=""):

    # detect if meteo df or feature df
    try:
        df["production"]
        check_meteo_only = False
    except:
        check_meteo_only = True # (do not check month and production, which are only available on feature)


    print("assert timing for: %s. check_meteo_only: %s. deep dive: %s" % (s, check_meteo_only, config_model.assert_timing))

    # date converted from str to datetime, IF NOT ALREADY THE CASE

    # complete days of 24 hours
    assert len(df) % 24 == 0
    nb = len(df) / 24
    print("timing for: %d days" % nb)

    # month columns well formed. only available on feature df
    if not check_meteo_only:
        if df.nunique()["month"] != 12: # only available on feature, not meteo
            print("!!!!! WARNING: df do not contains all 12 months. This is normal for delta df, or inference df")


    # as many dates as number of days
    assert df.nunique()["date"] == nb 

    # first and last dates are consistent with number of days
    start_date = df.iloc[0] ["date"]
    end_date = df.iloc[-1] ["date"]

    #first_day = df_model.iat[0,0]
    #last_day = df_model.iat[-1,0]

    # needed if need to convert from str to datetime
    if check_meteo_only:
        s_date = "%Y-%m-%d %H:%M:%S" # gosh, I need to clean this at some point.
    else:
        s_date = "%Y-%m-%d"

    # be extra carefull. 
    if isinstance(start_date, str):

        # convert dates from str to datetime
        # NOTE: will fail if date already a datetime
        start_date = datetime.datetime.strptime(start_date, s_date )
        end_date = datetime.datetime.strptime(end_date, s_date)

    assert isinstance(start_date, datetime.datetime)
    assert isinstance(end_date, datetime.datetime)

    print("check timing from: %s to: %s" %(start_date, end_date))

    # intervals  end_date - start_date is a datetime.timedelta
    assert (end_date - start_date).days == nb -1 

    # check hour value. 1st day and last day , ie First 24 and last 24 rows
    # hour of 1st day is are 0 to 23

    # watch for hour could be int or str

    nb = 24
    for x in range(nb): # 1st 24 rows
        #assert df_model.iat[x,1] == x
        assert df.iloc[x] ["hour"] == x 
    
    # hour of last day is 23 to 0
    for x in range(nb): # last 24 rows
        assert df.iloc[-1-x] ["hour"] == retain[x]
        #assert df_model.iat[-1-x,5] == retain[x] 


    ##########################
    # deep check of timing in df_model
    # set to True in config file if paranoiac, or just got a new set of data. False speed loading a bit
    #########################

    if config_model.assert_timing:
        # go thru all rows (ie hour)
        # days are every 24th index
        # days's date increment by 1
        # new days's date have not been seen yet
        # days contains all 24 hours
        # date in same day is the same
        # production in same day is the same 

        current = start_date +  datetime.timedelta(-1)
        l_date = [] # all date seen so far
        l_hours = []

        for i in range(len(df)):

            # date columns
            date = df.iloc[i] ["date"]
            
            if isinstance(date, str):
                date = datetime.datetime.strptime(date, s_date)

            if i % 24 == 0:
                ##########################################
                # start of a new day, ie every 24th entry
                #########################################

                # every 24th entries are one day apart from previous day
                assert date == current + datetime.timedelta(1)

                # day starts with hour 0
                assert df.iloc[i] ["hour"] == 0

                current = date

                # make sure date was never seen
                assert date not in l_date

                l_date.append(date) # list of all date seen so far

                hour = df.iloc[i] ["hour"]
                l_hours.append(hour) # list of all hours seen so far for one day. this one is 0

                if not check_meteo_only:
                    prod = df.iloc[i] ["production"] # get daily production for future check

            else:
                ##################
                # ongoing day
                ##################

                # all hours 1 to 23 have same (current) date
                assert date == current , "df_model date is not the same as the one seen at start of day (ie every 24 rows) ERROR at index %d\n dataframe around error %s"%(i, df.iloc[i-30:i+30][["date", "hour"]])

                # get all hours to make sure none is missing
                hour = df.iloc[i] ["hour"]
                l_hours.append(hour)

                if not check_meteo_only:
                    assert prod == df.iloc[i] ["production"] , "df_model production ERROR at index %d\n %s "%(i, df.iloc[i-10:i+10][["date", "hour"]])

                # last hours
                # check we got all hours 
                if hour == 23:
                    assert len(l_hours) == 24, "df_model hour ERROR at index %d\n %s "%(i, df.iloc[i-10:i+10][["date", "hour"]])
                    assert len(list(set(l_hours))) == 24, "df_model hour ERROR at index %d\n %s "%(i, i, df.iloc[i-10:i+10][["date", "hour"]])

                    l_hours=[]




#################
# GOAL: check integrity of ONE feature input dataframe. 
# columns FOR DF_MODEL, ie feature input.
# timing
# other tdb
#    timing's logic applies to both meteo only and df_model (with production, month)
#################
def assert_df_model_integrity(df1:pd.DataFrame, s = ""):

    print("assert df model integrity for:", s)

    # CHECK 1: dataframe has same columns as model
    # get reference colums list from shared variable

    assert (list(df1.columns)  == shared.feature_input_columns) , "df_model_integrity. columns list does not match"

    # CHECK 2: core timing
    # can do deep dive on timing based on config_model.assert_timing 
    assert_df_timing(df1, s=s) # including production

    # CHECK 3: TBD

    # BELOW is for print, not for assert

    # convert date column from str to dates
    df1[df1.columns[0]] =  pd.to_datetime(df1[df1.columns[0]], format=config_model.format_date)

    first_day = df1.iat[0,0]
    last_day = df1.iat[-1,0]

    # compute in years and months
    # no assert, just display
    nb_days1= len(df1) / len(retain)

    # complete years
    nb_years = int(nb_days1 / 365)

    # rest in fractional month
    r = nb_days1 % 365 # days after complete years
    r = r /30 # in fractional monts

    print("integrity check: %d days, from %s to %s. %d years, %0.1f months" %(nb_days1, first_day.date(), last_day.date(), nb_years, r))


#########################
# GOAL: get last(first) date in csv
#########################

def get_last_cvs_date(csv, mode=-1) -> datetime.datetime:
    # last date is default, mode =-1
    # use mode = 0 to get first date
    # date is assumed to be 1st columns and in date (not timestamp) format

    # assume df_model NOT stored with index colums
    # so date is 1st columns

    try:
        df = pd.read_csv(csv)
        
        #FutureWarning: Series.__getitem__ treating keys as positions is deprecated. 
        #In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). 
        #To access a value by position, use `ser.iloc[pos]`

        last_seen_date= df.iloc[mode][0] # str '2022-11-23'
        # convert from str to datetime
        last_seen_date = datetime.datetime.strptime(last_seen_date, "%Y-%m-%d")
        print('last existing feature date: ', last_seen_date )
        return(last_seen_date)

    except Exception as e:
        print("!!!!!!!!!!!!!!!cannot open or convert %s. exception: %s" %(csv, str(e)))
        return (None)
    

#################
# GOAL: can be concatenated (ie first day of df2 = last day of df1 + 1)
# NOTE: integrity check need to done before
################

def assert_df_model_concatenate(df1:pd.DataFrame, df2=pd.DataFrame, s=""):

    # can be concatenated ?

    # convert date from str to dates to do comparison
    df1[df1.columns[0]] =  pd.to_datetime(df1[df1.columns[0]], format=config_model.format_date)
    df2[df2.columns[0]] =  pd.to_datetime(df2[df2.columns[0]], format=config_model.format_date)

    # check last date of df1 is first date of df2 -1, ie can be concatenated
    last_day = df1.iat[-1,0]
    first_day = df2.iat[0,0]

    print("can be concatenated ?: df1 last entry %s, df2 first entry %s" %(str(last_day), str(first_day)))

    assert first_day - last_day == datetime.timedelta(1) ,"dataframe cannot be concatenated, df1 end date and df2 start date do not match"

    print("dataframes can be concatenated timewize")




# convert wind direction str to index into list
def direction_to_index(direction_str:str) -> int:
    # index of str in direction list
    return(meteo.direction.index(direction_str))


# map cyclical value
# https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c
# https://www.sefidian.com/2021/03/26/handling-cyclical-features-such-as-hours-in-a-day-for-machine-learning-pipelines-with-python-example/
# sin( (2PI * hour) / max_hour)

# 12h clock, ie 0 = 12 
# max is number of discrete different features, eg 0 to 11 = 12 features

# max(a) is 12 as you can’t have a number higher than 12 on your typical clock. 
# It’s important to note that 12:01 am is the same as 00:01, which needs to be taken into account with other cyclical features. 
# If max(a) isn’t the same as 0, then add 1 to the max (see below with the wind example).

# wind If we number the features 0 to 7
# sin( (2PI * index) / 8)

# need BOTH cos and sin, as cos sin(0) = sin(pi) = sin(2pi), ie jan = july. combining cos and sin deambiguate


def cyclical_value_to_trig(value, max, trig='sin'):

    n = 2*np.pi * value / max

    if trig == 'sin':
        return( round(np.sin(n),2))
    else:
        return(round(np.cos(n),2))
    # return one value to apply lambda to one column

# NOTE: could do without lamdba
# instead of hours extending from 0 to 23, we have two new features “hr_sin” and “hr_cos” which each extend from 0 to 
#df['hr_sin'] = np.sin(df.hr*(2.*np.pi/24)) 

#df['mnth_sin'] = np.sin((df.mnth-1)*(2.*np.pi/12))
#last_week['Sin_Hour'] = np.sin(2 * np.pi * last_week['Hour'] / max(last_week['Hour']))


###################
# GOAL: generic df cleaning 
# common to inference and training
###################

def generic_meteo_df_processing(df: pd.DataFrame, retain=config_model.retain) -> pd.DataFrame:

    # called by build_meteo_df_from_csv()

    # remove %
    # subsample (not used)
    # insert columns month, sin(months), sin(hour)
    # convert wind direction from str to int
    # returns df


    ### apply functions to remove % 
    def remove_percent(x):
        #print(x, type(x)) # 94% <class 'str'>
        #if x != np.nan: # But np.nan == np.nan still gives False
        if str(x) != 'nan': 
            return(int(x.replace('%', '')))
        else:
            return(x)

    df['humid'] = df['humid'].apply(remove_percent)

    ############### subsample hours with filter . NOT USED ANYMORE
    # so you should use "bitwise" | (or) or & (and) operations:
    # df['hour'][0] '0' convert hours to int
    try:
        d = df['hour'].astype(int, copy=True, errors = 'raise') # copy = True or will propagate
        filter = d.isin (retain) # serie of boolean
        df = df[filter]
    except Exception as e:
        pass

    ###################################
    # cyclical values
    ###################################

    # such as month, hours, wind direction, converted to trig 

    #############
    # wind direction from str to int (index)
    # then to sin/cos
    ############

    df["direction"] = df["direction"].apply (lambda x:direction_to_index(x))

    max_index = len(meteo.direction)-1
    l = df["direction"].to_list()

    for i in range(max_index):
        print("wind direction index %d, count %d" %(i, l.count(i) ))


    # now an integer in range len(direction)

    #0 or ‘index’, 1 or ‘columns’}, default 0 
    # args additional position argument  , tuple
   
    # 0 or ‘index’: apply function to each column.
    #1 or ‘columns’: apply function to each row.

    print("mapping direction index (%d discrete values) to trig" %len(meteo.direction)) # ie 4x4 +2 

    #  If max(a) isn’t the same as 0, then add 1 to the max

    # wind direction, 
    _max = len(meteo.direction) + 1 # nb of discrete +1
    # add at the end
    df["direction_sin"] = df["direction"].apply(cyclical_value_to_trig, args=(_max, 'sin')) # args is max (12 for 0 to 12 with 12 =0, +1 for wind)
    df["direction_cos"] = df["direction"].apply(cyclical_value_to_trig, args=(_max, 'cos'))

    # NOTE: can do without lamdba. see below
    
    for d in [df["direction_sin"], df["direction_cos"]]:
        print("wind direction : max: %0.1f, min %0.1f mean %0.1f std %0.1f median %0.1f"  %(d.max(), d.min(), d.mean(), d.std(), d.median()))

    # insert as columns 1. apply function to columns 0 , ie date as datetime
    df.insert(1,"month", df[df.columns[0]].apply(lambda x: int(x.strftime('%m'))), False ) # do not allow duplicate, in place (returns none)


    # month, represented as 1 to 12
    # max = 12 is the same as 0 , jan = 1 is close
    _max = 12 #  nb of different features, 

    # Insertion index. Must verify 0 <= loc <= len(columns).
    df.insert(len(df.columns), "sin_month", df["month"].apply(cyclical_value_to_trig, args=(_max, 'sin')))
    df['cos_month'] = round(np.cos(df["month"]*(2.*np.pi/_max)),2)


    # hours, represented as 0 to 23
    # 23h is not the same a 0
    _max = 24
    df.insert(len(df.columns), "sin_hour", df["hour"].apply(cyclical_value_to_trig, args=(_max,)))
    df['cos_hour'] = round(np.cos(df["hour"]*(2.*np.pi/_max)),2)
    #df.insert(2, "cos_hour", df["hour"].apply(cyclical_value_to_sin, args=(24, "cos" )))

    print('df meteo resulting colums:' , df.columns)

    return(df)




################################
# GOAL: clean meteo csv
# use for scrapping and running inferences 
# call generic_meteo_df_processing() 
################################

def clean_meteo_df_from_csv(meteo_file:str) -> pd.DataFrame:

    retain=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    installation = enphase.installation
    # add missing data (nan)
    # returns dataframe

    #date,hour,temp,humid,direction,wind,pressure
    #2021-03-01 00:00:00,0,0.7,86%,Sud,2,1029.4

    # interpolate missing data (web scrapping returns exactly 24 hours per day, but have Nan where some measurement are missing)
    # filter date before installation date (parameter)
    # convert date to panda datetime
    # remove last day which is likely uncomplete anyway  (unless launched scrapping very very late in the day)
    #   (even if contains 24 hours, likely later hours are from previous day, because of scrapping logic)
    # check no null

    print('\nclean meteo csv: %s' %meteo_file)

    #  np uses nan  pandas uses null

    df = pd.read_csv(meteo_file)

    nb_sample = len(df) # nb of rows

    # also used to create dataframe for inference, so have a last incomplete day
    #assert len(df) % len(retain) == 0 # contains only full days, ie 24 hours
    # df[df.columns[0]] = df['date'] '2023-11-24', object (str, not datetime)
    
    print('meteo to be cleaned: %d rows (ie hour), %d days' %(nb_sample, len(df)/24)) # meteo rows 15216, in days 634

    #### Convert 'date' columns ,  Strings, to Datetime in Pandas DataFrame columns 0
    # in original scrap 2021-03-01 00:00:00
    # in /tmp/unseen_meteo.csv 2023-04-19 (could change, but I do not feel like recrapping)
    # exception is date format is set explecityly

    #df[df.columns[0]] =  pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d %H:%M:%S') 


    """ error on PI
    time data "2023-04-19" doesn't match format "%Y-%m-%d %H:%M:%S", at position 0. You might want to try:
    - passing `format` if your strings have a consistent format;
    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;
    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this."""

    # post mortem/inference date is simpler than scrapped one (bummer, this is going to annoy me forever)
    #        date  hour  temp humid        direction  wind  pressure
    #0   2024-02-02     0   1.8   95%             Nord    11    1034.7

    # using format="%Y-%m-%d %H:%M:%S" generate error on PI. seems OK for windows .. WTF
    # tried format="mixed", dayfirst=False, does not work
    # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

    # do not let this screw me. if there is a space , it's a  %Y-%m-%d %H:%M:%S, else it's a %Y-%m-%d

    if " " in df.iat[0,0]:
        df[df.columns[0]] =  pd.to_datetime(df[df.columns[0]], format="%Y-%m-%d %H:%M:%S")
    else:
        df[df.columns[0]] =  pd.to_datetime(df[df.columns[0]], format="%Y-%m-%d")

    # converted to timestamp
    #5279   2023-11-24
    #Name: date, Length: 5280, dtype: datetime64[ns]#

    #### filter row to start sample at 1st solar pannel installation
    #print('1st pannel installation ', installation)
    #t = installation - datetime.timedelta(day_past)
        
    print("only retain meteo starting at ", installation)
    t = installation

    filter = df[df.columns[0]]  >= t # make sure date column is date and not string to be able to test
    df = df[filter]

    nb_sample = len(df) # update nb sample
    # WARNING: shape has changed, line possibly removed 


    ###############################################
    # common processing for training and inference
    ###############################################
    # columns 0 is date as datetime
    # remove % ,subsample (not used)
    # insert 4 colums 
    # convert wind direction from str to int
    # WARNING: shape has changed, columns added

    df = generic_meteo_df_processing(df, retain)

    org_shape = df.shape

    #### check how many missing values (nan)

    print("check for any missing values")

    #print(df.isnull()) # data frame of boolean, not very usefull
    print('=> any null values ? sum of nulls by columns\n', df.isnull().sum()) # sum of nulls by columns
    print("=> any missing temp ?: %d" %df.isnull().sum()["temp"])
    print("=> any missing pressure ?: %d" %df.isnull().sum()["pressure"])
    print("=> any missing humid ?: %d" %df.isnull().sum()["humid"])
    
    # check pd null is the same as np nan
    filter = pd.isnull(df['temp']) #pd.isnan does not exit   pd.notnull
    filter1 = np.isnan(df['temp']) # return list of boolean

    print('==> nan and null temp: len df %d, pandas null temp: %d, numpy nan temp: %d, non null temp: %d' 
    %(len(df), len(df[filter]), len(df[filter1]), len(df[filter == False])))
    # len df 15024, len null temp 267, len nan temp 267 len non null temp 14757

    assert len(df[filter]) == len(df[filter1])

    # look also nan as numpy array
    df_np = df.to_numpy(copy=True)
    print('==> dataframe to numpy ', df_np.shape, df_np.ndim, df_np.dtype ) # numpy  (15024, 6) 2 object same with false
    print('==> null temp represented as numpy array', df[filter].to_numpy().shape) # null temp numpy (267, 6)
    print('==> null temp numpy array info\n')
    print(df[filter].info())

    #### remove nan
    # generated by scrapping future, or some missing measurement value
    # ???? assume that there are no instance of missing meteo value when temp is not missing

    ### drop entiere row / colums is not a good idea
    # this would removes line entirely
    #df = df[filter = False] # pass list of boolean, generated by filter

    # df.dropna() # drop row with at least one Nan Null value
    # df.dropna(how = 'all') # drop row whose all valuer missing or contains null value (Nan)
    # df.dropna(axis=1) # drop colums with ar least one null
    
    ### rather fill missing values LINEAR

    # first interpolate
    # !!! issue with method linear and datetime DataFrame.interpolate has issues with timezone-aware datetime64ns columns
    
    # use d = df.iloc[:,1:] to exclude 1st column (ie date) Purely integer-location based indexing for selection by position
    ### BUT changes (interpolation) on this slice does not seem to propagate to df
    # neither 
    #select = df.columns[1:]
    #d = df[select]   d.interpolate do not propagte
    #d = df[['temp', "pressure"]] # [[]]

    #d = df['temp'] #interpolation on d propagate to df

    # select serie which needs interpolation 
    #df.isnull().sum().index is the same as df.columns

    for x in df.columns: # Index(['date', 'month', 'hour', 'temp', 'humid', 'wind', 'pressure'], dtype='object'
        nb_missing = df.isnull().sum()[x]
        if nb_missing == 0 or x == 'date':
            # do not touch date colums or colums with no missing data
            pass
        
        else:
            # interpolate one serie/colums
            # could also fill with previous/next valid (what is no previous/next valid ?)

            # Fill NaN values using an interpolation method
            # ‘linear’: Ignore the index and treat the values as equally spaced

            print("need to interpolate colums %s, as %d missing measurements" %(x, nb_missing))

            # https://note.nkmk.me/en/python-pandas-interpolate/
            df[x].interpolate(method='linear', limit_direction='forward', inplace=True)
    
    # all should be interpolated
    assert df.isnull().sum().sum() == 0, "still missing values after interpolating"
    assert df.isnull().sum().all() == 0, "still missing values after interpolating"
    # add .all() ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

    # other way to fill missing value
    # df.fillna(0)

    # The method argument of fillna() can be used to replace NaN with previous/following valid values.
    # df = df.fillna(method = 'bfill')  # pad or ffill: forward fill    backfill or bfill:  backward fill

    # df["temp"].fillna(99, inplace=True)
    # df.replace(to_replace = np.nan, value = 99)


    # len df should not have changed
    assert len(df) == nb_sample
    # WARNING. shape has changed, as columns added and line possibly removed, so get original shape correctly
    assert org_shape == df.shape

    print('====> meteo cleaning and assert DONE')

    return(df)


################################################
# GOAL: create delta feature from web (scrap and enphase API)
# cleaned, combined , with missing data filled
# for unseen, retrain
################################################

def create_unseen_delta_input_df_from_web(from_date:datetime, end_date:datetime, days_go_back:int) -> pd.DataFrame:

    # creates csv and use same cleaning/merge processing as for initial scapping/download
    # normally days_go_back = -2. used to testing (avoid large scrapping)

    import utils
    tmp_dir = utils.tmp_dir

    unseen_prod_csv = os.path.join(tmp_dir, "unseen_energy.csv")
    unseen_meteo_csv = os.path.join(tmp_dir, "unseen_meteo.csv")

    # called for unseen or retrain
    # starts from_date
    # creates /tmp/..csv using scrapping and enphase API
    # clean both and combine. same processing as original scrapping

    # returns pandas dataframe
    # meant to update feature_untrained.csv
    
    print("create delta dataframe. from %s (included), to %s (included. yesterday -1)" %(from_date.date(), end_date.date()))
    
    # end date is supposed to be yesterday -1 WARNING: is time of running -1 day, 

    nb_day_unseen = (end_date - from_date).days +1 # from and end date included
    print("get %d untrained days" %nb_day_unseen)

    import config_features
    df_model =  pd.read_csv(config_features.features_input_csv)

    #################
    # scrap and API
    #################

    get_from_web = True # use with care. test with existing unseen.csv to avoid lengthy scrapping

    if get_from_web: 
        print("get unseen from web")

        # no leftover

        try:
            os.remove(unseen_prod_csv)
        except:
            pass

        try:
            os.remove(unseen_meteo_csv)
        except:
            pass


        ######create solar and meteo csv, as for initial scrapping
        # thus reuse all post processing

        #######  SOLAR ###########
        # creates unseen_prod_csv
        # same format as the one downloaded from enphase cloud
        ##########################
    
        # creates solar csv file, same format as initial scrapping
        print ("\ngetting unseen production from %s to %s into: %s, (using enphase API)" %(from_date.date(), end_date.date(), unseen_prod_csv))
        
        # get solar data using enphase API and store in csv file
        production_list = enphase_API_v4.get_daily_solar_from_date_to_date(from_date, end_date)
        assert len(production_list) == nb_day_unseen

        #Date/Time,Energy Produced (Wh)
        #03/11/2021,92
        #03/12/2021,"4,233"

        # creates production csv file, same format as enphase
        # so reuse existing cleaning, normalization, etc .. do not duplicate

        date = from_date
        list_0 = []
        list_1 = []

        for i in range(nb_day_unseen):

            list_1.append(production_list[i]/1000) #  stored as 1.000 is csv
            list_0.append(date.strftime("%m/%d/%Y"))
            date = date + datetime.timedelta(1)

        # add dummy last line. enphase report has a summary at the end, which is removed
        list_0.append(0)
        list_1.append(0)

        data = {"date":list_0, "energy":list_1}

        df_prod = pd.DataFrame(data) # use dict key as colums names. 
        assert len(df_prod) == nb_day_unseen+1

        ###########################
        # this csv should look like the one downloaded from enphase
        ###########################

        # store production as str, to be consistent with downloaded csv.
        # enphase.clean_solar_df_from_csv() manages outliers with apply(manage_watt), which expect a str and convert to float

        #date,energy
        #01/30/2024,14.051

        # float_format = str still does not force prod as str, ie is read later as float
        # manage on the reading side
    
        df_prod.to_csv(unseen_prod_csv, header=True, index=False)  # index = True creates columns 0,1,2 in csv file
        print("%s saved" %unseen_prod_csv)
        
        ###### METEO ###################
        # creates unseen_meteo_csv
        # one day at a time
        # same format as the one created from initial scrapping
        # if does not exists or not already created today (avoid uncesseray scrapping when debugging)
        ################################

        # creates meteo csv file, same format as initial scrapping
        print ("\ngetting unseen meteo from %s to %s into %s (using web scrapping)" %(from_date.date(), end_date.date(), unseen_meteo_csv))
        date = from_date

        sleep_between_scrap = 5 # sec
        sleep_before_retry = 20 # in case get timed out or error

        # debug. 
        # 2023-05-15 missing hours 0,1,2
        # 2023-05-26 missing temp, humid
        # 2 au 4 juin. all hours missing, 5 huin miss many hours
        #date = datetime.datetime(2023, 5, 25)

        days = []
        # store list of dict. each entry is one day , ie one day_result
        # element is a dict {"23",[timestamp,hour,temp,humid,wind,pressure],} 

        while date <= end_date:

            ####################
            # scrap one day
            ####################
            day_result = meteo.one_day(date, expected_hours=24) # returs dict

            if day_result != {}:
                print("scrapped: %s" %date)

            else:
                print("scapping %s failed!!. sleeping %d sec and retry (forever)" %(date, sleep_before_retry)) # do not increment date
                time.sleep(sleep_before_retry)
                break

            # day_result should have 24 hours, so ["0"] and ["24"] exists
            assert len(day_result) == 24, "day_result does not have 24h"

            for i in range(24):
                assert date == day_result[str(i)][0] , "date in day_result dict %s does not match input %s" %(date, day_result[str(i)][0])  # pandas timestamp is the same as datetime
            
            days.append(day_result)

            # days is a list of dict
            # each dict has len = 24
            # dict_keys(['23', '22', '21', '20', '19', '18', '17', '16', '15', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'])
            # each key's value is 1 hour
            # '1': [Timestamp('2024-01-30 00:00:00'), '1', '9.5', '26%', 'Ouest-Sud-Ouest', '17', '1031.3']
            
            date = date + datetime.timedelta(1)

            ################
            # good citizen. do not overflow the web site
            #time.sleep(sleep_between_scrap)
            time.sleep(1)
            ################

        # all days scrapped

        assert len(days) == nb_day_unseen 

        # type(day_result["0"][0])) # python type is  <class 'pandas._libs.tslibs.timestamps.Timestamp'>
        # but type is not preserved in csv. when reading csv, date will be seen a O, unless converted to datetime

        # convert list of dict (one per day) to list of list (one per hour), to be used to create panda dataframe (and save as csv)

        hours = [] # list of hours. each hour is a list [Timestamp('2022-11-2...00:00:00'), '0', '0.7', '96%', 'Nord', '0', '1018.8']
        
        
        for day in days: # all dicts, ie all days
            assert len(day) == 24 

            for h in range(24): # all hours in a day

                r = day[str(h)] # one row per hour
                assert len(r) == len(header_csv)

                hours.append(r)

        assert len(hours) == len(days) * len(retain)
        assert len(hours) / len(retain) == nb_day_unseen
        assert len(hours) % len(retain) == 0

        # check last day in delta meteo is yesterday -1
        assert days[-1]["0"][0].day == (datetime.datetime.now() + datetime.timedelta(days_go_back)).day
        assert hours[-1] [0].day == (datetime.datetime.now() + datetime.timedelta(days_go_back)).day

        # create meteo dataframe from list and save to csv
        df_meteo = pd.DataFrame(hours, columns = header_csv)  # format same as after initial scappping

        df_meteo.to_csv(unseen_meteo_csv, header=True, index=False)
        print("%s saved" %unseen_meteo_csv)

        # issues in assert below with hours str. check later after cleaning. because problem not there when reading from csv
        # assert_df_timing(df_meteo, s="scrapped meteo (unseen or retrain)", check_meteo_only=True)

    else:
        print ("\n!!!!WARNING!!!!! using existing unseen meteo and production csv. TESTING ONLY")
    

    ##########################################
    # clean and combine
    # reuse processing for initial scrapping (starting from csv)
    # start from csv files just created
    ##########################################


    ###################
    # clean meteo
    ###################
    # create CLEAN Pandas dataframe from meteo csv
    # incl fill missing data
        

    df_meteo = clean_meteo_df_from_csv(unseen_meteo_csv) # subsampled

    assert_df_timing(df_meteo, s="scrapped and cleaned meteo (unseen or retrain)")


    ###################
    # clean production
    ###################
    # create CLEAN Pandas dataframe from solar production csv
    df_prod, max_prod  = enphase.clean_solar_df_from_csv(unseen_prod_csv,plot_histograms=False)  

    assert len(df_meteo)/24 == len(df_prod), "len of meteo and prod df created by unseen do no match"

    # warning: if scrap and enphase API not done the same day, different len (should not happen, unless debugging)
    
    ##################
    # combine meteo and production to create df_delta
    ##################
    # COMBINE meteo and solar production, to create input features
    df_delta = create_feature_df_from_df(df_meteo, df_prod, "production") # associate solar production to meteo data (same day, all day's row have same production)
    assert len(df_delta) == len(df_meteo)

    # integrity check done in caller, easier to read

    # ???? do not check can be concatenated to df_model timewize (unseen and inference do not garentee this)

    # df_delta will be concatenated to df_model. df_model when read from csv file has date as str
    # df.to_csv(filename, date_format='%Y%m%d')

    return df_delta


"""
########## FFT
def fft(feature_df):
    fft = tf.signal.rfft(feature_df) # compute capability: 5.0
    f_per_dataset = np.arange(0, len(fft))  # fft 1235

    nb_samples = len(feature_df)
    print('samples ', nb_samples)

    hourly_samples_per_year = len(retain)*365.2524 # 1461
    years_per_dataset = nb_samples/(hourly_samples_per_year) # 1.68
    print('hourly samples per year %0.1f, nb years per dataset %0.1f' %(hourly_samples_per_year, years_per_dataset))

    f_per_year = f_per_dataset/years_per_dataset # 

    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 10000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')
    plt.show()
"""

if __name__ == "__main__":
    print("testing cyclical values")

    #def cyclical_value_to_sin(value, period, trig='sin'):

    for m in range(1,13):
        print("month: %d, %0.1f" %(m, cyclical_value_to_trig(m,12)))

    for m in range(24):
        print("hour: %d, %0.1f" %(m, cyclical_value_to_trig(m,24)))
