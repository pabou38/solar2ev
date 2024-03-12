#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#################################
# application specific
# called by whoever needs it (plot_input_features(), plot_training_results())
# calls pabou.various_plots.py, which is designed to be generic
#  fig saves in "plot" dir by default (set in various_plots.py)
################################

# plt: 9 images, scatter, histogram, correlation/heatmap, scatter 2D, 
# PANDAS:  box, histogram, hex bin, line
# seaborn: confusion, join (3D histo), pair (correlation), box, violin, histo/kde
# calplot: heatmap 

import sys
import os

sys.path.insert(1, '../PABOU') # this is in above working dir 

try:
    import various_plots
    import pabou
except:
    print("%s: cannot import from PABOU" %__name__)
    sys.exit(1)

import model_solar
import train_and_assess
import config_model

save_fig_dir = various_plots.save_fig_dir


###########################
# input features, aka training data
# data analysis
###########################
    
# save fig in dir: various_plots.save_fig_dir
# title used as file name to save
#  Suptitle is on top, centered suptitle to the figure. 
#  Title
 
# uses matplotlib, pandas, sns, calplot

def plot_input_features(df_model):

    print("start plotting input features in many ways")

    ###############
    # calplot
    ###############

    x = df_model[["date", "production"]]

    # 24 hours per days, same production, same date

    assert len(df_model) % 24 == 0
    nb_days = len(df_model) / 24

    d = []
    for i in range(len(df_model)):
        if i % 24 == 0:
            d.append(i)

    assert len(d) == nb_days

    values = x.iloc[d]

    # convert to series with date as index , ie values.index DateTimeindex
  
    values["date"] = pd.to_datetime(values["date"]) # not needed (done already). see values.dtypes   date datetime64[ns] production  float64
    # values.index dtype int64 (the original index)

    values = values.set_index('date')
    # still a dataframe with Index and 1 columns type(values)

    values = values.squeeze()
    assert max(values) == values.max()

    """
    days = pd.date_range('01/01/2020', periods = 1277, freq = 'D') # Sequence of dates
    values = pd.Series(np.random.randn(len(days)), index = days) # Pandas time series with random data for each day
    #values.index DateTimeIndex
    #values.columns  does not exist
    # values
    #2020-01-01   -0.331121
    #2020-01-02   -0.561178
    """
    
    
    # serie Must be indexed by a DatetimeIndex.
    various_plots.calplot_heatmap(values,  title= "production Kwh per day")
 

    ###########
    # line(s) , no subplots
    ###########

    # 1 line per plot
    # X is samples

    # each sample is a different measurement
    various_plots.pd_line(df_model["pressure"], title="pressure",xlabel="samples", ylabel="pressure Hpa") # 
    various_plots.pd_line(df_model["temp"], title="temp", xlabel="samples", ylabel="temperature C")

    # not very good all prod for same day the same
    various_plots.pd_line(df_model["production"], title="production", xlabel="samples", ylabel="production Kwh") # figure 2 is displayed 


    # 2 lines on same plot
    # if unscaled data, better use similar range , eg temp and prod are both in the 10, 30
    various_plots.pd_line(df_model[["production", "temp"]], title="prod and temp", xlabel="samples", ylabel="production, temp")

    # doing the same for unscaled production pressure does not work
    d = df_model[["production", "pressure"]]
    for column in d: 
        d[column] = (d[column]-d[column].mean()) / d[column].std()
      
    various_plots.pd_line(d, title="prod and pressure (normalized)", xlabel="samples", ylabel="production, pressure (normalized)")

    ###########
    # lines, subplots
    ###########
    # subplot, include multiple plots (of 1 lines) in same figure
    various_plots.pd_subplot_lines(df_model[["production", "pressure", "temp", "humid", "wind"]], title="overview")

    ##############
    # scatter 
    # using plt
    ############## 

    y = df_model["production"].to_numpy().astype('float')
    x = df_model["pressure"].to_numpy().astype('float')

    # can add a 3rd dim, ie different size, color based on z
    z = df_model[df_model.columns[-1]].to_numpy().astype('float') # production

    # can use serie for marker size and color, ie each dot will have a different size and color (using cmap)
    # if marker size = none, or s= 30, all dot same size, otherwize , larger value = larger dots
    # can have single color c="blue"
    # can specify shape, marker="d"
    
    #############################
    # WARNING: for a few plot below, z not used (marker size ie s, and color list ie c, are constant)
    #############################

    # scatter pressure vs production 
    _= various_plots.scatter(x,y, marker_size = 10, color_list = "blue",
    title='production vs pressure', axis_text ='solar production as function of pressure', ylabel="production Kwh", xlabel="pressure")


    # scatter wind direction vs production
    x = df_model["direction"].to_numpy().astype('float')
    _= various_plots.scatter(x,y, marker_size = None, color_list = None,
    title='production vs wind direction', axis_text = 'solar production as function of wind direction', ylabel="production Kwh", xlabel="wind direction")

    # scatter temp vs production
    x = df_model["temp"].to_numpy().astype('float')
    _= various_plots.scatter(x,y, marker_size = None, color_list = None,
    title='production vs temperature', axis_text = 'solar production as function of temperature', ylabel="production Kwh", xlabel="temperature")

    ##############################################
    # scatter temp vs pressure vs production. use z
    ##############################################
    x = df_model["pressure"].to_numpy().astype('float')
    y = df_model["temp"].to_numpy().astype('float')
    _= various_plots.scatter(x,y, marker_size = z, color_list = z,
    title='pressure vs temperature vs production', axis_text = 'solar production as function of temperature and pressure', xlabel="pressure", ylabel="temperature")


    ####################
    # heat map. Compute pairwise correlation of columns
    ####################

    #df = df_model.drop(["date", "sin_hour", "cos_hour", "hour", "sin_month"], axis=1) # excludes date, hour, month, keep syn_month
    df = df_model[["temp", "pressure", "humid" , "wind", "direction" , "production"]]

    # month 1 to 12 can be misleading. maybe abs(month-6) 

    df = df_model[["temp", "pressure", "humid" , "wind", "direction" , "month", "production"]]

    def f(val):
        return abs(val-6)
    
    df["month"] = df["month"].apply(f)

    #method = "pearson" #  Measures the strength and direction of the linear relationship, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.
    #method = "spearman" # Measures the similarity in ranking order between two variables, where 1 indicates perfect agreement, -1 indicates perfect disagreement, and 0 suggests no association.
    #method = "kendall"

    _= various_plots.heat_map(df, title="feature correlation_linear", method="pearson")

    _= various_plots.heat_map(df, title="feature correlation_monotonic", method="spearman")

    _= various_plots.sns_heat_map(df, title="feature correlation_linear1", method="pearson")

    _= various_plots.sns_heat_map(df, title="feature correlation_monotonic1", method="spearman")

    

    ##################
    # A lot of histogram, using plt, pandas and sns. 
    # 1D: number of sample per bins
    # 2D: no hue: number  of sample per binx x biny
    # 2D: hue: value of z per binx x biny
    ##################


    #######################
    # 2D, using plt.
    #######################
    y = df_model["production"].to_numpy().astype('float')


    x = df_model["pressure"].to_numpy().astype('float')
    _= various_plots.histo_2D(x,y, ylabel="production", xlabel="pressure", title = "production pressure 2D hist", bins=(20,20))

    x = df_model["humid"].to_numpy().astype('float')
    _= various_plots.histo_2D(x,y, ylabel="production", xlabel="humid", title = "production humid 2D hist", bins=(20,20))

    x = df_model["temp"].to_numpy().astype('float')
    _= various_plots.histo_2D(x,y, ylabel="production", xlabel="temp", title = "production temp 2D hist", bins=(20,20))

    ################################
    # 1D pandas: histogram one serie
    ################################
    df = df_model["production"]
    various_plots.pd_histogram(df, bins=20, title = "production histogram")

    df = df_model["humid"]
    various_plots.pd_histogram(df, bins=20, title = "humid histogram")

    df = df_model["temp"]
    various_plots.pd_histogram(df, bins=20, title = "temperature histogram")

    df = df_model["pressure"]
    various_plots.pd_histogram(df, bins=20, title = "pressure histogram")

    ################################
    # 1D pandas: histogram TWO serie on same plot
    # normalized
    ################################
    # pandas: histogram 2 serie, ie 2 hist on same plot
    #df = df_model[["production", "pressure"]] not good if data not scaled. use same x range

    df_1 = df_model["production"]
    df_1 = (df_1 - df_1.mean())/df_1.std()
    df_2 = df_model["pressure"]
    df_2 = (df_2 - df_2.mean())/df_2.std()
    df = pd.concat([df_1, df_2], axis=1) # (23712, 2)
    # WARNING: will create own fig
    various_plots.pd_histogram(df, bins=20, title = "production and pressure histograms - normalized")


    ############################
    # pandas: 2D with hexagonal bins
    # If C is None (the default), this is a histogram of the number of occurrences of the observations at (x[i], y[i]).
    ############################
    various_plots.pd_hex_histogram(df_model, "pressure" , "production", c=None, title="production pressure hex histogram")

    various_plots.pd_hex_histogram(df_model, "humid" , "production", c=None, title="production humid hex histogram")

    various_plots.pd_hex_histogram(df_model, "wind" , "production", c=None, title="production wind hex histogram")

    various_plots.pd_hex_histogram(df_model, "wind" , "temp", c="production", title="production wind temp hex histogram")

    ##########################################
    # sns hist, kde (ie "curved histogram"), ecdf
    ##########################################

    ############################################
    # sns: 1D histogram, with histogram bars and curve approximating histo
    ############################################
    various_plots.sns_histplot(data = df_model, x= "production", bins= 20, title = 'production histogram1')

    ####################################
    # sns: 2D histograms 
    ####################################
    # When both x and y are assigned, a bivariate histogram is computed and shown as a heatmap:
    # number of sample per bins of productionxpressure
    various_plots.sns_histplot(data = df_model, y= "production", x = "pressure",  bins= 20 , title = 'production histogram3')

    various_plots.sns_histplot(data = df_model, y= "production", x = "temp",  bins= 20 , title = 'production histogram4')

    various_plots.sns_histplot(data = df_model, y= "production", x = "humid",  bins= 20 , title = 'production histogram5')

    various_plots.sns_histplot(data = df_model, y= "production", x = "wind",  bins= 20 , title = 'production histogram6')

    various_plots.sns_histplot(data = df_model, y= "production", x = "direction",  bins= 20 , title = 'production histogram7')

    # sns: kde, ie "curved histo", same curve as in sns 1D histo with kde = True
    various_plots.sns_kdeplot(data = df_model["production"], title = 'production histogram (kde)')
    #various_plots.sns_kdeplot(data = df_model, x= "production", title = 'kde production')

    various_plots.sns_kdeplot(data = df_model["temp"], title = 'temp histogram (kde)')

    various_plots.sns_kdeplot(data = df_model["pressure"], title = 'pressure histogram (kde)')

    various_plots.sns_kdeplot(data = df_model["humid"], title = 'humid histogram (kde)')

    various_plots.sns_kdeplot(data = df_model["wind"], title = 'wind histogram (kde)')

    # similar to join kind='kde'
    #various_plots.sns_kdeplot(data = df_model, x= "pressure", y = "production", title = 'kde production pressure')

    plt.close() # close current figure

    #################
    # quentile, box
    #################

    # pandas: box
    # A box plot is a method for graphically depicting groups of numerical data through their quartiles.
    df = df_model["production"]
    various_plots.pd_box(df, title="production quartile")

    df = df_model["pressure"]
    various_plots.pd_box(df, title="pressure quartile")

    df = df_model["temp"]
    various_plots.pd_box(df, title="temp quartile")

    df = df_model["humid"]
    various_plots.pd_box(df, title="humid quartile")

    df = df_model["wind"]
    various_plots.pd_box(df, title="wind quartile")

    # sns: quartile
    # single box. x is serie
    # two series: x = select serie (ie months), y = value to consider (ie production) , resulting in quartile per month
    # if use data, x,y are names (not serie)
    df = df_model["production"]
    various_plots.sns_box_plot(x=df, title = "solar quartile")

    df = df_model["temp"]
    various_plots.sns_box_plot(x=df, title = "temperature quartile")

    # multiple quartile. prod quartile per month
    # swap x/y and orient
    various_plots.sns_box_plot(x=df_model["month"], y=df_model["production"], orient='v', title = "production quartile per month")

    various_plots.sns_box_plot(x=df_model["month"], y=df_model["temp"], orient='v', title = "temperature quartile per month")

    various_plots.sns_box_plot(x=df_model["month"], y=df_model["pressure"], orient='v', title = "pressure quartile per month")

    various_plots.sns_box_plot(x=df_model["month"], y=df_model["wind"], orient='v', title = "wind quartile per month")

    various_plots.sns_box_plot(x=df_model["month"], y=df_model["humid"], orient='v', title = "humid quartile per month")

    ###################
    # violin
    # Draw a combination of boxplot (quentile) and kernel density estimate (ie "continous histograms").
    # Violin plots are used to visualize data distributions, displaying the range, median, and distribution of the data.
    ###################

    df = df_model["production"]
    # We can pass in just the X variable and the function will automatically compute the values on the Y-axis
    various_plots.sns_violin_plot(x=df, title = "production violin", orient='h')
    # UserWarning: Vertical orientation ignored with only `x` specified.
    # can also pass sns.violinplot(x="life_exp", data = dataframe)

    # quentile per month
    # pass in a categorical X-variable and a continuous Y-variable,
    # hue: group by categorical data. will use palette
    various_plots.sns_violin_plot(x=df_model["month"], y=df_model["production"], hue= df_model["month"], orient='v', title = "production violin per month")


    ######################
    # sns join plot, ie 2 variable
    #######################

    # Draw a plot of two variables with bivariate (aka 2D?) and univariate (aka 1D?) graphs.
    # ie outside: distribution for each variable
    # inside number of sample per X x Y

    print("starting sns join plot (2 variables)")

    # use dataframe or numpy
    y = df_model["pressure"].to_numpy().astype('float')
    x = df_model["production"].to_numpy().astype('float')

    # kind { “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }

    # use data = df_model and x,y, hue as serie name
    # or x,y,   [hue] serie

    # try different kind, prpduction vs pressure

    various_plots.sns_join_plot(x, y, title = "join production pressure scatter", kind="scatter")

    various_plots.sns_join_plot(x, y, title = "join production pressure kde", kind="kde")

    various_plots.sns_join_plot(x, y, title = "join production pressure hist", kind="hist")
    
    # hex is like hist, except box are hex vs square
    various_plots.sns_join_plot(x, y, title = "join production pressure hex", kind="hex")
    
    # add regression on side and inside
    various_plots.sns_join_plot(x, y, title = "join production pressure reg", kind="reg")

    various_plots.sns_join_plot(x, y, title = "join production pressure resid", kind="resid")

    # using hue, multiple colors per categorical 
    various_plots.sns_join_plot(x, y, title = "join production pressure per month", kind="hist", hue=df_model["month"])

    

    #########################################
    # sns pairwise relationships: n x n 
    #########################################

    # takes LOONG time
    # n x n  "sub plots"
    # diagonal and join plot of 2 variables
    #  kind = type of join plot. drive diagonal and outside. default is scatter (diag: hist). kde, hist, reg
    #    can "force" diagonal diag_kind

    # number of sample per bin x bin 
    # sems cannot use value of 3rd serie instead of number of samples

    print("starting sns pair plot. N x N.  can take SOME time ....")

    vars = ["pressure", "temp", "production"]

    df = df_model[vars]  # not really needed, as we use vars = vars already

    # kde way longer than others (200 sec vs 1 sec)

    various_plots.sns_pair_plot(df, kind='kde', vars= vars, title = "pair plot kde") # 200 sec
    various_plots.sns_pair_plot(df, kind='hist', vars = vars, title = "pair plot hist") # 1sec
    #various_plots.sns_pair_plot(df, kind='reg', vars = vars, title = "pair plot reg")
    various_plots.sns_pair_plot(df, kind='scatter', vars = vars, title = "pair plot scatter")

    # hue Variable to map plot per hue to different colors.
    various_plots.sns_pair_plot(df_model, kind='kde', vars = vars, title = "pair plot per month", hue = "month") # 200 sec
    various_plots.sns_pair_plot(df_model, kind='hist', vars = vars, title = "pair plot per month1", hue = "month") # 6 sec
    various_plots.sns_pair_plot(df_model, kind='scatter', vars = vars, title = "pair plot per month2", hue = "month")

    # a really big one
    vars = ["pressure", "temp", "production", "humid" , "wind", "direction", "month"]
    various_plots.sns_pair_plot(df_model, kind='hist', vars = vars, title = "pair plot1 hist") # 4 sec

    print("end plot input features")
