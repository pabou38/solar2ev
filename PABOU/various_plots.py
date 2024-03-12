#!/usr/bin/env python3

import datetime
import sys, os

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt 

import seaborn as sns
import calplot # calendar heatmap from pandas time serie. x year, y day

# https://levelup.gitconnected.com/5-extremely-useful-plots-for-data-scientists-that-you-never-knew-existed-5b92498a878f

sys.path.insert(1, '../my_modules') # this is in above working dir 

try:
    from my_decorators import dec_elapse
except:
    print("!!!! %s: cannot import" %__name__)
    sys.exit(1)


"""
# https://www.scaler.com/topics/matplotlib/introduction-to-figures-in-matplotlib/
The figure is the top-level container of all the axes and properties of a plot, 
or we can say that it is a canvas that holds the drawings (graphs or plots) on it. 

some pandas call create new figure (multiple serie) whereas other (single serie/line) seems to use currently opened figure
some pandas call allow to pass a figure (to be created) , whereas other do not
unless figure are closed, warning after 20 are opened (memory)
==> plt.close()     close() current figure, close(num), close(fig), close(str)
"""


# https://matplotlib.org/stable/users/explain/customizing.html

# runtime configuration
# global to matplotlib
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.linestyle'] = '--'

# Another way to change the visual appearance of plots is to set the rcParams in a so-called style sheet and import that style sheet with matplotlib.style.use
# Each style defines background color, text color and new cycles colors.
# https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
# https://mljar.com/blog/matplotlib-colors/

#plt.style.use("ggplot") 
plt.style.use("Solarize_Light2") # set the style globally for all plots

"""
with plt.style.context("ggplot"):
    plt.plot([1,2,3])
"""

##################
# matplotlib colormap
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

#Sequential: change in lightness and often saturation of color incrementally, often using a single hue; 
# should be used for representing information that has ordering.
# ('Perceptually Uniform Sequential',
#  ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
# 'Sequential',
#                     ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
# ('Sequential (2)',
#                     ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
#                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
#                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])

#Diverging: change in lightness and possibly saturation of two different colors that meet in the middle 
# at an unsaturated color; should be used when the information being plotted has a critical middle value, 
# such as topography or when the data deviates around zero.
# white in the middle
 
#'Diverging',
#                     ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
#                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])


#Cyclic: change in lightness of two different colors that meet in the middle and beginning/end at 
# an unsaturated color; should be used for values that wrap around at the endpoints, 
# such as phase angle, wind direction, or time of day.
# 'Cyclic', ['twilight', 'twilight_shifted', 'hsv']


#Qualitative: often are miscellaneous colors; should be used to represent information which does not have 
# ordering or relationships.
# 'Qualitative',
#                     ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
#                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
#                      'tab20c'])

# colormap name used to map scalar data to colors. 
# MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed two minor releases later. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap(obj)`` instead.
cm = plt.cm.get_cmap('plasma') # default: 'viridis'
cm1=plt.cm.jet
# can also use plt.___( , cmap="plasma", )


###############
# sns
##############

# Set the parameters that control the scaling of plot elements.
# This affects things like the size of the labels, lines, and other elements of the plot, “notebook”, and the other contexts are “paper”, “talk”, and “poster”,
# Calling this function modifies the global matplotlib rcParams
# The base context is “notebook”, and the other contexts are “paper”, “talk”, and “poster”, which are version of the notebook parameters scaled by different values.
# Separate scaling factor to independently scale the size of the font elements.
sns.set_context("paper", font_scale=1.3) 


# color of the background and whether a grid is enabled by default. 
# This is accomplished using the matplotlib rcParams system. 
# darkgrid, whitegrid, dark, white, ticks
sns.set_style ('dark') 


## color
# https://www.codecademy.com/article/seaborn-design-ii
# https://seaborn.pydata.org/tutorial/color_palettes.html

#qualitative palettes, good for representing categorical data
#sequential palettes, good for representing numeric data
#diverging palettes, good for representing numeric data with a categorical boundary


# Qualitative
# Qualitative (hue-based) palettes are well-suited to representing categorical data because most of their variation is in the hue component. 
# The default color palette in seaborn is a qualitative palette with ten distinct hue (similat to matplotlob Ten10)
# deep, muted, pastel, bright, dark, and colorblind

#   circular. simple transformation of RGB values
#   sns.color_palette("husl", 12)
# see also brewer
# sns.color_palette("Set2")

# Sequential
# primary dimension of variation in a sequential palette is luminance
# data range from relatively low or uninteresting values to relatively high or interesting values
# luminance-based palette to represent numbers
#   perceptually uniform, meaning that the relative discriminability of two colors is proportional to the difference between the corresponding data values
#   "rocket", "mako", "flare", and "crest", magma, viridis

# see also brewer
# "Blues", "YlOrBr"

# Diverging
# data where both large low and high values are interesting and span a midpoint value (often 0) that should be de-emphasized
# "vlag" and "icefire".
# see also brewer
# "Spectral", "coolwarm"


# get a palette into a var
palette = sns.color_palette("deep") # variation of default 

# warning: will create a SNS fig. if not closed, later sns plot may use it
sns.palplot(palette)  # plot palette
plt.close()


# Color Brewer is the name of a set of color palettes inspired by the research of cartographer Cindy Brewer
# nb of colors
sns.color_palette("Set3", 10) # Qualitative Palettes for Categorical Datasets
sns.color_palette("RdPu", 10) # Sequential color palettes are appropriate when a variable exists as ordered categories
sns.color_palette("RdBu", 10) # both the low and high values might be of equal interest, such as hot and cold temperatures.

# use a palette
# even if not using hie, seem to set the (only) color of sns plots
sns.set_palette("Paired", n_colors=None)

# use sns default pallet PRIOR using plt
sns.set()


# default dir where to save figure
save_fig_dir = "plot"
print("reusable plot module saving figure in %s" %save_fig_dir)
#  plt.savefig(os.path.join(dir, title))

if os.path.exists(save_fig_dir) == False:
    os.mkdir(save_fig_dir)


# plt: 9 images, scatter, histogram, correlation/heatmap, scatter 2D, 
# PANDAS:  box, histogram, hex bin, line
# seaborn: confusion, join (3D histo), pair (correlation), box, violin, histo/kde
# calplot: heatmap 

"""
# GENERIC FIGURE, SUBPLOT,
plt.figure(figzize=(14,5))
plt.suptitle()

plt.subplot(1,2,1)

sns.box, df.hist , 
plt.xlabel('')
plt.title('')

plt.subplot(1,2,2)
sns.......
plt.xlabel('')
plt.title('')

"""

################################
# using plt
################################

# NOTE: as plt is defined as global, once set by one caller, will modify plotting for next caller

def generic_plt(plt, x_label = "x_label", y_label = "y_label", title = "title", suptitle = "suptitle"):

    #plt.xticks(range(df.shape[1]), df.columns, fontsize=8, rotation=90) # add "label" on top, using colums names
    #plt.yticks(range(df.shape[1]), df.columns, fontsize=8) # add label on the left

    #plt.gca().xaxis.tick_bottom() # get current axis. Move ticks and ticklabels (if present) to the bottom of the Axes.
    #plt.gca().yaxis.tick_left()

    #cb = plt.colorbar(label="", orientation="vertical") # add color bar
    #cb.ax.tick_params(labelsize=8)

    #plt.xticks(rotation=90)
    #plt.yticks(rotation=0) 

    #ax.set_xticks(np.arange(len(labels)), labels=labels)  # set tick location and labels . bot miust have same size
    #ax.set_yticks(np.arange(len(labels)), labels=labels)

    plt.grid(True)
    plt.legend(loc = "lower left", fontsize = 5)
    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()  # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.

    plt.xlabel(x_label) # str
    plt.ylabel(y_label)

    plt.title(title , color = 'grey', loc = 'left', fontsize=10) # below suptitle. 

    plt.suptitle(suptitle) #main title, centered on top

    plt.style.use("ggplot")


    return(plt)



def plot_9_images_1 (plt, dataset, fig_size, title, class_list = None, do_predictions = False, model=None):

    ######################################
    # plot images
    # dataset element must be single picture
    # pic 0 255 float
    # imshow  0 255 int or 0 1 float
    # if augmentation not None, assumed to be a keras layers
    # predictions is an array of softmax
    #####################################

    if len(dataset) < 9:
        print('PABOU: plot image: dataset len is less than 9. cannot plot')
        return
    
    # print image characteristics
    for image, label in  dataset: # TensorShape([1, 160, 160, 3]) dtype=float32) 74.599976,  40.399994,  25.    
        image = image[0].numpy() # remove batch , got either  TensorShape([1, 224, 224, 3]) or TensorShape([32, 224, 224, 3])
        label = label[0].numpy()
        print('image: ', image.shape, image.dtype, np.max(image), np.min(image)) # (224, 224, 3) float32 255.0 0.0
        print('label: ', type(label), label) # <class 'numpy.int32'>
        break

    # subplotSSSS need to specify each subplots coordinates
    #pyplot.subplots creates a figure and a grid of subplots with a single call, while providing reasonable control over how the individual plots are created.
    #figure, arrx = plt.subplots(2,2)
    #plt.title('subplotsSSSS', loc='left', color = 'blue')
    #arrx[0,0].imgshow()
    # subplot WITHOUT S Add an Axes to the current figure or retrieve an existing Axes.

    plt.figure(figsize=fig_size, facecolor = 'blue', edgecolor = 'green') # create current figure (reperes) = separate window. size in inches
    # all plot will go to current figure (repere); with button, configure, save, zoom etc ..
    # figure closed with show
    plt.suptitle(title , x=0.2, y=0.95, fontsize=16)  # on TOP of entiere figure

    for i, (image, label) in enumerate(dataset.take(9)): # get 9 pic one after the other, i is used to go thru sub plots
    
        if do_predictions:
            if not model is None:
                predictions = model(image) # on a single image
                predictions = predictions[0]
            else:
                print('PABOU: model missing')
                return
        else:
            pass

        # type(images) <class 'tensorflow.python.framework.ops.EagerTensor'>
        # type(images.numpy()) <class 'numpy.ndarray'>
        # images.numpy().shape (1, 160, 160, 3) , ie has batch dim even if single image
        # cannot use ds[0] . object not subscriptable 

        # subplot WITHOUT S Add an Axes to the current figure or retrieve an existing Axes.
        ax = plt.subplot(3, 3, i + 1) # dans la figure precedemment cree. AUTO placement. create ONE sub

        # remove batch dim for imshow
        # imshow for (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
        # need to use .numpy. 'EagerTensor' object has no attribute 'astype'.
        # astype needed otherwize pic scrambled. as original pixel is float 0 to 255, convert float to int. or rescale to 0 1

        # image and label are TENSORS, not array

        # either use np.squeeze(image.numpy()) or tf.squeeze
        #image = np.squeeze(image.numpy(), axis = 0).astype("uint8") # remove batch, also convert to nparray, convert from 255.0 to 255 
        # or image = image[0].numpy().astype() 
        # or augmented_image = augmented_image / 255
        
        image = image.numpy()[0].astype("uint8")
        # another way to get rid of batch dim 
        label = label.numpy()[0]

        plt.imshow(image) # will go in each subplot

        # prediction is either None, or array of array
        # None has no .any() 
        # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

        # NOTE: if do_prediction  class_list is assumed to be set
        if do_predictions:
            p = np.argmax(predictions)
            s = 'prob:%0.1f pred:%s ' %(predictions[p], class_list[p])
        else:
            if class_list:
                s = 'label:%d class:%s' %(label, class_list[label])
            else:
                s = 'label:%d' %label
            
        
        plt.axis("off")

        plt=generic_plt(plt, title=s)

    return(plt) 



################
# loss
################

# figure(num=None,  is the same as not naming

def loss(history_dict, dir = save_fig_dir):
    global plt

    fig = plt.figure(num=None, figsize=(10,6), dpi=100, facecolor="white", edgecolor = "blue" )

    plt.plot(history_dict['loss'], label='loss')
    plt.plot(history_dict['val_loss'], label='val_loss')
    plt.ylim([0, len(history_dict['loss'])])

    plt = generic_plt(plt, x_label="epoch", y_label="loss", title="per epoch", suptitle="training loss")
    plt.legend(loc = "lower left", fontsize = 10)

    plt.tight_layout()
    plt.style.use("ggplot")
    plt.grid(True)
        

    plt.savefig(os.path.join(dir, "training loss"))
    plt.close()


##########
# scatter
# z could drive marker size (s) and/or color (c). in that case, color map map value to color 
# marker_size = z, color_list = z,
# use ptl
##########

def scatter(x,y, marker_size = None, color_list = None, axis_text = 'axis text', title='title', xlabel='xlabel', ylabel='ylabel', dir=save_fig_dir):

    global plt
    # x, y float or array-like, shape (n, )
    # title 
    # (num=None, .. is the same as not naming Fig (so will use Figure i++ on top left)
    fig = plt.figure(title, figsize=(10,6), dpi=100, facecolor="white", edgecolor = "blue" )  # do not use name for figure, as it may be used multiple time

    # param x, y: float or array-like, shape (n, )

    
    # can use serie for marker size and color, ie each dot will have a different size and color (using cmap)
    # if marker size = none, or s= 30, all dot same size
    # can have single color c="blue"

    # param s: size of dot, ie marker size

    # param c: array-like or list of colors or color. 
    # ie, single color, or list per dot or sequence of n numbers to be mapped to colors using cmap and norm.

    # param cmap: cmap is like a list of colors, where each color has a value that ranges from 0 to 100.
    # 'viridis' default, purple to yellow

    # alpha transparency
    #marker ="^"
    if isinstance(color_list, np.ndarray):
        cmap="jet" #  rainbow
        cmap = "turbo" # better smoother rainbow
    else:
        cmap=None  # would anyway be ignored if no color_list (ie c param)

    sc = plt.scatter(x,y, s=marker_size, c=color_list, cmap=cmap, marker="d", alpha=0.7)

    if isinstance(color_list, np.ndarray):
        # color bar label is what is used for z, ie color list
        plt.colorbar(sc, label='production') # add color bar to plot. derived from z. only really needed if z and x not the same


    plt = generic_plt(plt, x_label=xlabel, y_label=ylabel, suptitle=title, title= "scatter plot")

    # Add the text s to the Axes at location x, y in data coordinates.
    plt.text(0.0,0.0, axis_text, fontsize=8, horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5))

    plt.savefig(os.path.join(dir, title))
    plt.close()

    return(plt)


#######################################
# 1 histograms, use plt
#######################################


def single_histogram(x, bins, title, dir=save_fig_dir):
    global plt # UnboundLocalError: local variable 'plt' referenced before assignment

    # facecolor = color of bar
    # edgecolor = color of edge of bar

    figure= plt.figure(figsize=(5,5), facecolor="blue", edgecolor = "red" ) # figsize = size of box/window
 
    (bins_values, bin_edges, _) = plt.hist(x, bins=bins, facecolor='blue', alpha=0.5, edgecolor="gray", color="green")

    plt = generic_plt(plt, x_label="error", y_label="count", title="histogram", suptitle=title)

    plt.savefig(os.path.join(dir, title))

    plt.close(figure)



#######################################
# 2 histograms (subplot), use plt
#######################################

def histograms(values_1, values_2, bins, title, dir=save_fig_dir):
    # creates new figure = window . size in inch, create figure (10, size = ... figure 10 will show on upper left of window
    # subplot will go there

    global plt
    
    figure= plt.figure(figsize=(5,5), facecolor="blue", edgecolor = "red" ) # figsize = size of box/window

    plt.subplot(2,1,1) # 2xsubplot 1,1, 1,2 horizontal stack.  1,1 this is the top one. 2,1,2 is bottom
    plt = generic_plt(plt, x_label="Kwh", y_label="days", suptitle="before normalization", title=title)
    #plt.ylim(top=1) # set to limits turns autoscaling off for the y-axis. get also retrive current
    (bins_values, bin_edges, _)= plt.hist(values_1, bins=bins, facecolor='blue', alpha=0.5, edgecolor="#6A9662", color="#DDFFDD")

    plt.subplot(2,1,2) # 2xsubplot 1,1, 1,2 horizontal stack.  
    plt = generic_plt(plt, x_label="Kwh", y_label="days", suptitle="after normalization", title=title)
    (bins_values, bin_edges, _) = plt.hist(values_2, bins=bins)

    figure.savefig(os.path.join(dir,'prod_histogram.png'))
    plt.close()

##########
# correlation, aka heatmap
# Compute pairwise correlation of columns
# use plt
##########
def heat_map(df, title="heat map", dir=save_fig_dir, method = "pearson"):

    global plt 
    # so "inherit" state of plt so far
    #fig = plt.figure('heatmap', (6,6), dpi=100 ) # sns creates own figure

    # https://ishanjainoffical.medium.com/choosing-the-right-correlation-pearson-vs-spearman-vs-kendalls-tau-02dc7d7dd01d#:~:text=Data%20Type%3A%20Pearson%20is%20best,linear%20relationships%20compared%20to%20Pearson.
    # pearson: Sensitive to linear relationships, good for capturing linear trends.
    # Kendall and Spearman measure monotonic relationships,
    #method = "pearson" #  Measures the strength and direction of the linear relationship, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.
    #method = "spearman" # Measures the similarity in ranking order between two variables, where 1 indicates perfect agreement, -1 indicates perfect disagreement, and 0 suggests no association.
    #method = "kendall"

    c = df.corr(method=method) # n x n dataframeCompute pairwise correlation of columns, excluding NA/null values. 1 = fully correlated
    # -1 to +1

    # use sequantial color map, as correlation goes from -1 to 1 ?
    # rather use diverging
    cmap = "seismic"
    cmap = "RdBu"

    plt.matshow(c, cmap = cmap) # Display an array as a matrix in a new figure window. c 2D array

    plt.grid(False)
    plt.legend(loc = "lower left", fontsize = 5)
    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()  # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.

    if method == "pearson":
        s = "%s, ie assumes linear relationship" %method
    else:
        s = "%s, ie assumes monotonic relationship" %method

    plt.title(s, color = 'grey', loc = 'left', fontsize=10) # below suptitle. 
    plt.suptitle(title) #main title, centered on top

    plt.xticks(range(df.shape[1]), df.columns, fontsize=8, rotation=90) # add "label" on top, using colums names
    plt.yticks(range(df.shape[1]), df.columns, fontsize=8) # add label on the left

    plt.gca().xaxis.tick_bottom() # get current axis. Move ticks and ticklabels (if present) to the bottom of the Axes.
    plt.gca().yaxis.tick_left()

    cb = plt.colorbar(label="correlation", orientation="vertical") # add color bar
    cb.ax.tick_params(labelsize=8)


    plt.savefig(os.path.join(dir, title+"_%s" % method))
    plt.close()


###############
# histograms 2D
# input array like
# use bins 
# use plt
################
def histo_2D(x,y, xlabel=None, ylabel=None, title = "histo2D", bins=(10,10), dir = save_fig_dir):

    global plt

    fig = plt.figure(title, figsize=(8,6), dpi=100)
    
    cmap=plt.cm.jet # get a color map 

    plt.hist2d(x,y, bins=bins, cmap = cmap)
    
    plt.colorbar(label="count", orientation="vertical")

    plt = generic_plt(plt, x_label=xlabel, y_label=ylabel, title="2D histogram", suptitle=title)
    
    
    #ax = plt.gca() # get current axis <AxesSubplot: xlabel='month', ylabel='production'>
    #ax.axis('tight') # tight_layout attempts to resize subplots in a figure so that there are no overlaps between axes objects and labels on the axes.

    plt.savefig(os.path.join(dir, title))
    plt.close()

################################
# using PANDAS
panda_figsize = (8,6)
# line (with or without subplots) , box, histogram, hexbin
################################
    
# WARNING: BEWARE. Some plot will use existing figure (so all kind of problems, ticks, overwrite) whereas some will create own
# it seems that if df has more than 1 serie (hexbin, ..), a figure is automatically created by df.plot
    
#ax(axis or array of axis for subplot) = df.plot()
    
###### PANDAS line
    
# no subplots, all line in same figure
def pd_line(df, title="title", dir=save_fig_dir, xlabel = "xlabel", ylabel="ylabel"):

    # for panda, better not to create fig manually BEFORE df.plot()
    #   anyway WARNING: it seems that if df has more than 1 serie, a figure is automatically created by df.plot
    # so pass what would have been in figure() directly to pandas.plot

    # color="blue" with multiple lines , means all lines same color

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot
    ax = df.plot(kind='line', figsize = panda_figsize , legend=True, subplots=False, grid=True, title = title, xlabel = xlabel, ylabel=ylabel) # write data in fig
    # or df.plot.line()
    # return axes
 
    # not needed . already done in df.plot
    ax.set_ylabel(ylabel) # as soon as executed, displayed figure is updated
    ax.set_xlabel(xlabel)

    ### plt.savefig is the same as figure.savefig
    figure = ax.get_figure()

    figure.savefig(os.path.join(dir, title)) 

    plt.close(figure)# close figure. warning is >20 fig open (memory)


# use subplot=True to include multiple plots (of 1 lines) in same figure
def pd_subplot_lines(df, title, dir=save_fig_dir):

    ax_s = df.plot(title = title, figsize=panda_figsize, kind='line', legend=True, subplots=True,  grid=True)
    # array of axis 
    # below need to be synched with df.columns
    ax_s[0].set_ylabel('kWh')
    ax_s[1].set_ylabel('hpa')
    ax_s[2].set_ylabel('C')
    ax_s[3].set_ylabel('%')
    ax_s[4].set_ylabel('km/h')

    plt.savefig(os.path.join(dir,title+".png"))
    plt.close() # close current figure
    
###### PANDAS box plot quartile plot 
    # a method for graphically depicting groups of numerical data through their quartiles.
    # The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2). 
    # The whiskers extend from the edges of box to show the range of the data.
    # by default pandas would plot in current figure
    # '25% of data above box'

    #The box represents the data that exists between the first and third quartile also called the interquartile range (IQR = Q3-Q1). 
    # It contains 50% of the data and is divided into two parts by the median. 
    # The whiskers are represented according to the IQR proximity rule. Upper boundary = Third quartile + (1.5 * IQR). Lower boundary = First quartile — (1.5 * IQR)

    # https://medium.com/swlh/identify-outliers-with-pandas-statsmodels-and-seaborn-2766103bf67c

def pd_box(df, title='quartile', dir=save_fig_dir):

    # if does not create a new fig, issue with nb of ticks (from previous hist plots using plt. what a mess)
    fig= plt.figure(title,figsize=panda_figsize, facecolor="white", edgecolor = "green")
    plt.grid(False)

    df.plot(kind="box", figsize=panda_figsize, title = title, legend=True, subplots=False)

    plt.savefig(os.path.join(dir, title+'.png'))

    plt.close(fig) # close current or this one
    

###### PANDAS histograms plot 
#figure= plt.figure(title,figsize=(5,5), facecolor="white", edgecolor = "green")
#figure.savefig(os.path.join(dir, title+'.png'))

# or

#df.plot.hexbin(x,y,c, figsize=(8,5),
#plt.savefig(os.path.join(dir,title+".png"))
    
# see also plt histograms


def pd_histogram(df, bins= 10, title="pd histograms", dir=save_fig_dir):

    # if do not create new fig, plot over something previous. 
    fig= plt.figure(title,figsize=panda_figsize)
    plt.grid(False)

    df.plot(kind='hist', bins=bins, title = title, figsize=panda_figsize, facecolor="blue", edgecolor = "green", legend=True, subplots=False, color="blue", alpha=0.5) # this plots
    
    # df.plot(kind='hist', bins=3) same as df.plot.hist( bins=3)
    # Return a histogram plot.

    plt.savefig(os.path.join(dir, title + '.png'))
    plt.close()


###### PANDAS hexagonal bin 
# Generate a hexagonal binning plot of x versus y.  
# x columns label or index
# If C is None (the default), this is a histogram of the number of occurrences of the observations at (x[i], y[i]).  
# otherwize can supply own algo
# gridsize, number of hex

def pd_hex_histogram(df, x,y,c=None, title='pd hex', dir=save_fig_dir):

    # WTF. hexbin creates its own figure, so if I create one here, will be left lingering. I guess it does because more than, one serie
    #fig= plt.figure(title,figsize=panda_figsize)

    df.plot.hexbin(x,y,c, figsize=panda_figsize, title = title, facecolor="white", edgecolor = "green", gridsize=20, legend=True, subplots=False, grid=False)

    plt.savefig(os.path.join(dir,title+".png"))
    plt.close()

###################
# SEABORN
###################

def sns_heat_map(df, title = "sns heat map", dir=save_fig_dir, method = "pearson"):

    sns.set_context("paper", font_scale=1.3) 
    sns.set_style ('dark')

    #sns.color_palette("Set3", 10) # Qualitative Palettes for Categorical Datasets
    #sns.color_palette("RdPu", 10) # Sequential color palettes are appropriate when a variable exists as ordered categories
    #sns.color_palette("RdBu", 10) # both the low and high values might be of equal interest, such as hot and cold temperatures.
    sns.set_palette("Set3", n_colors=None)

    
    # https://ishanjainoffical.medium.com/choosing-the-right-correlation-pearson-vs-spearman-vs-kendalls-tau-02dc7d7dd01d#:~:text=Data%20Type%3A%20Pearson%20is%20best,linear%20relationships%20compared%20to%20Pearson.
    # pearson: Sensitive to linear relationships, good for capturing linear trends.
    # Kendall and Spearman measure monotonic relationships,
    #method = "pearson" # ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.
    #method = "spearman"  # Measures the similarity in ranking order between two variables, where 1 indicates perfect agreement, -1 indicates perfect disagreement, and 0 suggests no association.
    #method = "kendall"

    c = df.corr(method=method) # n x n dataframeCompute pairwise correlation of columns, excluding NA/null values. 1 = fully correlated
    # -1 to +1

    ax = sns.heatmap(c, annot=True, linewidth=.5, cmap="crest")

    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()

    plt.savefig(os.path.join(dir,title+"_%s" %method))
    plt.close()



    

############################
# confusion matrix
# columns represent the prediction labels and the rows represent the real labels
############################
def plot_confusion_matrix_sns_heatmap(confusion, labels, title='confusion', dir = save_fig_dir):

    #fig = plt.figure(title, (8,8)) # do not name it, can create mutiple

    sns.set_context("paper", font_scale=1.3) 
    sns.set_style ('dark')

    #sns.color_palette("Set3", 10) # Qualitative Palettes for Categorical Datasets
    #sns.color_palette("RdPu", 10) # Sequential color palettes are appropriate when a variable exists as ordered categories
    #sns.color_palette("RdBu", 10) # both the low and high values might be of equal interest, such as hot and cold temperatures.
    sns.set_palette("Set2", n_colors=None)

    # annot If True, write the data value in each cell
    # fmtstr, optiona String formatting code to use when adding annotations.
    # xticklabel only works if data is dataframe with columns names
    # colar bar does not add too much if using annotation
    # default matplotlib cmap 
    ax = sns.heatmap(confusion, annot=True, fmt='g', cmap=None, linewidth=2.0,xticklabels = True, yticklabels = True, cbar=False )

    #sns.set(rc={'figure.figsize':(8, 8)})   deprecated
    #sns.set(font_scale=1.4)

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0) 

    #ax.xaxis.set_ticklabels(labels) # The use of this method is discouraged, because of the dependency on tick positions
    #ax.yaxis.set_ticklabels(labels)

    ax.set_xticks(np.arange(len(labels)), labels=labels)  # set tick location and labels . both must have same size
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    #fig.savefig(os.path.join(dir,title+'.png'))
    plt.savefig(os.path.join(dir,title+'.png'))
    plt.close()


###################################
# 2 values, histograms on the side
# Draw a plot of two variables with bivariate and univariate graphs.
# ie outside: distribution for each variable
# inside number of sample per X x Y
###################################

def sns_join_plot(x, y, title = "join plot", dir=save_fig_dir, kind="hex", hue=None):

    sns.set_context("paper", font_scale=1.3) 
    sns.set_style ('dark') # color of the background and whether a grid is enabled by default.

    #sns.color_palette("Set3", 10) # Qualitative Palettes for Categorical Datasets
    #sns.color_palette("RdPu", 10) # Sequential color palettes are appropriate when a variable exists as ordered categories
    #sns.color_palette("RdBu", 10) # both the low and high values might be of equal interest, such as hot and cold temperatures.
    sns.set_palette("Set3", n_colors=None)

    if hue is not None:
        palette = 'husl' # cylic (eg use month as hue)
        #sns.set_palette("Set3", n_colors=None)
    else:
        palette = None
        sns.set_palette("Set3", n_colors=None)

    sns.jointplot(x=x, y=y, kind=kind, hue=hue, palette=palette) #xlim=(-5,5), ylim=(-5,5)#
    plt.suptitle(title)

    plt.savefig(os.path.join(dir, title))
    plt.close()


############################### 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# This is a high-level interface for PairGrid 
# correlation and histograms
# kind 'kde', 'hist' , 'reg' : diagonal either histogram or probability density
# Plot pairwise relationships in a dataset.
#############################

# palette='hls' # set palette only if hue is set
# input is dataframe
# vars: Variables within data to use, otherwise use every column with a numeric datatype.
# hue: Variable in data to map plot aspects to different colors.

@dec_elapse
def sns_pair_plot(df, kind='kde', vars=None, hue = None, title = "pair plot", dir = save_fig_dir, palette=None ):

    sns.set_context("paper", font_scale=1.3) 
    sns.set_style ('dark')

    #sns.color_palette("Set3", 10) # Qualitative Palettes for Categorical Datasets
    #sns.color_palette("RdPu", 10) # Sequential color palettes are appropriate when a variable exists as ordered categories
    #sns.color_palette("RdBu", 10) # both the low and high values might be of equal interest, such as hot and cold temperatures.

    if hue != None:
        palette = 'husl' # circular, as use month as hue
    else:
        palette = None
        sns.set_palette("Pastel1", n_colors=None)

    g = sns.pairplot(df, vars= vars, kind=kind, hue=hue, palette=palette)

    # creates courbe de niveaux
    #g.map_lower(sns.kdeplot, levels=4, color=".2")

    plt.suptitle(title)
    plt.savefig(os.path.join(dir, title))
    plt.close()


#############################################
# box (or box-and-whisker) plot to show distributions with respect to categories. (quartile)
############################################

# The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution,
#  except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.
# x, y are for grouping using categorical values
# for one categorie, use x = df[""], 
# x = series to include in one plot, eg type of petal, typically categorical or month, etc .., y = value, eg petal len
# if use data , x and y are name (not serie)
# https://seaborn.pydata.org/generated/seaborn.boxplot.html

def sns_box_plot(data=None, x=None, y=None, title= "box plot", dir = save_fig_dir, orient='h'):

    # palette required hue

    ax = sns.boxplot(data=data, x=x, y=y, orient=orient, width=0.5)

    ax.set_title("quentile")

    plt.suptitle(title)
    plt.savefig(os.path.join(dir, title))
    plt.close()


####################################################
# violin
# Draw a combination of boxplot and kernel density estimate.
####################################################
# x categorival, y continuous  
# or x continuous
    
# data and x = ""
# of x = df[""]
    
def sns_violin_plot(data = None, x = None, y = None, hue = None , title = "violin plot", dir=save_fig_dir, orient='v'):

    sns.set_palette("Pastel1", n_colors=None)
    sns.set_palette("RdBu")
    sns.set_style("darkgrid")

    # can use hue to color based on categorical data
    #inner{“box”, “quart”, “point”, “stick”, None} "box": draw a miniature box-and-whisker plot, "quart": show the quartiles of the data
    # split Show an un-mirrored distribution, alternating sides when using hue.
    # legend“auto”, “brief”, “full”, or False

    ax = sns.violinplot(data=data, x=x, y=y, hue = hue, cut=0, orient=orient, linewidth= 2.0, fill=True, inner ="quart", split=True, legend="auto")

    ax.set_title("distribution and quentile") # for this graph, so below suptitle

    if y is None:
        ax.set_ylabel("count")
        ax.set_xlabel("Kwh")    # simple histograms , Y kwh, X nb of sample
    else:
        ax.set_ylabel("Kwh")
        ax.set_xlabel("month")


    plt.suptitle(title) # of entiere chart
    plt.savefig(os.path.join(dir, title))
    plt.close()

#########################################
# hist , kde (ie "curved" histo), ecdf
#########################################
def sns_histplot(data=None, x = None, y = None, bins = 20, title = "sns hist", dir = save_fig_dir):

    fig = plt.figure(title, figsize=(7,7), dpi=100 )

    # element (bars, step (no separation for bins), poly (histo follows kde)), kde only for univariate (1 serie)

    # stat: 
    #count: show the number of observations in each bin
    #frequency: show the number of observations divided by the bin width
    #probability or proportion: normalize such that bar heights sum to 1
    #percent: normalize such that bar heights sum to 100
    #density: normalize such that the total area of the histogram equals 1

    # https://seaborn.pydata.org/tutorial/color_palettes.html

    # palette requires hue
    palette = sns.color_palette("Set2")
    palette = "mako" 

    sns.histplot(data=data, x=x, y=y, kde=True, element = "bars", bins=bins, legend = True, stat = "count")

    plt.suptitle(title)
    plt.savefig(os.path.join(dir, title))
    plt.close()

# use y to flip the axe
# do not specify x,y to plot all
# bw_adjust : smoothing
# 2 dim, use x, y 
# palette only used for hue= (fifferent color for different categories)

def sns_kdeplot(data=None, x = None, y = None, title = "sns kde", dir = save_fig_dir):

    fig = plt.figure(figsize=(6,6), dpi=100 )

    sns.kdeplot(data=data, x=x, y=y, bw_adjust=1, fill=True)

    plt.suptitle(title)
    plt.savefig(os.path.join(dir, title))
    plt.close()

#######################
# CALPLOT
# calendar heatmap from pandas time serie. per day
# https://calplot.readthedocs.io/en/latest/ 
# https://python-charts.com/evolution/calendar-heatmap-matplotlib/
#######################
    
def calplot_heatmap(serie,title = "title"):
   
    # serie Must be indexed by a DatetimeIndex.

    cmap='YlGn'
    cmap = 'Spectral_r'

    # https://calplot.readthedocs.io/en/latest/

    # fix figsize, otherwize years not displayed correctly 
    (fig, ax) = calplot.calplot(serie, suptitle = title,  figsize=(10,6.5) , cmap=cmap, edgecolor = 'black', linewidth = 1, colorbar = True)

    #(fig, ax) = calplot.yearplot(serie, edgecolor = 'black', linewidth = 1) # only one year



    """
    plt.show starts an event loop, creates interactive windows and shows all current figures in them. If you have more figures than you actually want to show in your current pyplot state, 
    you may close all unneeded figures prior to calling plt.show().

    ie: show all figures NOT closed
    
    In contrast fig.show() will not start an event loop. It will hence only make sense in case an event loop already has been started, e.g. after plt.show() has been called.
    In non-interactive mode that may happen upon events in the event loop.
    
    """
 
    fig.savefig(os.path.join(save_fig_dir, title))  # save CURRENT figure

    plt.close(fig)






