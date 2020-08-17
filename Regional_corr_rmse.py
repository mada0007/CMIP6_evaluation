



### code to calculate errors between predicted (models) and observed metrics:
## Author Mustapha Adamu:
## 02-03-2016
## REF ::https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
## Import Libraries::


import xarray as xr
import numpy as np
import xarray as xr
import cartopy.crs as ccrs  # This a library for making 2D spatial plots in python
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from cartopy.util import add_cyclic_point
import numpy as np
import scipy
import scipy.signal
import scipy.stats as s
import cartopy as cart
import matplotlib as mpl
import cmocean as cm  #special library for making beautiful colormaps
import glob
import sys
import os
import cmocean.cm as cmo
from scipy.stats import genextreme as gev
from math import sqrt
from sklearn.metrics import mean_squared_error
import regionmask
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats 
import warnings
warnings.filterwarnings("ignore")


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted
def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))
def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    # return rmse(actual, predicted) / np.nanmean(actual)
    return rmse(actual, predicted)




## PATH FOR CMIP6 DATA

data_path1= "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/miss"  # path for CMIP  dat
regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/low_res_obs/reg_mon.nc').p


## get arrays for lon and lat use for regridding models
lon = regen_data.lon
lat = regen_data.lat

## select African here from the global precipitation data
data_reg = regen_data

## create Regional mask
mask = regionmask.defined_regions.srex.mask(data_reg)
## select time slice here
data_reg= data_reg.sel(time=slice('1950-01', '2005-12')) # select to match historical CMIP6
data_cmip_trend = data_reg.sel(lon=slice(-20,50), lat = slice(-40,47)).mean(dim=['lat','lon'])
slope, intercept, r_value, p_value, std_err = stats.linregress(data_cmip_trend,np.arange(0,len(data_cmip_trend),1))
print(f'the trend of is {slope} and p-value is {p_value}')
        
time = data_reg.time
data_reg = data_reg.groupby('time.month').mean('time')






# Selection regions of Africa here::a


## West Africa
WAF =data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:].mean(dim=['lon','lat'])


## North Africa
NAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:].mean(dim=['lon','lat'])


## East Africa
EAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:].mean(dim=['lon','lat'])

## S Africa
SAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:].mean(dim=['lon','lat'])



# create empty arrays for storing models REGIONAL  data

WAF_all = []

NAF_all = []

EAF_all = []

SAF_all = []

SAHEL_all = []


## create arrays for storing overlap metric

WAF_ovl = []

NAF_ovl = []

EAF_ovl = []

SAF_ovl = []

SAHEL_ovl = []


## create array for storing exceedance::

waf_all = []
saf_all = []
naf_all = []
eaf_all = []
sahel_all = []

JJA_models = np.zeros(regen_data.shape) * np.nan

mmm = []
models = sorted((os.listdir(data_path1))) # list all the data in the model

models_cmi6 = models


for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid of missing data\
       continue
    # print(models)

    files = (glob.glob(data_path1 + "/" + models[m]))



    for data in files:  # find files in  folder

        #** Grad model dataset

        # dset = xr.open_dataset(data)
        # print(dset.source)
        # print(dset.institution)


        dset = xr.open_dataset(data).pr * 86400


        dset = dset.where(dset>0)
        # print(dset)


        # dset = dset.sel(time=dset.time.dt.month.isin([12,1,2]))


        data_cmip = dset.sel(time=slice('1950-01', '2005-12'))
        data_cmip_trend = data_cmip.sel(lon=slice(-20,50), lat = slice(-40,47)).mean(dim=['lat','lon'])
        mmm.append(data_cmip_trend)     #append MMM for checking trend in MMM
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_cmip_trend,np.arange(0,len(data_cmip_trend),1))
        print(f'the trend of {models[m]} is {slope} and p-value is {p_value}')
        
        data_cmip = data_cmip.groupby('time.month').mean('time')







        data_reg = data_cmip
        mask = regionmask.defined_regions.srex.mask(data_reg)

        ## West Africa
        WAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:].mean(dim=['lon','lat']) ## obtain the time series
        # WAF2 = nrmse(np.array(WAF[:len(WAF1)]),np.array(WAF1))
        WAF2 = (np.mean(WAF1))- (np.mean(WAF[:len(WAF1)]))
        WAF_all.append(np.array(WAF2))





        ## North Africa
        NAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:].mean(dim=['lon','lat'])
        # NAF1 = nrmse(np.array(NAF[:len(WAF1)]),np.array(NAF1))
        NAF1 = (np.mean(NAF1)) - (np.mean(NAF[:len(WAF1)]))
        NAF_all.append(np.array(NAF1))





        ## East Africa
        EAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:].mean(dim=['lon','lat'])
        # EAF1 = nrmse(np.array(EAF[:len(WAF1)]),np.array(EAF1))
        EAF1 =(np.mean(EAF1))- (np.mean(EAF[:len(WAF1)]))


        EAF_all.append(np.array(EAF1))



        ## S Africamean(dim=['lon','lat'])
        SAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:].mean(dim=['lon','lat'])
        # SAF1 = nrmse(np.array(SAF[:len(WAF1)]),np.array(SAF1))
        SAF1 = (np.mean(SAF1))-(np.mean(SAF))
        SAF_all.append(np.array(SAF1))


mmm_all = np.nanmean(mmm, 0)
slope, intercept, r_value, p_value, std_err = stats.linregress(mmm_all,np.arange(0,len(data_cmip_trend),1))
print(f'the trend of is {slope} and p-value is {p_value}')




#--------------------------------------------------------------------------------------------------------------------
## for CMIP5
##------------------------------------------------------------------------------------------------------------------


data_path1= "/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/miss"


# create empty arrays for storing models REGIONAL  data

WAF_all1 = []

NAF_all1 = []

EAF_all1 = []

SAF_all1 = []
mmm = []



JJA_models = np.zeros(regen_data.shape) * np.nan

models = sorted((os.listdir(data_path1))) # list all the data in the model

models_cmi5 = models


for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid of missing data\
       continue
    if not  models[m].endswith('.nc'):
        continue

    # print(models)

    files = (glob.glob(data_path1 + "/" + models[m]))

    for data in files:  # find files in  folder

        #** Grad model dataset

        # dset = xr.open_dataset(data)
        # # print(dset.institution)
        # # print(dset.source)
        # dset = xr.open_dataset(data).pr
        dset = xr.open_dataset(data).pr * 86400
        dset = dset.where(dset>0)




        data_cmip = dset.sel(time=slice('1950-01', '2005-12'))
        data_cmip_trend = data_cmip.sel(lon=slice(-20,50), lat = slice(-40,47)).mean(dim=['lat','lon'])
        mmm.append(mmm)
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_cmip_trend,np.arange(0,len(data_cmip_trend),1))

        print(f'the trend of {models[m]} is {slope} and p-value is {p_value}')

        data_cmip = data_cmip.groupby('time.month').mean('time')


        ## Change data_reg to match models i.e. Data_reg::
        data_reg = data_cmip
        mask = regionmask.defined_regions.srex.mask(data_reg)

        ## West Africa
        WAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:324].mean(dim=['lon','lat']) ## obtain the time series
        # WAF2 = nrmse(np.array(WAF[:len(WAF1)]),np.array(WAF1))
        WAF2 = (np.mean(WAF1)) - (np.mean(WAF))
        WAF_all1.append(np.array(WAF2))

        ## North Africa
        NAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:324].mean(dim=['lon','lat'])
        NAF1 =  (np.mean(NAF1))-(np.mean(NAF))
        NAF_all1.append(np.array(NAF1))


        ## East Africa
        EAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:324].mean(dim=['lon','lat'])
        # EAF1 = nrmse(np.array(EAF[:len(WAF1)]),np.array(EAF1))
        EAF1 = (np.mean(EAF1)) - (np.mean(EAF))
        EAF_all1.append(np.array(EAF1))


        SAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:324].mean(dim=['lon','lat'])
        # SAF1 = nrmse(np.array(SAF[:len(WAF1)]),np.array(SAF1))

        SAF1 = (np.mean(SAF1)) -(np.mean(SAF))
        SAF_all1.append(np.array(SAF1))

# mmm_all = np.nanmean(mmm, 0)
# slope, intercept, r_value, p_value, std_err = stats.linregress(mmm_all,np.arange(0,len(data_cmip_trend),1))
# print(f'the trend of is {slope} and p-value is {p_value}')

# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_rmse_cmip6', WAF_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_rmse_cmip6', NAF_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_rmse_cmip6', EAF_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_rmse_cmip6', SAF_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_rmse_cmip5', WAF_all1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_rmse_cmip5', NAF_all1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_rmse_cmip5', EAF_all1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_rmse_cmip5', SAF_all1)
#-----------------------------------------------------------------------------------------------------------------------
#### plotting results as boxplots:::::::
#-----------------------------------------------------------------------------------------------------------------------
ticks = ['CMIP6', 'CMIP5', ]   ## Phase titles::::::::::::::;;;

q = 0.2       ## scale factor for positioning box plots on xaxis ::::::::::::::;;
n_colors = 22 ## total number of colors needed for plots::::::::::::::::::::::;;


#************************ simple function to set plotting resources:::::::::::::::::::::::
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color,linewidth=3)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color,linewidth=3)
    plt.setp(bp['medians'], color=color, linewidth=3)


#**********************************************************************************

colors = [ 'chocolate','goldenrod','red', 'brown', 'grey', 'plum', 'springgreen', 'olive', 'pink', 'magenta', 'wheat', 'darkgoldenrod',
          'k', 'darkgreen', 'teal', 'deepskyblue', 'royalblue', 'indigo', 'violet', 'cyan', 'grey', 'cadetblue', 'skyblue', 'teal']

c_keys = ['o', 'v', '^', '<', '>', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', 'P', 'X', 0, 1, 2,
          3, 4, 5, 6, 7, 8, 9, 10, 11, '4', '8', 's', 'p', '*','p', '*',  ] # Markes for plotting 

markers1 = ['o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*', 'o', 'v', 'p', 'd', 's','*', ]
colors1 = ['red','red','red','red','red','red','green', 'green', 'green', 'green','green', 'green' ,'skyblue', 'skyblue', 'skyblue', 'skyblue','skyblue', 'skyblue','magenta', 'magenta', 'magenta', 'magenta', 'magenta' ]


colors2 = ['orange','orange', 'orange', 'orange', 'orange','orange','gold', 'gold' ,'gold', 'gold', 'gold','gold','pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'cyan','cyan', 'cyan', 'cyan', 'cyan' ]

# alpha = np.linspace(0.2, 1, 30)
# import itertools
# marker = itertools.cycle((',', '+', '.', 'o', '*'))


print(models_cmi5)

#******************************************* Begin plotting here ***********************************

f, ax = plt.subplots(2, 2, figsize=(10, 12), sharex='all', gridspec_kw={'wspace': 0.29, 'hspace': 0.2})

axes = ax.flatten()

#******************************************************* WEST AFRICA AXIS1 *********************************************
a = WAF_all
b = WAF_all1
data_a = [a, b, ]             ##put all data together   ************************
axes[0].grid(linestyle = '--', linewidth=1.5,)
axes[0].axhline(linestyle = '--', linewidth=2.5, color='r',)
#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues

print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmi6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=90,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
for i in range(len(b)):
    lab = models_cmi5[i].strip('.nc')
    bpp = axes[0].scatter(x[i], b[i], s=90,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab)

set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
axes[0].set_title('WAF      (a)', fontsize=15, weight='bold')
axes[0].set_ylabel('BIAS [mm]', fontsize=15, weight='bold')
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)



#******************************************************* NAF AFRICA AXIS1 *********************************************
axes[0] = axes[2]
axes[0].grid(linestyle = '--', linewidth=1.5,)

a = NAF_all
b = NAF_all1
data_a = [a, b, ]           ##put all data together   ************************

axes[0].axhline(linestyle = '--', linewidth=2.5, color='r',)
#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmi6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=90,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 
#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
for i in range(len(b)):
    lab = models_cmi5[i].strip('.nc')
    bpp = axes[0].scatter(x[i], b[i], s=90,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab+'*')

set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
axes[0].set_xticklabels(ticks, fontsize=10, weight='bold')
axes[0].set_title('SAH     (c)', fontsize=15, weight='bold')
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
axes[0].set_ylabel('BIAS [mm]', fontsize=15, weight='bold')
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)

#****************************************************************************


###*************************************** EAST
axes[0] = axes[1]
a = EAF_all
b = EAF_all1
data_a = [a, b, ]
axes[0].grid(linestyle = '--', linewidth=1.5,)

axes[0].axhline(linestyle = '--', linewidth=2.5, color='r',)
#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmi6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=90,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
for i in range(len(b)):
    lab = models_cmi5[i].strip('.nc')
    bpp = axes[0].scatter(x[i], b[i], s=90,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab)
set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
axes[0].set_title('EAF      (b)', fontsize=15, weight='bold')
# axes[0].set_ylabel('RMSE [mm]', fontsize=15, weight='bold')
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)


for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)


###************************************* South Africa ****************************
#*******************************************************************************
axes[0] = axes[3]
a = SAF_all
b = SAF_all1
data_a = [a, b, ]
axes[0].grid(linestyle = '--', linewidth=1.5,)

axes[0].axhline(linestyle = '--', linewidth=2.5, color='r',)
#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmi6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=90,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
for i in range(len(b)):
    lab = models_cmi5[i].strip('.nc')
    bpp = axes[0].scatter(x[i], b[i], s=90,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab)

set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
axes[0].set_title('SAF      (d)', fontsize=15, weight='bold')
# axes[0].set_ylabel('RMSE [mm/day]', fontsize=15, weight='bold')
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.labelsize'] = 10 
axes[1].legend(bbox_to_anchor=(-0.15, -1.57),loc='lower center',ncol=7,prop={'size': 8,'weight':'bold'})
# f.tight_layout()
# axes[0].legend(bbox_to_anchor=(0.94, 0.8),loc='upper left',prop={'size': 6,'weight':'bold'})
plt.subplots_adjust(wspace=1, hspace=0.5,left=0.1,top=0.9,right=0.9,bottom=0.15)
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)

plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/BIAS_ALL_REGIONS_v3', dpi=1000)