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
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import numpy as np
import scipy
import scipy.signal
import scipy.stats as s
import glob
import sys
import os
from scipy.stats import genextreme as gev
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import regionmask
import matplotlib.patches as mpatches





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


data_path1=  "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid/mask"    # path for CMIP6 Historical  dat




## This is a function to compute the overlap skill score of two arrays







regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/low_res_obs/reg_mon.nc').p


## get arrays for lon and lat use for regridding models
lon = regen_data.lon
lat = regen_data.lat

## select African here from the global precipitation data
# data_reg = regen_data

# ## create Regional mask

# mask = regionmask.defined_regions.srex.mask(data_reg)



# ## select time slice here
# # data = data_reg.sel(time=data_reg.time.dt.month.isin([12,1,2]))
# data_reg= data_reg.sel(time=slice('1950-01', '2005-12'))
# data_reg = data_reg.sel(lon=slice(-20,50), lat = slice(-35,47))
# # mean_obs = data_reg.mean(dim=['lon', 'lat'])
# mean_obs = np.array(data_reg.groupby('time.month').mean('time').mean(dim=['lon','lat']))

# mean_obs = mean_obs.ravel()
# data_reg =  data_reg.groupby('time.month').mean('time').mean(dim='lon')
# print(data_reg)
# max_obs = []

# for i in range(len(data_reg.month)):
#     ts = data_reg[i, :]
#     print(ts)
#     max = ts.max()
#     ind_max = np.max(np.where(ts ==max)[0])
#     print(ind_max)
#     max_obs.append(np.array(data_reg.lat[ind_max]))
# plt.plot(max_obs)
# plt.show()

# # print(data_reg)









# # create empty arrays for storing models REGIONAL  data

# WAF_all = []

# NAF_all = []

# EAF_all = []

# SAF_all = []

# SAHEL_all = []


# ## create arrays for storing overlap metric

# WAF_ovl = []

# NAF_ovl = []

# EAF_ovl = []

# SAF_ovl = []

# SAHEL_ovl = []




# #
# # ## create array for storing exceedance::

# waf_all = []
# saf_all = []
# naf_all = []
# eaf_all = []
# sahel_all = []

# max_models_all =[]

# rmse_all = []
# corr_all = []

# JJA_models = np.zeros(regen_data.shape) * np.nan

# models = sorted((os.listdir(data_path1))) # list all the data in the model

# mod = models
# for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
#     # print(models)

#     if  models[m].startswith('.'): # get rid of missing data\
#        continue
#     # print(models)

#     files = (glob.glob(data_path1 + "/" + models[m]))
#     print(files)



#     for data in files:  # find files in  folder

#         #** Grad model dataset

#         dset = xr.open_dataset(data).pr * 86400

#         # dset = dset.sel(time=dset.time.dt.month.isin([12,1,2]))
#         data_reg = dset.sel(time=slice('1950-01', '2005-12'))
#         data_reg = data_reg.sel(lon=slice(-20, 50), lat=slice(-35, 47)) ## select Africa
#         # mean_model = data_reg.groupby('time.month').mean('time').mean(dim=['lon', 'lat'])
#         mean_model = np.array(data_reg.groupby('time.month').mean('time').mean(dim=['lon','lat']))
#         # print(len(mean_model.ravel()))
#         # mean_model = mean_model.ravel()
#         corr = np.corrcoef(mean_model, mean_obs[:len(mean_model)])[0, 1] ## corr
#         corr_all.append(corr)
#         rmse1 = rmse(np.array(mean_obs[:len(mean_model)]), np.array(mean_model))
#         rmse_all.append(rmse1)
#         print(f'rmse1 {rmse1}')
#         print(f'corr {corr}')

#         data_reg = data_reg.groupby('time.month').mean('time').mean(dim='lon') # group by month and lon
#         max_models = []

#         for i in range(len(data_reg.month)):
#             ts1 = data_reg[i, :] ## get lat time series
#             # print(ts1)
#             max = ts1.max() ## maximum over each lat
#             ind_max = np.max(np.where(ts1 == max)[0]) ## get index
#             # print(ind_max)
#             max_models.append(np.array(data_reg.lat[ind_max])) ## Append data o

#         max_models_all.append(max_models)

#         # corr = np.corrcoef(np.array(max_obs), np.array(max_models))[0,1] ## corr
#         # corr_all.append(corr)
#         # rmse1 = rmse(np.array(max_obs), np.array(max_models))
#         # rmse_all.append(rmse1)
# #--------------------------------------------------------------------------------------------------------------------
# ## for CMIP5
# ##------------------------------------------------------------------------------------------------------------------

# # data_path1= "/Volumes/G/RAWORK/Historical/regrid"


# ## This is a function to compute the overlap skill score of two arrays

# data_path1= "/g/data/w35/ma9839/PRECIP_CMIP5/regrid/mask"


# # create empty arrays for storing models REGIONAL  data

# max_models_all1 =[] ## empty array for storing all models itcz CMIP5
# rmse_all1 = []
# corr_all1 = []

# JJA_models = np.zeros(regen_data.shape) * np.nan


# models1 = sorted((os.listdir(data_path1))) # list all the data in the model


# for m in range(len(models1)):  #**** loop through all models, amip and hist data must be have same names in different folders
#     # print(models)

#     if  models1[m].startswith('.'): # get rid of missing data\
#        continue
#     # print(models)

#     files = (glob.glob(data_path1 + "/" + models1[m]))
#     # print(files)


#     for data in files:  # find files in  folder

#         #** Grad model dataset

#         dset = xr.open_dataset(data).pr * 86400
       

#         data_reg = dset.sel(time=slice('1950-01', '2005-12'))
#         data_reg = data_reg.sel(lon=slice(-20, 50), lat=slice(-35, 47)) ## select Africa
#         mean_model = np.array(data_reg.groupby('time.month').mean('time').mean(dim=['lon','lat']))
#         # print(len(mean_model.ravel()))
#         # mean_model = mean_model.ravel()
#         # print(len(mean_model), len(mean_obs[:len(mean_model)]))
#         corr = np.corrcoef(mean_model, mean_obs[:len(mean_model)])[0, 1] ## corr
#         corr_all1.append(corr)
#         rmse1 = rmse(np.array(mean_obs[:len(mean_model)]), np.array(mean_model))
#         rmse_all1.append(rmse1)
#         print(f'rmse1 {rmse1}')
#         print(f'corr {corr}')
        

#         data_reg = data_reg.groupby('time.month').mean('time').mean(dim='lon')
#         max_models = []

#         for i in range(len(data_reg.month)):
#             ts = data_reg[i, :]
#             max = ts.max()
#             ind_max = np.max(np.where(ts == max)[0])
#             max_models.append(np.array(data_reg.lat[ind_max]))
#         max_models_all1.append(max_models)

        # # corr = np.corrcoef(np.array(max_obs), np.array(max_models))[0,1] ## corr
        # corr_all1.append(corr)
        # rmse1 = rmse(np.array(max_obs), np.array(max_models))
        # rmse_all1.append(rmse1)
        # print('itcz_rmse2', rmse(np.array(max_obs), np.array(max_models)))
        # print('itcz_corr', np.corrcoef(np.array(max_obs), np.array(max_models))[0,1])




# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip6', rmse_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip5', rmse_all1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip6', corr_all)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip5', corr_all1)


#-----------------------------------------------------------------------------------------------------------------------
#### plotting results as boxplots:::::::
#-----------------------------------------------------------------------------------------------------------------------

markers1 = ['o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*', 'o', 'v', 'p', 'd', 's','*', ]
colors1 = ['red','red','red','red','red','red','green', 'green', 'green', 'green','green', 'green' ,'skyblue', 'skyblue', 'skyblue', 'skyblue','skyblue', 'skyblue','magenta', 'magenta', 'magenta', 'magenta', 'magenta' ]
colors2 = ['orange','orange', 'orange', 'orange', 'orange','orange','gold', 'gold' ,'gold', 'gold', 'gold','gold','pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'cyan','cyan', 'cyan', 'cyan', 'cyan' ]


ticks = ['CMIP6', 'CMIP5',]
q=0.2
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color= color)
    plt.setp(bp['medians'], color=color, linewidth=2)


f, ax = plt.subplots(1,2,figsize=(7,5),gridspec_kw={'wspace': 0.3,})

axes = ax.flatten()

## WAF
# DJF=max_obs
# DJF_all = max_models_all
# DJF_all1 = max_models_all1
# c1=(np.corrcoef(DJF,np.mean(DJF_all,0)))[0,1] ## correlation betweem obs and CMIP6 MMM
# c2=(np.corrcoef(DJF,np.mean(DJF_all1,0)))[0,1] ## correlation between obs and CMIP6 MMM
# x=np.arange(0,12,1)
#     # axes[0].plot((DJF_all)[i],c='grey')
# axes[0].plot((np.mean(DJF_all,0)),c='k',label='CMIP6-MMM', linewidth=3.5, marker ='o',markersize=8)## plot CMIP6 MMM with corcoeff
# axes[0].plot((np.mean(DJF_all1,0)),c='r',label='CMIP5-MMM', linewidth=3.5, marker ='d',markersize=8, linestyle='--')## plot CMIP6 MMM with corcoeff
# axes[0].fill_between(x,(np.percentile(DJF_all,5,0)), (np.nanpercentile(DJF_all,95,0)),facecolor='grey',alpha=0.4)
# axes[0].fill_between(x,(np.percentile(DJF_all,5,0)), (np.nanpercentile(DJF_all1,95,0)),facecolor='lightcoral',alpha=0.3)
# axes[0].plot(DJF,c='b',lw=3, label='REGEN', linestyle='--',linewidth=3.5, marker ='*',markersize=8)
# axes[0].set_ylabel('Latitude (degrees)',fontsize=12, weight='bold')
# # axes[0].set_title('(a)',fontsize=12,weight='bold')
# axes[0].legend(loc='upper left', prop={'size': 7,'weight':'bold'})
# axes[0].grid(linestyle='--')
# axes[0].set_xticks(x)
# axes[0].set_xticklabels(ticks, fontsize=8, weight= 'bold')
# for axis in ['top','bottom','left','right']:
#       axes[0].spines[axis].set_linewidth(3)
# axes[0].set_xticklabels(['JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP','OCT', 'NOV', 'DEC'], fontsize=7, weight='bold')

## plotting boxplot for RMSE
models_6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip6.npy')
m_all = []
for i in range(len(models_6)):
    m_all.append(models_6[i].strip('.nc'))
mod = m_all

# print(models_6)


models_5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip5.npy')
m_all = []
for i in range(len(models_5)):
    m_all.append(models_5[i].strip('.nc')+'*')
models1 = m_all


rmse_all = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_rmse_cmip6.npy')
rmse_all1=  np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_rmse_cmip5.npy')
corr_all = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_cor_cmip6.npy')
corr_all1=  np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_cor_cmip5.npy')
axes[0] = axes[0]
a = rmse_all
# print(rmse_all)
b = rmse_all1
data_a= [a, b,]

#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
# print(len(x), len(colors1), len(models))
for i in range(len(mod)):
    print(mod[i])
    lab = mod[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=30,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    axes[0].scatter(-0.2, np.mean(rmse_all), s=100,marker= '*', color = 'b',)

 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
print(x)
for i in range(len(models1)):
    lab = models1[i].strip('.nc')
    print('b',models1[i])
    bpp = axes[0].scatter(x[i], rmse_all1[i], s=30,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab)
    axes[0].scatter(0.2, np.mean(rmse_all1), s=100,marker= '*', color = 'g',)


set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)
axes[0].set_title('(d)', fontsize=8, weight='bold')
axes[0].set_xticklabels(ticks, fontsize=8, weight='bold')
axes[0].grid(linestyle = '--', linewidth=1.5,)


for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)
axes[0].set_xticklabels(ticks, fontsize=8, weight= 'bold')
# axes[0].set_ylim(0.2,1)
axes[0].set_ylabel('RMSE (mm)',fontsize=8,weight='bold')








#***********************************************************************************
## plotting boxplot for correlation coefficient 
axes[0] = axes[1]
a = corr_all
b = corr_all1
data_a= [a, b,]


#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
# print(len(x), len(colors1), len(models))
for i in range(len(mod)):
    lab = mod[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=30,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    axes[0].scatter(-0.2, np.mean(corr_all), s=100,marker= '*', color = 'b',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
print(x)
for i in range(len(models1)):
    lab = models1[i].strip('.nc')
    bpp = axes[0].scatter(x[i], corr_all1[i], s=30,marker= markers1[i], color = colors2[i], edgecolor='k', label=lab)
    axes[0].scatter(0.2, np.mean(corr_all1), s=100,marker= '*', color = 'g',)


set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
# axes[0].set_ylim(0.2,0.8)
axes[0].set_ylabel('Correlation Coefficient',fontsize=8,weight='bold')
axes[0].grid(linestyle = '--', linewidth=1.5,)
axes[0].set_title('(e)', fontsize=8, weight='bold')

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)
axes[0].set_xticklabels(ticks, fontsize=8, weight= 'bold')
plt.subplots_adjust(wspace=1, hspace=0.5,left=0.1,top=0.9,right=0.9,bottom=0.25)
axes[0].legend(bbox_to_anchor=(-0.15, -0.39),loc='lower center',ncol=6,prop={'size': 6,'weight':'bold'})

plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/ITCZ_ALL_ANNUAL_v2')
