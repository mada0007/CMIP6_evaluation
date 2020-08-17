




##************** code to make regional pdf and plot overlapping metric

## author Mustapha Adamu::
## date: 13-07-2020
### Code for regional analysis of CMIP6 data


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






## PATH FOR CMIP6 DATA

data_path1=  "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/miss"    # path for CMIP6 Historical  dat


# data_path1= "/Volumes/G/RAWORK/Historical/regrid"


## This is a function to compute the overlap skill score of two arrays

bins = 12
def OVL_two_random_arr(arr1, arr2, number_bins):
    ''' arr1: input vector 1

    arr2: input vector 2

    bins: number of bins

    usage example : OVL_two_random_arr( TOP10,BOT10, 100)
    '''
    # Determine the range over which the integration will occur
    min_value = np.min((arr1.min(), arr2.min()))
    max_value = np.min((arr1.max(), arr2.max()))
    # Determine the bin width
    bin_width = (max_value-min_value)/number_bins
    #For each bin, find min frequency
    lower_bound = min_value #Lower bound of the first bin is the min_value of both arrays
    min_arr = np.empty(number_bins) #Array that will collect the min frequency in each bin
    for b in range(number_bins):
        higher_bound = lower_bound + bin_width #Set the higher bound for the bin
        #Determine the share of samples in the interval
        freq_arr1 = np.ma.masked_where((arr1<lower_bound)|(arr1>=higher_bound), arr1).count()/len(arr1)
        freq_arr2 = np.ma.masked_where((arr2<lower_bound)|(arr2>=higher_bound), arr2).count()/len(arr2)
        #Conserve the lower frequency
        min_arr[b] = np.min((freq_arr1, freq_arr2))
        lower_bound = higher_bound #To move to the next range
    return min_arr.sum()






# extracting all REGEN data
# processing regen_data



regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/reg_mon.nc').p

clim = regen_data.groupby('time.month').mean('time')
# *** calculating anomalies
regen_data = regen_data.groupby('time.month') - clim

## get arrays for lon and lat use for regridding models
lon = regen_data.lon
lat = regen_data.lat

## select African here from the global precipitation data
data_reg = regen_data

## create Regional mask

mask = regionmask.defined_regions.srex.mask(data_reg)



## select time slice here
data = data_reg.sel(time=data_reg.time.dt.month.isin([6,7,8]))
data_reg= data.sel(time=slice('1950-01', '2005-12'))

print(data_reg)

mean_obs = data_reg.mean(dim='time')




# Selection regions of Africa here::


## West Africa
WAF =data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF')).mean(dim=['lon','lat'])


## North Africa
NAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH')).mean(dim=['lon','lat'])



## East Africa
EAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF')).mean(dim=['lon','lat'])
print(f'mean_EAF {np.nanmean(EAF)}')


## S Africa
SAF =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF')).mean(dim=['lon','lat'])
print(f'mean_SAF_obs {np.nanmean(SAF)}')









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




#
# ## create array for storing exceedance::

waf_all = []
saf_all = []
naf_all = []
eaf_all = []
sahel_all = []

JJA_models = np.zeros(regen_data.shape) * np.nan

models = sorted((os.listdir(data_path1))) # list all the data in the model

models_cmip6 = models
# #
# np.save('models_cmi6', models[1:])
# #
for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid of missing data\
       continue
    # print(models)

    files = (glob.glob(data_path1 + "/" + models[m]))
    print(files)



    for data in files:  # find files in  folder

        #** Grad model dataset

        dset = xr.open_dataset(data).pr * 86400

        clim = dset.groupby('time.month').mean('time')
        # *** calculating anomalies
        dset = dset.groupby('time.month') - clim


        dset = dset.sel(time=dset.time.dt.month.isin([6,7,8]))

        data_cmip = dset.sel(time=slice('1950-01', '2005-12'))


        # mean_mods = data_cmip.mean(dim='time')


        # JJA_models[m] = mean_mods


        data_reg = data_cmip
        mask = regionmask.defined_regions.srex.mask(data_reg)

        ## West Africa
        WAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:324].mean(dim=['lon','lat'])
        WAF_all.append(np.array(WAF1))

        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(WAF1, WAF, bins)
        WAF_ovl.append(ovl)



        ## North Africa
        NAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:324].mean(dim=['lon','lat'])
        NAF_all.append(np.array(NAF1))

        ovl = OVL_two_random_arr(NAF1, NAF, bins)
        NAF_ovl.append(ovl)
        print(ovl)




        ## East Africa
        EAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:324].mean(dim=['lon','lat'])
        EAF_all.append(np.array(EAF1))

        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(EAF1, EAF, bins)
        print(f'mean_EAF {np.nanmean(EAF1)}')
        EAF_ovl.append(ovl)

        ## S Africamean(dim=['lon','lat'])
        SAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:324].mean(dim=['lon','lat'])

        print(f'mean_SAF {np.nanmean(SAF1)}')

        SAF_all.append(np.array(SAF1))

        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(SAF1, SAF, bins)
        SAF_ovl.append(ovl)


# del SAF_all[7]
# del WAF_all[7]
# del EAF_all[7]
# del NAF_all[7]





WAF_esm =  [val for sublist in WAF_all for val in sublist]
NAF_esm = [val for sublist in NAF_all for val in sublist]
EAF_esm = [val for sublist in EAF_all for val in sublist]
SAF_esm = [val for sublist in SAF_all for val in sublist]




#---------------------------------------------------------------------------------------
### FOR CMIP5
data_path1= "/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/miss"

WAF_all1 = []

NAF_all1 = []

EAF_all1 = []

SAF_all1 = []

WAF_ovl1 = []

NAF_ovl1 = []

EAF_ovl1 = []

SAF_ovl1 = []

models = sorted((os.listdir(data_path1))) # list all the data in the model
models_cmip5 = models
# #
# np.save('models_cmi6', models[1:])
# #
for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid of missing data\
       continue
    # print(models)

    files = (glob.glob(data_path1 + "/" + models[m]))
    print(files)



    for data in files:  # find files in  folder

        #** Grad model dataset

        dset = xr.open_dataset(data).pr * 86400



        clim = dset.groupby('time.month').mean('time')  ### compute climatology
        # *** calculating anomalies
        dset = dset.groupby('time.month') - clim

        dset = dset.sel(time=dset.time.dt.month.isin([6,7,8]))
        data_cmip = dset.sel(time=slice('1950-01', '2005-12'))




        ## selecting regions for MODELS and apppend them to respective empty arrays::

        ## Change data_reg to match models i.e. Data_reg::

        data_reg = data_cmip
        mask = regionmask.defined_regions.srex.mask(data_reg)

        ## West Africa
        WAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF')).mean(dim=['lon','lat'])
        WAF_all1.append(np.array(WAF1))

        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(WAF1, WAF, bins)
        WAF_ovl1.append(ovl)



        ## North Africa
        NAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH')).mean(dim=['lon','lat'])
        NAF_all1.append(np.array(NAF1))


        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(NAF1, NAF, bins)
        NAF_ovl1.append(ovl)
        print(ovl)




        ## East Africa
        EAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF')).mean(dim=['lon','lat'])
        EAF_all1.append(np.array(EAF1))

        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(EAF1, EAF, bins)



        EAF_ovl1.append(ovl)

        ## S Africamean(dim=['lon','lat'])
        SAF1 = data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF')).mean(dim=['lon','lat'])


        SAF_all1.append(np.array(SAF1))
        ## Compute overlap metric and append the results
        ovl = OVL_two_random_arr(SAF1, SAF, bins)
        SAF_ovl1.append(ovl)





WAF_esm1 =  [val for sublist in WAF_all1 for val in sublist]  ## flattern list of list ::::
NAF_esm1 = [val for sublist in NAF_all1 for val in sublist]
EAF_esm1 = [val for sublist in EAF_all1 for val in sublist]
SAF_esm1 = [val for sublist in SAF_all1 for val in sublist]




# west Africa
WAF_ovl_esm = np.mean(WAF_ovl,0)

## for cmip5
WAF_ovl_esm1 = np.mean(WAF_ovl1,0)





## North
NAF_ovl_esm = np.mean(NAF_ovl,0)



## for CMIP5
NAF_ovl_esm1 = np.mean(NAF_ovl1,0)



##SAF
SAF_ovl_esm = np.mean(SAF_ovl,0)

## For CMIP5
SAF_ovl_esm1= np.mean(SAF_ovl1,0)


##Eastt
EAF_ovl_esm = np.mean(EAF_ovl,0)

## for CMIP5
EAF_ovl_esm1 = np.mean(EAF_ovl1,0)


## strip model names of .nc


m_all = []
for i in range(len(models_cmip6)):
   m = models_cmip6[i].strip('.nc')

   m_all.append(m)

models= m_all
# print((models))



## creating dictionaries for ovls and model names::::




from collections import OrderedDict
WAF_cmip6 = dict(zip(np.array(models_cmip6), np.array(WAF_ovl)))
WAF_cmip5 = dict(zip(np.array(models_cmip5), np.array(WAF_ovl1)))

dall = {}
dall.update(WAF_cmip6)
dall.update(WAF_cmip5)

# >>> print(dictionary)

# OrderedDict(sorted(d.items(), key=lambda t: t[0]))
print('dict', dict(sorted(dall.items())))


## Save all ovls as numpy arrays:::

# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip6', models_cmip6)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip5', models_cmip5)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip6', WAF_ovl)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip6', NAF_ovl)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip6', EAF_ovl)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip6', SAF_ovl)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip5', WAF_ovl1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip5', NAF_ovl1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip5', EAF_ovl1)
# np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip5', SAF_ovl1)




#************************** loading ovls **************************



#*************************************** for overlap metric *************************** 
#**************************************** DJF  ****************************************

#**************************************************************************************
WAF_djf_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_djf_cmip6.npy')
SAH_djf_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_djf_cmip6.npy')
EAF_djf_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_djf_cmip6.npy')
SAF_djf_cmip6= np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_djf_cmip6.npy')
WAF_djf_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_djf_cmip5.npy')
SAH_djf_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_djf_cmip5.npy')
EAF_djf_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_djf_cmip5.npy')
SAF_djf_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_djf_cmip5.npy')


#*******************************  JJA ***************************************


WAF_jja_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_jja_cmip6.npy')
SAH_jja_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_jja_cmip6.npy')
EAF_jja_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_jja_cmip6.npy')
SAF_jja_cmip6= np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_jja_cmip6.npy')
WAF_jja_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_jja_cmip5.npy')
SAH_jja_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_jja_cmip5.npy')
EAF_jja_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_jja_cmip5.npy')
SAF_jja_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_jja_cmip5.npy')




#*******************************  MAM ***************************************


WAF_mam_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_mam_cmip6.npy')
SAH_mam_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_mam_cmip6.npy')
EAF_mam_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_mam_cmip6.npy')
SAF_mam_cmip6= np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_mam_cmip6.npy')
WAF_mam_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_mam_cmip5.npy')
SAH_mam_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_mam_cmip5.npy')
EAF_mam_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_mam_cmip5.npy')
SAF_mam_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_mam_cmip5.npy')


#*******************************  SON **************************************
WAF_son_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip6.npy')
SAH_son_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip6.npy')
EAF_son_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip6.npy')
SAF_son_cmip6= np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip6.npy')
WAF_son_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip5.npy')
SAH_son_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip5.npy')
EAF_son_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip5.npy')
SAF_son_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip5.npy')


models_cmip6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip6.npy')

mods = []
for i in range(len(models_cmip6)):
    m = models_cmip6[i].strip('.nc')
    mods.append(m)
# mods.append('CMIP6_MMM')
models_cmip6 = mods

models_cmip5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip5.npy')


mods = []
for i in range(len(models_cmip5)):
    m = models_cmip5[i].strip('.nc')+str('*')
    mods.append(m)

# mods.append('CMIP5_MMM')
models_cmip5 = mods


fig, axes = plt.subplots(nrows=4,ncols=2, figsize=(10, 10),gridspec_kw={'wspace': 0.27, 'hspace': 0.4})

axes = axes.flatten()

import seaborn as sns

# for i in range(len(WAF_all)):
#     sns.distplot(WAF_all[i], ax=axes[0],hist=False,kde=True,color='grey',kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(WAF), ax=axes[2],hist=False,kde=True,color='b',kde_kws={'linewidth':3,})
sns.distplot(WAF_esm, ax=axes[2],hist=False,color='r',kde_kws={'linewidth':3,'linestyle':'--','alpha':0.6})
sns.distplot(WAF_esm1, ax=axes[2],hist=False,color='green',kde_kws={'linewidth':3,'linestyle':'-.','alpha':0.6},)
axes[2].set_ylabel('PDF', fontsize=11,weight='bold')
axes[2].tick_params(axis='both', which='major', labelsize=8)
axes[2].tick_params(axis = 'both', which = 'major', labelsize = 10)
axes[2].grid(linestyle='--')
axes[2].set_xlim(-1.5,1.5)
axes[2].set_title('(c)', fontsize=11, weight='bold')

axes[2].set_ylim(0.,2.5)
for axis in ['top','bottom','left','right']:
      axes[2].spines[axis].set_linewidth(2)

black_patch = mpatches.Patch(color='r', label='CMIP6-MMM')
blue_patch = mpatches.Patch(color='blue', label='REGEN')
# grey_patch = mpatches.Patch(color='grey', label='CMIP6 MODELS')
red_patch = mpatches.Patch(color='green', label='CMIP5-MMM')

axes[0].legend(handles=[black_patch, blue_patch, red_patch], prop={'size': 6.5,'weight':'bold'})
# axes[0].set_xlim(-1.3,1)

l1 = axes[2].lines[0]
l2 = axes[2].lines[1]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
axes[2].fill_between(x, y, color="grey", alpha=0.3)







# for i in range(len(WAF_all)):
#     sns.distplot(NAF_all[i], ax=axes[2],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(NAF), ax=axes[0],hist=False,color='b',kde_kws={'linewidth':3,})
sns.distplot(NAF_esm, ax=axes[0],hist=False,color='r',kde_kws={'linewidth':3,'linestyle':'--','alpha':0.6})
sns.distplot(NAF_esm1, ax=axes[0],hist=False,color='green',kde_kws={'linewidth':3,'linestyle':'-.','alpha':0.6})

axes[0].set_ylabel('PDF', fontsize=11,weight='bold')
axes[0].tick_params(axis='both', which='major', labelsize=8)
axes[0].set_xlim(-0.4,0.4)
axes[0].set_ylim(0.,9)
axes[0].grid(linestyle='--')
axes[0].set_title('(a)', fontsize=11, weight='bold')

axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)

l1 = axes[0].lines[0]
l2 = axes[0].lines[1]
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
axes[0].fill_between(x, y, color="grey", alpha=0.3)
# Get the xy data from the lines so that we can shade






# for i in range(len(WAF_all)):
#     sns.distplot(EAF_all[i], ax=axes[4],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(EAF), ax=axes[4],hist=False,color='b',kde_kws={'linewidth':3,})
sns.distplot(EAF_esm, ax=axes[4],hist=False,color='r',kde_kws={'linewidth':3,'linestyle':'--','alpha':0.6})
sns.distplot(EAF_esm1, ax=axes[4],hist=False,color='green',kde_kws={'linewidth':3,'linestyle':'-.','alpha':0.6})
axes[4].set_ylabel('PDF', fontsize=11,weight='bold')
axes[4].tick_params(axis='both', which='major', labelsize=8)
axes[4].set_xlim(-1.3,1.2)
axes[4].set_ylim(0.,2)
axes[4].set_title('(e)', fontsize=11, weight='bold')

axes[4].grid(linestyle='--')
axes[4].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[4].spines[axis].set_linewidth(2)

l1 = axes[4].lines[0]
l2 = axes[4].lines[1]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
axes[4].fill_between(x, y, color="grey", alpha=0.3)






# for i in range(len(WAF_all)):
#     sns.distplot(SAF_all[i], ax=axes[6],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(SAF), ax=axes[6],hist=False,color='b',kde_kws={'linewidth':3,})
sns.distplot(SAF_esm, ax=axes[6],hist=False,color='r',kde_kws={'linewidth':3,'linestyle':'--','alpha':0.6})
sns.distplot(SAF_esm1, ax=axes[6],hist=False,color='green',kde_kws={'linewidth':3,'linestyle':'-.','alpha':0.6,})
axes[6].set_ylabel('PDF', fontsize=11,weight='bold')
axes[6].tick_params(axis='both', which='major', labelsize=8)
axes[6].grid(linestyle='--')
axes[6].set_title('(g)', fontsize=11, weight='bold')
axes[6].set_xlabel('precipitation [mm/day]', fontsize=9,weight='bold')
axes[6].set_xlim(-0.3,0.4)
axes[6].set_ylim(0.,9)

axes[6].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[6].spines[axis].set_linewidth(2)

l1 = axes[6].lines[0]
l2 = axes[6].lines[1]

# Get the xy data from the lines so that we can shade
x1, y1 = l1.get_xydata().T
x2, y2 = l2.get_xydata().T

xmin = max(x1.min(), x2.min())
xmax = min(x1.max(), x2.max())
x = np.linspace(xmin, xmax, 100)
y1 = np.interp(x, x1, y1)
y2 = np.interp(x, x2, y2)
y = np.minimum(y1, y2)
axes[6].fill_between(x, y, color="grey", alpha=0.3)




#-----------------------------------------------------------------------------------------------------------------------
ticks = ['CMIP6', 'CMIP5','orange','orange', 'orange', 'orange' ]   ## Phase titles::::::::::::::;;;

q = 0.2       ## scale factor for positioning box plots on xaxis ::::::::::::::;;
n_colors = 22 ## total number of colors needed for plots::::::::::::::::::::::;;


#************************ simple function to set plotting resources:::::::::::::::::::::::
def set_box_color(bp, color,alpha=1):
    plt.setp(bp['boxes'], color=color,alpha=alpha,linewidth=1.5)
    plt.setp(bp['whiskers'], color=color,alpha=alpha)
    plt.setp(bp['caps'], color=color,alpha=alpha,linewidth=1.5)
    plt.setp(bp['medians'], color=color,alpha=alpha, linewidth=2.2)



#**********************************************************************************

colors = [ 'chocolate','goldenrod','red', 'brown', 'grey', 'plum', 'springgreen', 'olive', 'pink', 'magenta', 'wheat', 'darkgoldenrod',
          'k', 'darkgreen', 'teal', 'deepskyblue', 'royalblue', 'indigo', 'violet', 'cyan', 'grey', 'cadetblue', 'skyblue', 'teal']

c_keys = ['o', 'v', '^', '<', '>', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', 'P', 'X', 0, 1, 2,
          3, 4, 5, 6, 7, 8, 9, 10, 11, '4', '8', 's', 'p', '*','p', '*',  ] # Markes for plotting 

markers1 = ['o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*', 'o', 'v', 'p', 'd', 's','*', ]
colors1 = ['red','red','red','red','red','red','green', 'green', 'green', 'green','green', 'green' ,'skyblue', 'skyblue', 'skyblue', 'skyblue','skyblue', 'skyblue','magenta', 'magenta', 'magenta', 'magenta', 'magenta' ]


colors2 = ['orange','orange', 'orange', 'orange', 'orange','orange','gold', 'gold' ,'gold', 'gold', 'gold','gold','pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'cyan','cyan', 'cyan', 'cyan', 'cyan' ]

ticks = ['DJF', 'JJA', 'MAM', 'SON']

#******************************************************* WEST AFRICA AXIS1 *********************************************

axes[0] = axes[3]
a = WAF_djf_cmip6 
b = WAF_djf_cmip5 
c = WAF_jja_cmip6 
d = WAF_jja_cmip5 

e = WAF_mam_cmip6
f = WAF_mam_cmip5
g = WAF_son_cmip6
h = WAF_son_cmip5

data_a = [a,c,e, g ]   ##put all data together   ************************


data_b = [b,d,f, h ]          ##put all data together   ************************


#************************************* CMIP_6 ******************************************
# bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.4, widths=q) ## position CMIP6 on the lef

# bpr = axes[0].boxplot(data_b, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef

bpl = axes[0].boxplot(data_a , positions=[2,3.5,5,6.5],widths=0.5)
bpr = axes[0].boxplot(data_b , positions=[2.6 ,4.1,5.6,7.1], widths=0.5)

set_box_color(bpl, 'r',alpha=0.4) 
set_box_color(bpr, 'green',alpha=0.4)
axes[0].set_xticklabels(ticks)
axes[0].set_xticks([2.3, 3.8, 5.3,6.8])
axes[0].set_title('(d)    Overlap skill score', fontsize=11, weight='bold')




labels = [item.get_text() for item in axes[0].get_xticklabels()]
print(labels)

x = [2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=15,marker= 'o', color ='y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot


x = [2.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], b[i], s=15,marker= 'o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


x = [3.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], c[i], s=15,marker= 'o', color = 'y')
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

x = [4.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], d[i], s=15,marker= 'o', color = 'grey')
 

 

x = [5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], e[i], s=15,marker= 'o', color = 'y' )
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 
 

x = [5.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], f[i], s=15,marker= 'o', color = 'grey',)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [6.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], g[i], s=15,marker= 'o', color = 'y' )
    # bpp.set_color(colors[i])  ## set colors for scatter plot

 

x = [7.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], h[i], s=15,marker= 'o', color = 'grey')
    # bpp.set_color(colors[i])  ## set colors for scatter plot

axes[0].text(1.02, 0.5, "WAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[0].transAxes)

axes[0].grid(linestyle='--')
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)


axes[0].scatter(2, np.mean(a), s=80,marker= '*', color = 'b')
axes[0].scatter(2.6, np.mean(b), s=80,marker='*', color = 'k')
axes[0].scatter(3.5, np.mean(c), s=80,marker= '*', color = 'b')
axes[0].scatter(4.1, np.mean(d), s=80,marker= '*', color = 'k')
axes[0].scatter(5, np.mean(e),  s=80,marker= '*', color = 'b',)
axes[0].scatter(5.6, np.mean(f), s=80,marker= '*', color = 'k',)
axes[0].scatter(6.5, np.mean(g), s=80,marker= '*', color = 'b',)
axes[0].scatter(7.1, np.mean(h), s=80,marker= '*', color = 'k',)
axes[0].set_yticks([0.25,0.40,0.55,.70,0.85,1])
axes[0].set_yticklabels([25,40,55,70,85,100])










 #*******************************************************  SAH AXIS1 *********************************************
axes[0] = axes[1]
a = SAH_djf_cmip6
b = SAH_djf_cmip5
c = SAH_jja_cmip6
d = SAH_jja_cmip5

e = SAH_mam_cmip6
f = SAH_mam_cmip5
g = SAH_son_cmip6
h = SAH_son_cmip5

data_a = [a,c,e, g ] ##put all data together   ************************


data_b = [b,d,f, h ]             ##put all data together   ************************


#************************************* CMIP_6 ******************************************
# bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.4, widths=q) ## position CMIP6 on the lef

# bpr = axes[0].boxplot(data_b, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef

bpl = axes[0].boxplot(data_a , positions=[2,3.5,5,6.5],widths=0.5)
bpr = axes[0].boxplot(data_b , positions=[2.6 ,4.1,5.6,7.1], widths=0.5)

set_box_color(bpl, 'r',alpha=0.4) 
set_box_color(bpr, 'green',alpha=0.4)
axes[0].plot([], c='r', label='CMIP6')
axes[0].plot([], c='green', label='CMIP5')
axes[0].legend(bbox_to_anchor=(1.0, 1.3),prop={'size': 6.5,'weight':'bold'})
axes[0].set_xticklabels(ticks)
axes[0].set_xticks([2.3, 3.8, 5.3,6.8])

labels = [item.get_text() for item in axes[0].get_xticklabels()]
print(labels)

x = [2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=15,marker= 'o', color = 'y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot


x = [2.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], b[i], s=15,marker= 'o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


x = [3.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], c[i], s=15,marker= 'o', color = 'y', )
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

x = [4.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], d[i], s=15,marker= 'o', color = 'grey',)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], e[i], s=15,marker= 'o', color = 'y', )
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 
 

x = [5.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], f[i], s=15,marker= 'o', color = 'grey' )
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [6.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], g[i], s=15,marker= 'o', color = 'y', )
    # bpp.set_color(colors[i])  ## set colors for scatter plot

 

x = [7.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], h[i], s=15,marker= 'o', color = 'grey',)
    # bpp.set_color(colors[i])  ## set colors for scatter plot

axes[0].text(1.02, 0.5, "SAH",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[0].transAxes)
 
 

axes[0].grid(linestyle='--')
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)

axes[0].set_title('(b)   Overlap skill score', fontsize=11, weight='bold')

axes[0].scatter(2, np.mean(a), s=80,marker= '*', color = 'b')
axes[0].scatter(2.6, np.mean(b), s=80,marker='*', color = 'k')
axes[0].scatter(3.5, np.mean(c), s=80,marker= '*', color = 'b')
axes[0].scatter(4.1, np.mean(d), s=80,marker= '*', color = 'k')
axes[0].scatter(5, np.mean(e),  s=80,marker= '*', color = 'b')
axes[0].scatter(5.6, np.mean(f), s=80,marker= '*', color = 'k')
axes[0].scatter(6.5, np.mean(g), s=80,marker= '*', color = 'b',)
axes[0].scatter(7.1, np.mean(h), s=80,marker= '*', color = 'k',)
axes[0].set_yticks([0.25,0.40,0.55,.70,0.85,1])
axes[0].set_yticklabels([25,40,55,70,85,100])


#******************************************************* EAF AXIS1 *********************************************
axes[0] = axes[5]
a = EAF_djf_cmip6
b = EAF_djf_cmip5
c = EAF_jja_cmip6
d = EAF_jja_cmip5

e = EAF_mam_cmip6
f = EAF_mam_cmip5
g = EAF_son_cmip6
h = EAF_son_cmip5

data_a = [a,c,e, g ]  ##put all data together   ************************


data_b = [b,d,f, h ]            ##put all data together   ************************


#************************************* CMIP_6 ******************************************
# bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.4, widths=q) ## position CMIP6 on the lef

# bpr = axes[0].boxplot(data_b, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef

bpl = axes[0].boxplot(data_a , positions=[2,3.5,5,6.5],widths=0.5)
bpr = axes[0].boxplot(data_b , positions=[2.6 ,4.1,5.6,7.1], widths=0.5)

set_box_color(bpl, 'r',alpha=0.4) 
set_box_color(bpr, 'green',alpha=0.4)
axes[0].set_xticklabels(ticks)
axes[0].set_xticks([2.3, 3.8, 5.3,6.8])




labels = [item.get_text() for item in axes[0].get_xticklabels()]
print(labels)

x = [2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=15,marker= 'o', color = 'y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot


x = [2.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], b[i], s=15,marker='o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


x = [3.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], c[i], s=15,marker= 'o', color = 'y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

x = [4.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], d[i], s=15,marker= 'o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], e[i], s=15,marker= 'o', color = 'y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 
 

x = [5.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], f[i], s=15,marker= 'o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [6.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], g[i], s=15,marker= 'o', color = 'y', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot

 

x = [7.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], h[i], s=15,marker= 'o', color = 'grey', label=lab)
    # bpp.set_color(colors[i])  ## set colors for scatter plot

axes[0].scatter(2, np.mean(a), s=80,marker= '*', color = 'b')
axes[0].scatter(2.6, np.mean(b), s=80,marker='*', color = 'k')
axes[0].scatter(3.5, np.mean(c), s=80,marker= '*', color = 'b')
axes[0].scatter(4.1, np.mean(d), s=80,marker= '*', color = 'k')
axes[0].scatter(5, np.mean(e),  s=80,marker= '*', color = 'b')
axes[0].scatter(5.6, np.mean(f), s=80,marker= '*', color = 'k')
axes[0].scatter(6.5, np.mean(g), s=80,marker= '*', color = 'b')
axes[0].scatter(7.1, np.mean(h), s=80,marker= '*', color = 'k')


axes[0].text(1.02, 0.5, "EAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[0].transAxes)



axes[0].set_yticks([0.55,.70,0.85,1])
axes[0].set_yticklabels([55,70,85,100])
axes[0].grid(linestyle='--')
axes[0].set_title('(f)   Overlap skill score', fontsize=11, weight='bold')

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)



#******************************************************* SAF AXIS1 *********************************************
axes[0] = axes[7]
a = SAF_djf_cmip6
b = SAF_djf_cmip5
c = SAF_jja_cmip6
d = SAF_jja_cmip5

e = SAF_mam_cmip6
f = SAF_mam_cmip5
g = SAF_son_cmip6
h = SAF_son_cmip5

data_a = [a,c,e, g ]   ##put all data together   ************************


data_b = [b,d,f, h ]            ##put all data together   ************************


#************************************* CMIP_6 ******************************************
# bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.4, widths=q) ## position CMIP6 on the lef

# bpr = axes[0].boxplot(data_b, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef

bpl = axes[0].boxplot(data_a , positions=[2,3.5,5,6.5],widths=0.5)
bpr = axes[0].boxplot(data_b , positions=[2.6 ,4.1,5.6,7.1], widths=0.5)

set_box_color(bpl, 'r',alpha=0.4) 
set_box_color(bpr, 'green',alpha=0.4)
axes[0].set_xticklabels(ticks)
axes[0].set_xticks([2.3, 3.8, 5.3,6.8])





labels = [item.get_text() for item in axes[0].get_xticklabels()]
print(labels)

x = [2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=15,marker= 'o', color = 'y', label=lab)
    axes[0].scatter(2, np.mean(a), s=80,marker= '*', color = 'b',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot


x = [2.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], b[i], s=15,marker= 'o', color = 'grey', label=lab)
    axes[0].scatter(2.6, np.mean(b), s=80,marker= '*', color = 'k',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


x = [3.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], c[i], s=15,marker= 'o', color = 'y', label=lab)
    axes[0].scatter(3.5, np.mean(c), s=80,marker= '*', color = 'b',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

x = [4.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], d[i], s=15,marker= 'o', color = 'grey', label=lab)
    axes[0].scatter(4.1, np.mean(d), s=50,marker= '*', color = 'k',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], e[i], s=15,marker= 'o', color = 'y', label=lab)
    axes[0].scatter(5, np.mean(e), s=80,marker= '*', color = 'b',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 
 

x = [5.6] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], f[i], s=15,marker= 'o', color = 'grey', label=lab)
    axes[0].scatter(5.6, np.mean(f), s=80,marker= '*', color = 'k',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 

 

x = [6.5] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(a)):
    print(i)
    lab = models_cmip6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], g[i], s=20,marker= 'o', color = 'y', label=lab)
    axes[0].scatter(6.5, np.mean(g), s=80,marker= '*', color = 'b',)

    # bpp.set_color(colors[i])  ## set colors for scatter plot

 

x = [7.1] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(b)):
    print(i)
    lab = models_cmip5[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], h[i], s=15,marker= 'o', color = 'grey', label=lab)
    axes[0].scatter(7.1, np.mean(h), s=80,marker= '*', color = 'k')


axes[0].text(1.02, 0.5, "SAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[0].transAxes)

axes[0].set_yticks([0.25,0.40,0.55,.70,0.85,1])
axes[0].set_yticklabels([25,40,55,70,85,100])
axes[0].grid(linestyle='--')

# axes[3].legend(bbox_to_anchor=(-0.135, -4.0),loc='lower center',ncol=7,prop={'size': 7.5,'weight':'bold'})
# plt.subplots_adjust(wspace=1, hspace=0.5,left=0.1,top=0.9,right=0.9,bottom=0.2)
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)

axes[0].set_title('(h)   Overlap skill score', fontsize=11, weight='bold')



plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/PDFS_OVERLAP', dpi=1000)








