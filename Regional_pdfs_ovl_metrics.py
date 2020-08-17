

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

data_path1=  "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid/mask"    # path for CMIP6 Historical  dat


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



regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/low_res_obs/reg_mon.nc').p

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
data = data_reg.sel(time=data_reg.time.dt.month.isin([9,10,11]))
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


        dset = dset.sel(time=dset.time.dt.month.isin([9,10,11]))

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
data_path1= "/g/data/w35/ma9839/PRECIP_CMIP5/regrid/mask/"

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

        dset = dset.sel(time=dset.time.dt.month.isin([9,10,11]))
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
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip6', WAF_ovl)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip6', NAF_ovl)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip6', EAF_ovl)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip6', SAF_ovl)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip5', WAF_ovl1)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip5', NAF_ovl1)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip5', EAF_ovl1)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip5', SAF_ovl1)






fig, axes = plt.subplots(nrows=4,ncols=2, figsize=(10, 10),gridspec_kw={'wspace': 0.2, 'hspace': 0.3})

axes = axes.flatten()

import seaborn as sns

for i in range(len(WAF_all)):
    sns.distplot(WAF_all[i], ax=axes[0],hist=False,kde=True,color='grey',kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(WAF), ax=axes[0],hist=False,kde=True,color='b',kde_kws={'linewidth':5,})
sns.distplot(WAF_esm, ax=axes[0],hist=False,color='k',kde_kws={'linewidth':5,'linestyle':'--'})
sns.distplot(WAF_esm1, ax=axes[0],hist=False,color='r',kde_kws={'linewidth':4.5,'linestyle':'-.','alpha':0.6},)
axes[0].set_ylabel('PDF', fontsize=11,weight='bold')
axes[0].tick_params(axis='both', which='major', labelsize=8)
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)
axes[0].grid(linestyle='--')
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)

black_patch = mpatches.Patch(color='k', label='CMIP6-MMM')
blue_patch = mpatches.Patch(color='blue', label='REGEN')
grey_patch = mpatches.Patch(color='grey', label='CMIP6 MODELS')
red_patch = mpatches.Patch(color='red', label='CMIP5-MMM')

axes[0].legend(handles=[black_patch, blue_patch,grey_patch, red_patch], prop={'size': 6.5,'weight':'bold'})






for i in range(len(WAF_all)):
    sns.distplot(NAF_all[i], ax=axes[2],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(NAF), ax=axes[2],hist=False,color='b',kde_kws={'linewidth':5,})
sns.distplot(NAF_esm, ax=axes[2],hist=False,color='k',kde_kws={'linewidth':5,'linestyle':'--'})
sns.distplot(NAF_esm1, ax=axes[2],hist=False,color='r',kde_kws={'linewidth':5,'linestyle':'-.','alpha':0.8})
axes[2].set_ylabel('PDF', fontsize=11,weight='bold')
axes[2].tick_params(axis='both', which='major', labelsize=8)
axes[2].set_xlim(-0.05,0.05)
axes[2].grid(linestyle='--')
axes[2].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[2].spines[axis].set_linewidth(2)







for i in range(len(WAF_all)):
    sns.distplot(EAF_all[i], ax=axes[4],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(EAF), ax=axes[4],hist=False,color='b',kde_kws={'linewidth':5,})
sns.distplot(EAF_esm, ax=axes[4],hist=False,color='k',kde_kws={'linewidth':5,'linestyle':'--'})
sns.distplot(EAF_esm1, ax=axes[4],hist=False,color='r',kde_kws={'linewidth':5,'linestyle':'-.','alpha':0.8})
axes[4].set_ylabel('PDF', fontsize=11,weight='bold')
axes[4].tick_params(axis='both', which='major', labelsize=8)
axes[4].set_xlim(-2,2)
axes[4].grid(linestyle='--')
axes[4].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[4].spines[axis].set_linewidth(2)





for i in range(len(WAF_all)):
    sns.distplot(SAF_all[i], ax=axes[6],color='grey',hist=False,kde_kws={'linewidth':1,'alpha':0.7})

sns.distplot(np.array(SAF), ax=axes[6],hist=False,color='b',kde_kws={'linewidth':5,})
sns.distplot(SAF_esm, ax=axes[6],hist=False,color='k',kde_kws={'linewidth':5,'linestyle':'--'})
sns.distplot(SAF_esm1, ax=axes[6],hist=False,color='r',kde_kws={'linewidth':5,'linestyle':'-.','alpha':0.8,})
axes[6].set_ylabel('PDF', fontsize=11,weight='bold')
axes[6].tick_params(axis='both', which='major', labelsize=8)
axes[6].grid(linestyle='--')
axes[6].set_xlabel('precipitation [mm/day]', fontsize=9,weight='bold')
axes[6].set_xlim(-1.5,1.5)
axes[6].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[6].spines[axis].set_linewidth(2)








y = np.arange(0,24,1)
w=0.5
y_ticks = [0,20,40,60,80,100]
print(len(models))
for i in range(len(WAF_ovl)):
    axes[1].bar(y[i], WAF_ovl[i]*100, width=w, align='center')

axes[1].bar(22, WAF_ovl_esm*100, width=w, align='center',color='k')
axes[1].bar(23, WAF_ovl_esm1*100, width=w, align='center',color='r')
axes[1].tick_params(axis='both', which='major', labelsize=8)
axes[1].set_ylabel('%', fontsize=11,weight='bold', )
axes[1].set_xticks([])
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels(y_ticks)
axes[1].grid(linestyle='--')
axes[1].set_ylim(0,100)
axes[1].text(1.02, 0.5, "WAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[1].transAxes)

axes[1].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[1].spines[axis].set_linewidth(2)




for i in range(len(WAF_ovl)):
    print(i)
    axes[3].bar(y[i], NAF_ovl[i]*100, width=w, align='center')
    print(i,NAF_ovl[i])

axes[3].bar(22, NAF_ovl_esm*100, width=w, align='center',color='k')
axes[3].bar(23, NAF_ovl_esm1*100, width=w, align='center',color='r')
axes[3].tick_params(axis='both', which='major', labelsize=8)
axes[3].set_ylabel('%', fontsize=11,weight='bold',)
axes[3].set_xticks([])
axes[3].grid(linestyle='--')
axes[3].set_yticks(y_ticks)
axes[3].set_yticklabels(y_ticks)
axes[3].set_ylim(0,100)
axes[3].text(1.02, 0.5, "SAH",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[3].transAxes)

axes[3].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[3].spines[axis].set_linewidth(2)


for i in range(len(WAF_ovl)):
    axes[5].bar(y[i], EAF_ovl[i]*100, width=w, align='center')
axes[5].bar(22, EAF_ovl_esm*100, width=w, align='center',color='k')
axes[5].bar(23, EAF_ovl_esm1*100, width=w, align='center',color='r')
axes[5].tick_params(axis='both', which='major', labelsize=8)
axes[5].set_ylabel('%', fontsize=11,weight='bold')
axes[5].set_xticks([])
axes[5].set_ylim(0,100)
axes[5].set_yticks(y_ticks)
axes[5].set_yticklabels(y_ticks)
axes[5].grid(linestyle='--')
axes[5].tick_params(axis = 'both', which = 'major', labelsize = 10)
axes[5].text(1.02, 0.5, "EAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[5].transAxes)
for axis in ['top','bottom','left','right']:
      axes[5].spines[axis].set_linewidth(2)




for i in range(len(WAF_ovl)):
    axes[7].bar(y[i], SAF_ovl[i]*100, width=w, align='center')

axes[7].bar(22, SAF_ovl_esm*100, width=w, align='center',color='k')
axes[7].bar(23, SAF_ovl_esm1*100, width=w, align='center',color='r')
axes[7].tick_params(axis='both', which='major', labelsize=8)
axes[7].set_ylabel('%', fontsize=11,weight='bold')
axes[7].set_ylim(0,100)
axes[7].set_xticks(y)
axes[7].set_yticks(y_ticks)
axes[7].set_yticklabels(y_ticks)
axes[7].grid(axis='y',linestyle='--')
axes[7].text(1.02, 0.5, "SAF",
             rotation=90, size=12, weight='bold',
             bbox=dict(edgecolor='lightgreen', facecolor='none', pad=7, linewidth=1.5),
             ha='left', va='center', transform=axes[7].transAxes)

# models = models_cmip6
models = list(models)
models.append('CMIP6_MMM')
models.append('CMIP5_MMM')

print(len(models), len(y))
print(models[23])
print(y[23])

axes[7].set_xticklabels(models, fontsize=10,weight='bold', rotation='vertical')
axes[7].tick_params(axis = 'both', which = 'major', labelsize = 10)
plt.subplots_adjust(bottom=0.18,right=0.975,)

# plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/MAM_PDFS_OVL')



