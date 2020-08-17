

## this script will compute seasonal cycle over analysis for African srex regions 

## Author Mustapha Adamu

## 11-05-2020



## Code analyse historical and projected precipitation and temperature over Africa

## Author: Mustapha Adamu

## 08-06-2020

## importing Libraries

#*************************************************************
import xarray as xr
import numpy as np
import sys
import os
import cartopy.crs as ccrs  # This a library for making 2D spatial plots in python
import matplotlib
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import pandas as pd
from cartopy.util import add_cyclic_point
import numpy as np
import scipy
import cartopy as cart
import cmocean.cm as cmo
from math import sqrt
from sklearn.metrics import mean_squared_error
import regionmask
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# PROPERGATE NC ATTRIIIBUTES
xr.set_options(keep_attrs=True, enable_cftimeindex=True)
#******************************************************************




# Empty arrays to store results for selected regions during SSP2-4.5 and 5-8.5
WAF_hist = []

SAF_hist = []

SAH_hist = []

EAF_hist = []


WAF_fut = []

SAF_fut = []

SAH_fut = []

EAF_fut = []



## setting path for data CMIIP6

data_path1 = "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/miss"

## setting path for data CMIP5

data_path2 = '/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/miss'

## get time from this file 


## ******************************** FOR REGEN DATASET ***********************************
regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/reg_mon.nc').p

## get arrays for lon and lat use for regridding models
lon = regen_data.lon
lat = regen_data.lat

mask = regionmask.defined_regions.srex.mask(regen_data) ### for region masking :::::::::::::


## select African here from the global precipitation data
data_reg = regen_data
data_reg= data_reg.sel(time=slice('1950-01', '2005-12')) # select to match historical CMIP6
data_reg = data_reg.groupby('time.month').mean('time')



WAF_reg =data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:].mean(dim=['lon','lat'])


## North Africa
NAF_reg =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:].mean(dim=['lon','lat'])


## East Africa
EAF_reg =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:].mean(dim=['lon','lat'])

## S Africa
SAF_reg =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:].mean(dim=['lon','lat'])

#******************************************************************************************************************
#******************************************************************************************************************


#****************************************** FOR CRU DATASET ********************************************************
regen_data = xr.open_dataset('/g/data/w35/ma9839/cru/cru_all_obs/cru_1x1.nc').pre



## select African here from the global precipitation data
data_reg = regen_data/31
# data = data_reg.sel(time=data_reg.time.dt.month.isin([12,1,2]))
data_reg= data_reg.sel(time=slice('1950-01', '2005-12')) # select to match historical CMIP6
data_reg = data_reg.groupby('time.month').mean('time')

# WAF
WAF_cru =data_reg.where(mask == regionmask.defined_regions.srex.map_keys('WAF'))[:].mean(dim=['lon','lat'])


## North Africa
NAF_cru =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAH'))[:].mean(dim=['lon','lat'])


## East Africa
EAF_cru =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('EAF'))[:].mean(dim=['lon','lat'])

## S Africa
SAF_cru =  data_reg.where(mask == regionmask.defined_regions.srex.map_keys('SAF'))[:].mean(dim=['lon','lat'])
#*****************************************************************
#*****************************************************************

time_6 = xr.open_dataset('/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/ACCESS-CM2.nc').time
time_5 = xr.open_dataset('/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/ACCESS1-0.nc').time







WAF_hist = []

SAF_hist = []

SAH_hist = []

EAF_hist = []


WAF_fut = []

SAF_fut = []

SAH_fut = []

EAF_fut = []





## loading modules:::

models = sorted((os.listdir(data_path1))) # list all the data in the model
# #
# np.save('models_cmi6', models[1:])
# #
for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid wrong netcdf files\
       continue
    # print(models)

    files_CMIP6 = (glob.glob(data_path1 + "/" + models[m]))

    print(files_CMIP6)


    files_CMIP5 = (glob.glob(data_path2 + "/" + models[m]))

    for data in files_CMIP6:  # find files in  folder

        #** Grad model dataset


        #************************************** For CMIP6***********************************

        dset = xr.open_dataset(data).pr * 86400  ## convert rainfall data into daily mm/day

        mask = regionmask.defined_regions.srex.mask(dset)  ## creating region mask
        
        dset_JJA_hist = dset.sel(time=slice('1950-01', '2005'))  ## selecting needed years
        
        dset_JJA_hist = dset_JJA_hist.groupby('time.month').mean('time') ## calculating annual cycle
        dset_JJA_hist = dset_JJA_hist.where(dset_JJA_hist>0)





        ## selecting regions::: time series 

        WAF =dset_JJA_hist.where(mask == regionmask.defined_regions.srex.map_keys('WAF')).mean(dim=['lon','lat'])
        WAF_hist.append(np.array(WAF))


        SAF =dset_JJA_hist.where(mask == regionmask.defined_regions.srex.map_keys('SAF')).mean(dim=['lon','lat'])
        SAF_hist.append(np.array(SAF))


        EAF =dset_JJA_hist.where(mask == regionmask.defined_regions.srex.map_keys('EAF')).mean(dim=['lon','lat'])
        EAF_hist.append(EAF)


        SAH =dset_JJA_hist.where(mask == regionmask.defined_regions.srex.map_keys('SAH')).mean(dim=['lon','lat'])
        SAH_hist.append(SAH)



        


models = sorted((os.listdir(data_path2))) # list all the data in the model


for m in range(len(models)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models[m].startswith('.'): # get rid of missing data\
       continue
    # print(models)

    files_CMIP5 = (glob.glob(data_path2 + "/" + models[m]))

    print(files_CMIP5)

    for data in files_CMIP5:  # find files in  folder

    #** Grad model dataset

    ## selecting time series for Future scenario

        dset = xr.open_dataset(data).pr * 86400  
      #   dset = xr.DataArray(dset, coords=dict(time=time_5[:len(dset)], lat = lat, lon=lon), dims=['time','lat', 'lon'])

        dset_JJA = dset.sel(time=slice('1950-01', '2005-12'))
        dset_JJA = dset_JJA.groupby('time.month').mean('time')
        dset_JJA = dset_JJA.where(dset_JJA>0)


    
       
        # dset = xr.DataArray(dset[:(len(time))], coords={'time':time[:len(dset)],'lat': dset.lat, 'lon': dset.lon},dims=['time','lat', 'lon'])

        mask = regionmask.defined_regions.srex.mask(dset)


        ## selecting regions
        WAF =dset_JJA.where(mask == regionmask.defined_regions.srex.map_keys('WAF')).mean(dim=['lon','lat'])
        print(WAF)
        WAF_fut.append(WAF)



        SAF =dset_JJA.where(mask == regionmask.defined_regions.srex.map_keys('SAF')).mean(dim=['lon','lat'])
        SAF_fut.append(SAF)


        EAF =dset_JJA.where(mask == regionmask.defined_regions.srex.map_keys('EAF')).mean(dim=['lon','lat'])
        EAF_fut.append(EAF)


        SAH =dset_JJA.where(mask == regionmask.defined_regions.srex.map_keys('SAH')).mean(dim=['lon','lat'])
        SAH_fut.append(SAH)



## calculating MMM and 95% confidence interval

SAF_hist_5 = np.nanpercentile(SAF_hist,q=5,axis=0)
WAF_hist_5 = np.nanpercentile(WAF_hist,q=5,axis=0)
EAF_hist_5 = np.nanpercentile(EAF_hist,q=5,axis =0)
SAH_hist_5 = np.nanpercentile(SAH_hist,q=5,axis =0)


SAF_hist_95 = np.nanpercentile(SAF_hist,q=95,axis=0)
WAF_hist_95 = np.nanpercentile(WAF_hist,q=95,axis=0)
EAF_hist_95 = np.nanpercentile(EAF_hist,q=95,axis=0)
SAH_hist_95 = np.nanpercentile(SAH_hist,q=95,axis=0)



SAF_hist = np.nanmean(SAF_hist,0)

WAF_hist = np.nanmean(WAF_hist,0)

EAF_hist = np.nanmean(EAF_hist,0)

SAH_hist = np.nanmean(SAH_hist,0)


SAF_fut_5 = np.nanpercentile(SAF_fut,q=5,axis=0)
WAF_fut_5 = np.nanpercentile(WAF_fut,q=5,axis=0)
EAF_fut_5 = np.nanpercentile(EAF_fut,q=5,axis=0)
SAH_fut_5 = np.nanpercentile(SAH_fut,q=5,axis=0)


SAF_fut_95 = np.nanpercentile(SAF_fut,q=95,axis=0)
WAF_fut_95 = np.nanpercentile(WAF_fut,q=95,axis=0)
EAF_fut_95 = np.nanpercentile(EAF_fut,q=95,axis=0)
SAH_fut_95 = np.nanpercentile(SAH_fut,q=95,axis=0)

SAF_fut = np.nanmean(SAF_fut,0)
WAF_fut = np.nanmean(WAF_fut,0)
SAH_fut = np.nanmean(SAH_fut,0)
EAF_fut = np.nanmean(EAF_fut,0)

 


print(' Begin plotting ')


f, ax = plt.subplots(2,2,figsize=(15,8 ) ,sharex='all',gridspec_kw={'wspace': 0.25, 'hspace': 0.30})
axes = ax.flatten()


x = np.arange(1,13, 1)

## ploting individual time series 
axes[0].plot(x, WAF_hist,'k-',label='CMIP6-MMM', linewidth=3.5, marker ='o',markersize=8 )
axes[0].fill_between(x,WAF_hist_5, WAF_hist_95, color = 'grey',alpha=0.2)
axes[0].plot(x, WAF_fut,'r-',label='CMIP5-MMM', linewidth=3.5, marker ='d',markersize=8 ,linestyle='--')
axes[0].fill_between(x,WAF_fut_5, WAF_fut_95, color = 'lightcoral', alpha=0.4)
axes[0].plot(x, WAF_reg,'b-',label='REGEN', linewidth=3.5, marker ='*',markersize=8 )
axes[0].plot(x, WAF_cru,'g-',label='CRU', linewidth=3.5, marker ='^',markersize=8,linestyle='--' )
axes[0].set_title('WAF     (a)',weight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['JAN', 'FEB', 'MAR','APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],weight='bold')
axes[0].grid(linestyle = '--')
axes[0].set_xlim(1,12)
axes[0].legend()
axes[0].set_ylabel('Precip (mm)',fontsize =12, weight='bold')
axes[0].set_yticks([0,2,4,6,8,10])
axes[0].set_yticklabels([0,2,4,6,8,10])
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)



#***************************************************************************************
# ### for SAF
axes[0] = axes[3]
axes[0].plot(x, SAF_hist,'k-',label='CMIP6-MMM', linewidth=3.5, marker ='o',markersize=8 )
axes[0].fill_between(x,SAF_hist_5, SAF_hist_95, color = 'grey',alpha=0.2)
axes[0].plot(x, SAF_fut,'r-',label='CMIP5-MMM', linewidth=3.5, marker ='d',markersize=8 ,linestyle='--')
axes[0].fill_between(x,SAF_fut_5, SAF_fut_95, color = 'lightcoral', alpha=0.4)
axes[0].plot(x, SAF_reg,'b-',label='REGEN', linewidth=3.5, marker ='*',markersize=8 )
axes[0].plot(x, SAF_cru,'g-',label='CRU', linewidth=3.5, marker ='^',markersize=8, linestyle='--' )
axes[0].set_title('SAF     (d)',weight='bold',)
axes[0].set_xticklabels(['JAN', 'FEB', 'MAR','APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],weight='bold')
axes[0].grid(linestyle = '--')
axes[0].set_xticks(x)
axes[0].set_xlim(1,12)
axes[0].set_yticks([0,2,4,6,8,10])
axes[0].set_yticklabels([0,2,4,6,8,10])

axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)



# ### for EAF

axes[0] = axes[1]
axes[0].plot(x, EAF_hist,'k-',label='CMIP6-MMM', linewidth=3.5, marker ='o',markersize=8 )
axes[0].fill_between(x,EAF_hist_5, EAF_hist_95, color = 'grey',alpha=0.2)
axes[0].plot(x, EAF_fut,'r-',label='CMIP5-MMM', linewidth=3.5, marker ='d',markersize=8, linestyle='--' )
axes[0].fill_between(x,EAF_fut_5, EAF_fut_95, color = 'lightcoral', alpha=0.4)
axes[0].plot(x, EAF_reg,'b-',label='REGEN', linewidth=3.5, marker ='*',markersize=8 )
axes[0].plot(x, EAF_cru,'g-',label='CRU', linewidth=3.5, marker ='^',markersize=8 ,linestyle='--')
# axes[0].set_ylim(0,180)
axes[0].set_xlim(1,12)
axes[0].set_title('EAF     (b)',weight='bold')
axes[0].set_xticklabels(['JAN', 'FEB', 'MAR','APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],weight='bold')
axes[0].grid(linestyle = '--')
axes[0].set_xticks(x)
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)
axes[0].set_yticks([0,2,4,6,8,10])
axes[0].set_yticklabels([0,2,4,6,8,10])

for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)





### for SAH

axes[0] = axes[2]
axes[0].plot(x, SAH_hist*10,'k-',label='CMIP6-MMM', linewidth=3.5, marker ='o',markersize=8 )
axes[0].fill_between(x,SAH_hist_5*10, SAH_hist_95*10, color = 'grey',alpha=0.2)
axes[0].plot(x, SAH_fut*10,'r-',label='CMIP5-MMM', linewidth=3.5, marker ='d',markersize=8,linestyle='--' )
axes[0].fill_between(x,SAH_fut_5*10, SAH_fut_95*10, color = 'lightcoral', alpha=0.4)
axes[0].plot(x, NAF_reg*10,'b-',label='REGEN', linewidth=3.5, marker ='*',markersize=8 )
axes[0].plot(x, NAF_cru*10,'g-',label='CRU', linewidth=3.5, marker ='^',markersize=8 ,linestyle='--')
axes[0].set_title('SAH     (c)',weight='bold')
axes[0].set_ylabel('Precip (mm)',fontsize =12, weight='bold')
axes[0].set_xticklabels(['JAN', 'FEB', 'MAR','APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],weight='bold')
axes[0].grid(linestyle = '--')
axes[0].set_xticks(x)
axes[0].set_xlim(1,12)
axes[0].set_ylim(0,10)
axes[0].set_yticks([0,2,4,6,8,10])
axes[0].set_yticklabels([0,.2,.4,.6,.8,1])
axes[0].tick_params(axis = 'both', which = 'major', labelsize = 10)
for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(2)

plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/Regional_annual_cycle_v3')

print('Success: Image is save as png')
