

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

data_path1=  "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/miss"   # path for CMIP6 Historical  dat
regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/reg_mon.nc').p




## get arrays for lon and lat use for regridding models
data_reg= regen_data.sel(time=slice('1950-01', '2005-12'))
obs_mean = data_reg.mean(dim='time')




lon = data_reg.lon
lat = data_reg.lat

## create Regional mask
mask = regionmask.defined_regions.srex.mask(data_reg)



## select time slice here
data_jja = data_reg.sel(time=data_reg.time.dt.month.isin([6,7,8]))
data_jja_aff = data_jja.mean(dim=['time',])


data_son = data_reg.sel(time=data_reg.time.dt.month.isin([9,10,11]))
data_son_aff = data_son.mean(dim=['time',])


data_djf = data_reg.sel(time=data_reg.time.dt.month.isin([12,1,2]))
data_djf_aff = data_djf.mean(dim=['time',])


data_mam = data_reg.sel(time=data_reg.time.dt.month.isin([3,4,5]))
data_mam_aff = data_mam.mean(dim=['time',])




# create empty arrays for storing models REGIONAL  data

DJF_all_6 = []

JJA_all_6 = []

MAM_all_6 = []

SON_all_6 = []


DJF_bias_6 = []

JJA_bias_6 = []

MAM_bias_6 = []

SON_bias_6 = []


mmm_all = []

mmm_bias = []










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
        dset = dset.sel(time=slice('1950-01', '2005-12'))
        dset = dset.where(dset>0)

        mmm = dset.mean(dim='time')
        mmm_all.append(mmm)

        mmm_b = mmm - obs_mean
        mmm_bias.append(mmm_b)


        dset_jja = dset.sel(time=dset.time.dt.month.isin([6,7,8])).mean(dim=['time'])
        bias = dset_jja - data_jja_aff
        JJA_bias_6.append(bias)
        JJA_all_6.append(dset_jja)


        dset_mam = dset.sel(time=dset.time.dt.month.isin([3,4,5])).mean(dim=['time',])
        bias = dset_mam - data_mam_aff
        MAM_all_6.append(dset_mam)
        MAM_bias_6.append(bias)


        dset_djf = dset.sel(time=dset.time.dt.month.isin([12,1,2])).mean(dim=['time',])
        bias = dset_djf - data_djf_aff
        DJF_all_6.append(dset_djf)
        DJF_bias_6.append(bias)

        dset_son = dset.sel(time=dset.time.dt.month.isin([9,10,11])).mean(dim=['time',])
        bias = dset_son - data_son_aff
        SON_all_6.append(dset_son)
        SON_bias_6.append(bias)






JJA_esm =  [val for sublist in JJA_all_6 for val in sublist]
MAM_esm = [val for sublist in MAM_all_6 for val in sublist]
SON_esm = [val for sublist in SON_all_6 for val in sublist]
DJF_esm = [val for sublist in DJF_all_6 for val in sublist]

np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/JJA_bias_all', JJA_bias_6)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/MAM_bias_all', MAM_bias_6)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/SON_bias_all', SON_bias_6)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/DJF_bias_all', DJF_bias_6)


# JJA_bias =  np.nanmean([val for sublist in JJA_bias_6 for val in sublist])
# MAM_bias = [val for sublist in MAM_bias_6 for val in sublist]
# SON_bias = [val for sublist in SON_bias_6 for val in sublist]
# DJF_bias = [val for sublist in DJF_bias_6 for val in sublist]


JJA_bias = np.nanmean(JJA_bias_6,0)
DJF_bias = np.nanmean(DJF_bias_6,0)
MAM_bias = np.nanmean(MAM_bias_6,0)
SON_bias = np.nanmean(SON_bias_6,0)


JJA_all = np.nanmean(JJA_all_6,0)
DJF_all = np.nanmean(DJF_all_6,0)
MAM_all = np.nanmean(MAM_all_6,0)
SON_all = np.nanmean(SON_all_6,0)

mmm_all = np.nanmean(mmm_all, 0)
mmm_bias = np.nanmean(mmm_bias, 0)



np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/JJA_bias1', JJA_bias)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/MAM_bias1', MAM_bias)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/SON_bias1', SON_bias)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/DJF_bias1', DJF_bias)

np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/JJA_mmm1', JJA_all)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/MAM_mmm1', MAM_all)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/SON_mmm1', SON_all)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/DJF_mmm1', DJF_all)


np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/mmm_all1', mmm_all)
np.save('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/data_to_desktop/mmm_bias1',mmm_bias)












# fig, ax = plt.subplot()


from cartopy.util import add_cyclic_point
lon_idx =regen_data.dims.index('lon')
wrap_data, wrap_lon = add_cyclic_point(JJA_bias, coord=lon)

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(projection=proj))
ax.contourf(wrap_lon, lat, wrap_data,vmin =-5, vmax=5, levels=np.arange(-5,5,1),cmap='RdBu')
ax.set_extent([-20, 51, -35, 39, ], ccrs.PlateCarree())
ax.coastlines()


plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/Spatial_test')






