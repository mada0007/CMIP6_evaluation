



### code to calculate errors between predicted (models) and observed metrics:

## Author Mustapha Adamu:

## 02-03-2016





import sys
import os, globe
import xarray as xr
import cartopy.crs as ccrs  # This a library for making 2D spatial plots in python
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.stats as s
import cartopy as cart
import metpy.calc as mpcalc
import cmocean.cm as cmo
from scipy.stats import genextreme as gev
import regionmask
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


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

data_path1= "/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/miss"   # path for CMIP  dat




## This is a function to compute the overlap skill score of two arrays







regen_data = xr.open_dataset('/g/data/w35/ma9839/DATA_OBS/reg_mon.nc').p


## get arrays for lon and lat use for regridding models
lon = regen_data.lon
lat = regen_data.lat

## select African here from the global precipitation data
data_reg = regen_data

## create Regional mask

mask = regionmask.defined_regions.srex.mask(data_reg)



## select time slice here
# data = data_reg.sel(time=data_reg.time.dt.month.isin([12,1,2]))
data_reg= data_reg.sel(time=slice('1950-01', '2009-12'))
data_reg = data_reg.sel(lon=slice(-20,50), lat = slice(-35,47))
data_reg = data_reg.sel(lon=slice(-20,50), lat = slice(-35,47)) ## save Africa for Hovmoeller diagram
data_reg = data_reg.groupby('time.month').mean('time')  
data_africa = data_reg.mean(dim='lon')


mean_obs = data_reg.mean(dim=['lon', 'lat'])
data_reg =  data_reg.mean(dim='lon')
print(data_reg)
max_obs = []

for i in range(len(data_reg.month)):
    ts = data_reg[i, :]
    print(ts)
    max = ts.max()
    ind_max = np.max(np.where(ts ==max)[0])
    print(ind_max)
    max_obs.append(np.array(data_reg.lat[ind_max]))
plt.plot(max_obs)
plt.show()

print(data_reg)









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

max_models_all =[]

rmse_all = []
corr_all = []

JJA_models = np.zeros(regen_data.shape) * np.nan

models = sorted((os.listdir(data_path1))) # list all the data in the model


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

        # dset = dset.sel(time=dset.time.dt.month.isin([12,1,2]))
        data_reg = dset.sel(time=slice('1950-01', '2005-12'))
        data_reg = data_reg.sel(lon=slice(-20, 50), lat=slice(-35, 47)) ## select Africa
        data_reg = data_reg.groupby('time.month').mean('time')  
        mean_model = data_reg.mean(dim=['lon', 'lat'])
        # print(len(mean_model), len(mean_obs[:len(mean_model)]))
        # corr = np.corrcoef(mean_model, mean_obs[:len(mean_model)])[0, 1] ## corr
        # corr_all.append(corr)
        # rmse1 = rmse(np.array(mean_obs[:len(mean_model)]), np.array(mean_model))
        # rmse_all.append(rmse1)
        # print(f'rmse1 {rmse1}')
        # print(f'corr {corr}')

        data_reg = data_reg.mean(dim='lon') # group by month and lon
        max_models = []

        for i in range(len(data_reg.month)):
            ts1 = data_reg[i, :] ## get lat time series
            # print(ts1)
            max = ts1.max() ## maximum over each lat
            ind_max = np.max(np.where(ts1 == max)[0]) ## get index
            print(ind_max)
            max_models.append(np.array(data_reg.lat[ind_max])) ## Append data o


        max_models_all.append(max_models) ## store full time series

#--------------------------------------------------------------------------------------------------------------------
## for CMIP5
##------------------------------------------------------------------------------------------------------------------

# data_path1= "/Volumes/G/RAWORK/Historical/regrid"


## This is a function to compute the overlap skill score of two arrays

data_path1= "/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/miss"


# create empty arrays for storing models REGIONAL  data

max_models_all1 =[] ## empty array for storing all models itcz CMIP5
rmse_all1 = []
corr_all1 = []

JJA_models = np.zeros(regen_data.shape) * np.nan

models1 = sorted((os.listdir(data_path1))) # list all the data in the model


for m in range(len(models1)):  #**** loop through all models, amip and hist data must be have same names in different folders
    # print(models)

    if  models1[m].startswith('.'): # get rid of missing data\
       continue
    # print(models)

    files = (glob.glob(data_path1 + "/" + models1[m]))
    print('cmip5', files)



    for data in files:  # find files in  folder

        #** Grad model dataset

        dset = xr.open_dataset(data).pr * 86400
        # print(dset)
        # dset = dset.sel(time=dset.time.dt.month.isin([12,1,2]))
        data_reg = dset.sel(time=slice('1950-01', '2005-12'))
        data_reg = data_reg.sel(lon=slice(-20, 50), lat=slice(-35, 47))
        data_reg = data_reg.groupby('time.month').mean('time')  
        # mean_model = data_reg.mean(dim=['lon','lat'])
        # corr = np.corrcoef(mean_model, mean_obs[:len(mean_model)])[0, 1]  ## corr
        # corr_all1.append(corr)
        # rmse1 = rmse(np.array(mean_obs[:len(mean_model)]), np.array(mean_model))
        # rmse_all1.append(rmse1)
        # print(f'rmse1 {rmse1}')
        # print(f'corr {corr}')

        data_reg = data_reg.mean(dim='lon')
        max_models = []

        for i in range(len(data_reg.month)):
            ts = data_reg[i, :]
            print(ts)
            max = ts.max()
            ind_max = np.max(np.where(ts == max)[0])
            print(ind_max)
            max_models.append(np.array(data_reg.lat[ind_max]))
        max_models_all1.append(max_models)




#-----------------------------------------------------------------------------------------------------------------------
#### plotting results as boxplots:::::::
#-----------------------------------------------------------------------------------------------------------------------


models_6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip6.npy')
m_all = []
for i in range(len(models_6)):
    m_all.append(models_6[i].strip('.nc'))
models_6 = m_all

# print(models_6)


models_5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip5.npy')
m_all = []
for i in range(len(models_5)):
    m_all.append(models_5[i].strip('.nc'))
models_5 = m_all

ticks = ['CMIP6', 'CMIP5',]
q=0.2
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color= color)
    plt.setp(bp['medians'], color=color, linewidth=2)

markers1 = ['o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*',  'o', 'v', 'p', 'd', 's','*', 'o', 'v', 'p', 'd', 's','*', ]
colors1 = ['red','red','red','red','red','red','green', 'green', 'green', 'green','green', 'green' ,'skyblue', 'skyblue', 'skyblue', 'skyblue','skyblue', 'skyblue','magenta', 'magenta', 'magenta', 'magenta', 'magenta' ]

colors2 = ['orange','orange', 'orange', 'orange', 'orange','orange','gold', 'gold' ,'gold', 'gold', 'gold','gold','pink', 'pink', 'pink', 'pink', 'pink', 'pink', 'cyan','cyan', 'cyan', 'cyan', 'cyan' ]

f, ax = plt.subplots(2,3,figsize=(12,8),gridspec_kw={'wspace': 0.2, 'hspace': 0.5},)
axes = ax.flatten()
from copy import copy
palette = copy(plt.get_cmap('plasma_r'))
palette.set_under('white', 1.0)


MMM_cmip6 = xr.open_dataset('/g/data/w35/ma9839/PREC_CMIP6/For_evaluation/regrid_1x1/mask/MMM/MMM.nc').pr * 86400
MMM_cmip6 = MMM_cmip6.groupby('time.month').mean('time')  
MMM_cmip6 = MMM_cmip6.sel(lon=slice(-20,50), lat = slice(-35,47))
MMM_cmip6_zonal = MMM_cmip6.mean(dim='lon')
CMIP6_bias = MMM_cmip6_zonal - data_africa



MMM_cmip5 = xr.open_dataset('/g/data/w35/ma9839/PRECIP_CMIP5/regrid_1x1/MMM/MMM.nc').pr * 86400
MMM_cmip5 = MMM_cmip5.groupby('time.month').mean('time') 
MMM_cmip5 = MMM_cmip5.sel(lon=slice(-20,50), lat = slice(-35,47))
MMM_cmip5_zonal = MMM_cmip5.mean(dim='lon')
CMIP5_bias = MMM_cmip5_zonal - data_africa
x=np.arange(0,12,1)


# cf = axes[0].contourf(lon, lat, JJA_MMM1, clevs, cmap=cmo.rain, extend='both')
# cs = axes[0].contour(lon, lat, mpcalc.smooth_n_point(JJA_MMM1, 9, 2), clevs, colors='k', linewidths=1)

month = CMIP6_bias.month
clevs = [0,1,2,3,4,5,6,7,8]
lat = data_africa.lat
print(data_africa.shape)
print(month)

# cf = axes[0].contourf(month, lat, np.transpose(data_africa), clevs, cmap=cmo.rain, extend='both')
# cs = axes[0].contour(month, lat, mpcalc.smooth_n_point(np.transpose(data_africa), 9, 2), clevs, colors='k', linewidths=1)

xr.plot.contourf(data_africa,
    x = 'month', y = 'lat', 
    ax=axes[0],cmap=palette,
    add_colorbar=False,add_labels=False,
    levels=[0,1,2,3,4,5,6,7,8]
    )
xr.plot.contour(data_africa,
    x = 'month', y = 'lat', 
    ax=axes[0],colors ='k',add_labels=False,
    add_colorbar=False,
    levels=[0,1,2,3,4,5,6,7,8],
    )
axes[0].set_xticks(month)
axes[0].set_title('(a)  Zonal mean (obs)',fontsize=12, weight='bold')
axes[0].set_xticklabels(['JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP','OCT', 'NOV', 'DEC'], fontsize=7, weight='bold')
axes[0].set_yticks([ -20, 0, 20, 40])
axes[0].set_ylabel('Latitude',fontsize=12, weight='bold')
lat_formatter = LatitudeFormatter()
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axes[0].yaxis.set_major_formatter(lat_formatter)


print(MMM_cmip6)

h=xr.plot.contourf(MMM_cmip6_zonal,
    x = 'month', y = 'lat', 
    ax=axes[1],cmap = palette,
    add_colorbar=False,add_labels=False,
    levels=[0,1,2,3,4,5,6,7,8]
    )

xr.plot.contour(MMM_cmip6_zonal,
    x = 'month', y = 'lat', 
    ax=axes[1],colors='k',
    add_colorbar=False,add_labels=False,
    levels=[0,1,2,3,4,5,6,7,8],
    )
axes[1].set_xticks(month)
axes[1].set_title('(b)  Zonal CMIP6(MMM)',fontsize=12, weight='bold')
axes[1].set_xticklabels(['JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP','OCT', 'NOV', 'DEC'], fontsize=7, weight='bold')

axes[1].set_yticklabels([])

cbar = f.colorbar(h,ax=[axes[0],axes[1]], shrink=0.6, orientation='horizontal',anchor =(0.5, -1.5))
cbar.set_label("mm", fontsize=7,labelpad=2,)
cbar.ax.tick_params(labelsize=8)


levels = [-5,-4.5,-3.5,-2,-1.5,0, 1.5,2,3.5,4.5,5]
h = xr.plot.contourf(CMIP6_bias,
    x = 'month', y = 'lat', 
    ax=axes[2], cmap ='RdBu',levels=levels,
    add_colorbar=False,add_labels=False,
   )

xr.plot.contour(CMIP6_bias,
    x = 'month', y = 'lat', 
    ax=axes[2],colors='k', levels=levels,
    add_colorbar=False,add_labels=False,
   )

axes[2].set_xticks(month)
axes[2].set_title('(c)  Bias (CMIP6 - Obs)',fontsize=12, weight='bold')
axes[2].set_xticklabels(['JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP','OCT', 'NOV', 'DEC'], fontsize=7, weight='bold')
axes[2].set_yticklabels([])

cbar = f.colorbar(h, ax=[axes[2]], shrink=0.8, orientation='horizontal',anchor =(0.6, -1))
cbar.set_label("mm", fontsize=7,labelpad=2, )
cbar.ax.tick_params(labelsize=8)






## WAF
axes[0] = axes[3]
DJF=max_obs
DJF_all = max_models_all
DJF_all1 = max_models_all1
print(DJF_all)
axes[0].plot((np.mean(DJF_all,0)),c='r',lw=2.5, label='CMIP6 MMM')## plot CMIP6 MMM with corcoeff
axes[0].plot((np.mean(DJF_all1,0)),c='g',lw=2.5,linestyle='--',alpha=1,label='CMIP5 MMM')## plot CMIP6 MMM with corcoeff
axes[0].fill_between(x,(np.percentile(DJF_all,5,0)), (np.nanpercentile(DJF_all,95,0)),facecolor='lightcoral',alpha=0.4)
axes[0].fill_between(x,(np.percentile(DJF_all,5,0)), (np.nanpercentile(DJF_all1,95,0)),facecolor='g',alpha=0.2)
axes[0].plot(DJF,c='b',lw=2.5, label='OBS')
axes[0].set_ylabel('Latitude',fontsize=12, weight='bold')
axes[0].set_title('(a)',fontsize=12,weight='bold')
axes[0].legend(loc='upper left', prop={'size': 7,'weight':'bold'})
axes[0].grid(linestyle='--')
axes[0].set_xticks(x)
axes[0].set_yticks([ -20,-10, 0, 10,20])
lat_formatter = LatitudeFormatter()
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axes[0].yaxis.set_major_formatter(lat_formatter)

axes[0].set_xticklabels(['JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP','OCT', 'NOV', 'DEC'], fontsize=7, weight='bold')
axes[0].set_title('(d)  Zonal max precip',fontsize=12, weight='bold')




## plotting boxplot for RMSE
axes[0] = axes[4]
a = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip6.npy')
b =  np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip5.npy')
data_a= [a, b,]

#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(models_6)):
    # print(mod[i])
    lab = models_6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=30,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)
    print(np.mean(a))


 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
print(x)
for i in range(len(models_5)):
    lab = models_5[i].strip('.nc')
    # print('b',models1[i])
    bpp = axes[0].scatter(x[i], b[i], s=30,marker= markers1[i], color = colors2[i],edgecolor='k', label=lab+'*')
    print(np.mean(b))


set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
axes[0].grid(linestyle = '--', linewidth=1.5,)


for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)
axes[0].set_xticklabels(ticks, fontsize=8, weight= 'bold')
# axes[0].set_ylim(0.2,1)
axes[0].set_title('(e)    RMSE (mm)',fontsize=12,weight='bold')
axes[0].scatter(0.2, np.mean(b), s=100,marker= '*', color = 'b',)
axes[0].scatter(-0.2, np.mean(a), s=100,marker= '*', color = 'g',)







#***********************************************************************************
## plotting boxplot for correlation coefficient 
axes[0] = axes[5]
a =  np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip6.npy')
b =  np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip5.npy')
data_a= [a, b,]


#************************************* CMIP_6 ******************************************
bpl = axes[0].boxplot(data_a, positions=np.array(range(len(data_a))) * 0.4 - 0.2, widths=q) ## position CMIP6 on the lef
x = [-0.2] * 22  # x coordinates to plot on ::::
x = x + np.random.randn(len(x)) * 0.040  ## jitter xvalues
print(len(x), len(colors1), len(models))
for i in range(len(models_6)):
    lab = models_6[i].strip('.nc') # strip .nc from model titles::::
    bpp = axes[0].scatter(x[i], a[i], s=30,marker= markers1[i], color = colors1[i],edgecolor='k', label=lab)

    # bpp.set_color(colors[i])  ## set colors for scatter plot
 


#****************************************** CMIP5 *************************************
x = [0.2] * 20
x = x + np.random.randn(len(x)) * 0.040
print(x)
for i in range(len(models_5)):
    lab = models_5[i].strip('.nc')
    bpp = axes[0].scatter(x[i], b[i], s=30,marker= markers1[i], color = colors2[i],edgecolor='k', label=lab+'*')


set_box_color(bpl, 'k')  # colors are from http://colorbrewer2.org/
axes[0].set_xticks(range(0, len(ticks) * -2, 2), ticks)

axes[0].set_xticklabels(ticks, fontsize=12, weight='bold')
# axes[0].set_ylim(0.2,0.8)

axes[0].grid(linestyle = '--', linewidth=1.5,)


for axis in ['top','bottom','left','right']:
      axes[0].spines[axis].set_linewidth(3)
axes[0].set_xticklabels(ticks, fontsize=8, weight= 'bold')
axes[0].set_title('(f)  Correlation Coefficient',fontsize=12,weight='bold')
axes[0].scatter(0.2, np.mean(b), s=100,marker= '*', color = 'b',)
axes[0].scatter(-0.2, np.mean(a), s=100,marker='*', color = 'g',)

plt.subplots_adjust(wspace=1, hspace=0.5,left=0.1,top=0.9,right=0.9,bottom=0.15)

axes[0].legend(bbox_to_anchor=(-0.5, -0.5),loc='lower center',ncol=7,prop={'size': 6,'weight':'bold'})
# f.tight_layout()
plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/ITCZ_hov_rmse_corr_v2',dpi=1000)
