## plotting heatmaps data to compare metrics for mustongo project 

## Author : Mustapha adamu 

## date  : 25/07/2020

import sys
import os, glob
import xarray as xr
import matplotlib.pyplot as plt  # Also for plotting in python
plt.switch_backend('agg')
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
from scipy import stats



## import models names and strip . nc and add * to CMIP5 names 


models_6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip6.npy')
m_all = []
for i in range(len(models_6)):
    m_all.append(models_6[i].strip('.nc'))
models_6 = m_all

# print(models_6)


models_5 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/models_cmip5.npy')
m_all = []
for i in range(len(models_5)):
    m_all.append(models_5[i].strip('.nc')+'*')
models_5 = m_all



##************************************  Annual RMSE  ********************

Ann_rmse_6 =  1-np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_rmse_cmip6.npy')
Ann_rmse_5 =  1-np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_rmse_cmip5.npy')


Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))
print(f'mean_all {np.percentile(Ann_rmse_6,50)}')
print(f'mean_all {np.percentile(Ann_rmse_5,50)}')
print(f'mean_all {np.mean(Ann_rmse_5)}')

Ann_rmse_n_6 =  ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))  ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6))) 
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))
# print(f'rmse_mean_6 {np.mean(Ann_rmse_n_6)}')
# print(f'rsme_mean_5 {np.mean(Ann_rmse_n_5)}')

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

Ann_rmse_val  = list(Ann_rmse_all.values())
Ann_rmse_val.append(np.mean(Ann_rmse_n_6))
Ann_rmse_val.append(np.mean(Ann_rmse_n_5))

# Ann_rmse_val = 1- np.array(Ann_rmse_val)
models_keys  =  list(Ann_rmse_all.keys())

models_keys.append('CMIP6_MMM')
models_keys.append('CMIP5_MMM')

print(len(Ann_rmse_val), len(models_keys))


##************************************  Annual correlation  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
## subtract one from each correlation so that values are directly comparable to other metrics
Ann_rmse_6 = np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_cor_cmip6.npy')
# print(f'corr_mean_6 {1-(Ann_rmse_6)}')
Ann_rmse_5 = (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/Ann_cor_cmip5.npy'))

# print(f'corr_mean_5 {1-(Ann_rmse_5)}')
print(f'mean_rmse {np.mean(Ann_rmse_6)}')
print(f'mean_all {np.mean(Ann_rmse_5)}')
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1  ##  normalize (remove mean and divide by std of all ens)
print(f'corr_mean_6 {np.mean(Ann_rmse_n_6)}')
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1
print(f'corr_mean_5 {np.mean(Ann_rmse_n_5)}')
all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

Ann_corr_val  = list(Ann_rmse_all.values())
Ann_corr_val.append(np.mean(Ann_rmse_n_6))
Ann_corr_val.append(np.mean(Ann_rmse_n_5))





##************************************  zonal RMSE  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip6.npy')
Ann_rmse_5 = 1-np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_rmse_cmip5.npy')
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))
print(f'mean_lat_rmse {np.mean(Ann_rmse_6)}')
print(f'mean_lat_rmse {np.mean(Ann_rmse_5)}')

Ann_rmse_n_6 =  ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =   ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))
# print(f'corr_mean_6 {np.mean(Ann_rmse_n_6)}')
# print(f'corr_mean_5 {np.mean(Ann_rmse_n_5)}')

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

zonal_rmse_val  = list(Ann_rmse_all.values())
zonal_rmse_val.append(np.mean(Ann_rmse_n_6))
zonal_rmse_val.append(np.mean(Ann_rmse_n_5))







##************************************  zonal Corr  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/lat_cor_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1  ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

zonal_cor_val  = list(Ann_rmse_all.values())
zonal_cor_val.append(np.mean(Ann_rmse_n_6))
zonal_cor_val.append(np.mean(Ann_rmse_n_5))





##************************************  WAF RMSE  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_rmse_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_rmse_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

WAF_rmse  = list(Ann_rmse_all.values())
WAF_rmse.append(np.mean(Ann_rmse_n_6))
WAF_rmse.append(np.mean(Ann_rmse_n_5))





##************************************  EAF RMSE  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_rmse_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_rmse_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

EAF_rmse  = list(Ann_rmse_all.values())
EAF_rmse.append(np.mean(Ann_rmse_n_6))
EAF_rmse.append(np.mean(Ann_rmse_n_5))




##************************************  SAH RMSE  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_rmse_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_rmse_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAH_rmse  = list(Ann_rmse_all.values())
SAH_rmse.append(np.mean(Ann_rmse_n_6))
SAH_rmse.append(np.mean(Ann_rmse_n_5))



##************************************  SAH RMSE  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_rmse_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_rmse_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAF_rmse  = list(Ann_rmse_all.values())
SAF_rmse.append(np.mean(Ann_rmse_n_6))
SAF_rmse.append(np.mean(Ann_rmse_n_5))



##************************************  WAF DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-  (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_djf_cmip6.npy'))
Ann_rmse_5 = 1-  (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_djf_cmip5.npy'))
# print(f'waf djf_mean_6 {(np.mean(Ann_rmse_6))}')
# print(f'waf djf_mean_5 {(np.mean(Ann_rmse_5))}')

Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

WAF_ovl_djf  = list(Ann_rmse_all.values())
WAF_ovl_djf.append(np.mean(Ann_rmse_n_6))
WAF_ovl_djf.append(np.mean(Ann_rmse_n_5))

##************************************  EAF DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_djf_cmip6.npy'))
Ann_rmse_5 = 1-  (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_djf_cmip5.npy'))
# print(f'eaf djf_mean_6 {(np.mean(Ann_rmse_6))}')
# print(f'eaf djf_mean_5 {(np.mean(Ann_rmse_5))}')

Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))*-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =   ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

EAF_ovl_djf  = list(Ann_rmse_all.values())
EAF_ovl_djf.append(np.mean(Ann_rmse_n_6))
EAF_ovl_djf.append(np.mean(Ann_rmse_n_5))


##************************************  SAH DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_djf_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_djf_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))*-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAH_ovl_djf  = list(Ann_rmse_all.values())
SAH_ovl_djf.append(np.mean(Ann_rmse_n_6))
SAH_ovl_djf.append(np.mean(Ann_rmse_n_5))



##************************************  SAF DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_djf_cmip6.npy'))
Ann_rmse_5 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_djf_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAF_ovl_djf  = list(Ann_rmse_all.values())
SAF_ovl_djf.append(np.mean(Ann_rmse_n_6))
SAF_ovl_djf.append(np.mean(Ann_rmse_n_5))





##************************************  WAF MAM OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =  1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_mam_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_mam_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))*-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

WAF_ovl_mam  = list(Ann_rmse_all.values())
WAF_ovl_mam.append(np.mean(Ann_rmse_n_6))
WAF_ovl_mam.append(np.mean(Ann_rmse_n_5))

##************************************  EAF MAM OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_mam_cmip6.npy'))
Ann_rmse_5 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_mam_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))*-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

EAF_ovl_mam  = list(Ann_rmse_all.values())
EAF_ovl_mam.append(np.mean(Ann_rmse_n_6))
EAF_ovl_mam.append(np.mean(Ann_rmse_n_5))


##************************************  SAH MAM OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_mam_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_mam_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAH_ovl_mam  = list(Ann_rmse_all.values())
SAH_ovl_mam.append(np.mean(Ann_rmse_n_6))
SAH_ovl_mam.append(np.mean(Ann_rmse_n_5))



##************************************  SAF MAM OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_mam_cmip6.npy'))
Ann_rmse_5 =1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_mam_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))*-1 

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAF_ovl_mam  = list(Ann_rmse_all.values())
SAF_ovl_mam.append(np.mean(Ann_rmse_n_6))
SAF_ovl_mam.append(np.mean(Ann_rmse_n_5))





##************************************  WAF JJA OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_jja_cmip6.npy'))
Ann_rmse_5 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_jja_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))
Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

# print(f'eaf jja_mean_6 {(np.mean(Ann_rmse_n_6))}')
# print(f'eaf jja_mean_5 {(np.mean(Ann_rmse_n_5))}')
all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

WAF_ovl_jja  = list(Ann_rmse_all.values())
WAF_ovl_jja.append(np.mean(Ann_rmse_n_6))
WAF_ovl_jja.append(np.mean(Ann_rmse_n_5))

##************************************  EAF DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_jja_cmip6.npy'))
Ann_rmse_5 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_jja_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

EAF_ovl_jja  = list(Ann_rmse_all.values())
EAF_ovl_jja.append(np.mean(Ann_rmse_n_6))
EAF_ovl_jja.append(np.mean(Ann_rmse_n_5))


##************************************  SAH DJF OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_jja_cmip6.npy'))
Ann_rmse_5 =  1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_jja_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 =((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1
print(f'sah jja_mean_6 {(np.mean(Ann_rmse_n_6))}')
print(f'sah jja_mean_5 {(np.mean(Ann_rmse_n_5))}')
all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAH_ovl_jja  = list(Ann_rmse_all.values())
SAH_ovl_jja.append(np.mean(Ann_rmse_n_6))
SAH_ovl_jja.append(np.mean(Ann_rmse_n_5))
# print(f'sah1 jja_mean_6 {(np.mean(Ann_rmse_n_6))}')
# print(f'sah1 jja_mean_5 {(np.mean(Ann_rmse_n_5))}')


##************************************  SAF JJA OVL  ********************
## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_jja_cmip6.npy'))
Ann_rmse_5 =1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_jja_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAF_ovl_jja  = list(Ann_rmse_all.values())
SAF_ovl_jja.append(np.mean(Ann_rmse_n_6))
SAF_ovl_jja.append(np.mean(Ann_rmse_n_5))






##************************************  WAF SON OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/WAF_ovl_son_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

WAF_ovl_son  = list(Ann_rmse_all.values())
WAF_ovl_son.append(np.mean(Ann_rmse_n_6))
WAF_ovl_son.append(np.mean(Ann_rmse_n_5))

##************************************  EAFSON OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =   1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip6.npy'))
Ann_rmse_5 =  1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/EAF_ovl_son_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

EAF_ovl_son  = list(Ann_rmse_all.values())
EAF_ovl_son.append(np.mean(Ann_rmse_n_6))
EAF_ovl_son.append(np.mean(Ann_rmse_n_5))


##************************************  SAH SON OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip6.npy'))
Ann_rmse_5 = 1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAH_ovl_son_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1 ##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 = ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all))  *-1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAH_ovl_son  = list(Ann_rmse_all.values())
SAH_ovl_son.append(np.mean(Ann_rmse_n_6))
SAH_ovl_son.append(np.mean(Ann_rmse_n_5))



##************************************  SAF SON OVL  ********************

## change the name of the most important variable leaving the rest as is RSME-Corr
Ann_rmse_6 =1- (np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip6.npy'))
Ann_rmse_5 =  1-(np.load('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Data/SAF_ovl_son_cmip5.npy'))
Ann_rmse_all = np.concatenate((Ann_rmse_6, Ann_rmse_5))
print('t_tetst', stats.ttest_ind(Ann_rmse_6, Ann_rmse_5))

Ann_rmse_n_6 = ((Ann_rmse_6)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) *-1##  normalize (remove mean and divide by std of all ens)
Ann_rmse_n_5 =  ((Ann_rmse_5)- np.mean(Ann_rmse_all))/(np.std(Ann_rmse_all)) * -1

all_6 = dict(zip(np.array(models_6), np.array(Ann_rmse_n_6)))
all_5 = dict(zip(np.array(models_5), np.array(Ann_rmse_n_5)))

Ann_rmse_all  = {}
Ann_rmse_all.update(all_6)
Ann_rmse_all.update(all_5)
Ann_rmse_all =  OrderedDict(sorted(Ann_rmse_all.items(), key=lambda t: t))

SAF_ovl_son  = list(Ann_rmse_all.values())
SAF_ovl_son.append(np.mean(Ann_rmse_n_6))
SAF_ovl_son.append(np.mean(Ann_rmse_n_5))

All_metrics = [Ann_rmse_val,Ann_corr_val,zonal_rmse_val,zonal_cor_val,WAF_rmse, EAF_rmse,
        SAH_rmse, SAF_rmse,WAF_ovl_djf, EAF_ovl_djf,
        SAH_ovl_djf, SAF_ovl_djf,WAF_ovl_mam, EAF_ovl_mam,
        SAH_ovl_mam,SAF_ovl_mam, WAF_ovl_jja,EAF_ovl_jja,SAH_ovl_djf, SAF_ovl_jja,WAF_ovl_son, EAF_ovl_son,
        SAH_ovl_son, SAF_ovl_son]

All_metrics_mean = np.mean(All_metrics, 0)

#******************************* plotting *********************************


data = {'Annual RMSE':Ann_rmse_val,'Annual CORR':Ann_corr_val,'Zonal RMSE':zonal_rmse_val,'Zonal CORR':zonal_cor_val,'WAF RMSE':WAF_rmse, 'EAF RMSE':EAF_rmse,
        'SAH RMSE':SAH_rmse, 'SAF RMSE':SAF_rmse,'WAF DJF OVL':WAF_ovl_djf, 'EAF DJF OVL':EAF_ovl_djf,
        'SAH DJF OVL':SAH_ovl_djf, 'SAF DJF OVL':SAF_ovl_djf,'WAF MAM OVL':WAF_ovl_mam, 'EAF MAM OVL':EAF_ovl_mam,
        'SAH MAM OVL':SAH_ovl_mam, 'SAF MAM OVL':SAF_ovl_mam, 'WAF JJA OVL':WAF_ovl_jja, 'EAF JJA OVL':EAF_ovl_jja,
        'SAH JJA OVL':SAH_ovl_jja, 'SAF JJA OVL':SAF_ovl_jja, 'WAF SON OVL':WAF_ovl_son, 'EAF SON OVL':EAF_ovl_son,
        'SAH SON OVL':SAH_ovl_son, 'SAF SON OVL':SAF_ovl_son, 'All metrics mean':All_metrics_mean}


data = {'Annual RMSE':Ann_rmse_val,'Annual CORR':Ann_corr_val,'Zonal RMSE':zonal_rmse_val,'Zonal CORR':zonal_cor_val,'WAF RMSE':WAF_rmse, 'EAF RMSE':EAF_rmse,
        'SAH RMSE':SAH_rmse, 'SAF RMSE':SAF_rmse,'SAH DJF OVL':SAH_ovl_djf, 'SAH JJA OVL':SAH_ovl_jja, 'SAH MAM OVL':SAH_ovl_mam, 'SAH SON OVL':SAH_ovl_son,
        'WAF DJF OVL':WAF_ovl_djf, 'WAF JJA OVL':WAF_ovl_jja, 'WAF MAM OVL':WAF_ovl_mam, 'WAF SON OVL':WAF_ovl_son,
        'EAF DJF OVL':EAF_ovl_djf, 'EAF JJA OVL':EAF_ovl_jja, 'EAF MAM OVL':EAF_ovl_mam, 'EAF SON OVL':EAF_ovl_son,
        'SAF DJF OVL':SAF_ovl_djf, 'SAF JJA OVL':SAF_ovl_jja, 'SAF MAM OVL':SAF_ovl_mam, 'SAF SON OVL':SAF_ovl_son,'All metrics mean':All_metrics_mean}

l_lab = ['RMSE_ANNUAL','RMSE_SEASONAL','CORRELATION','WAF_OVL','SAH_OVL','EAF_OVL','SAF_OVL','AVERAGE_ALL']


f, ax = plt.subplots(figsize=(15, 7))



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


df = pd.DataFrame(data ,index=models_keys)

df = df.transpose()

# Draw a heatmap with the numeric values in each cell
levs = range(10)
assert len(levs) % 2 == 0, 'N levels must be even.'

plt.show()
f, ax = plt.subplots(figsize=(20, 15),constrained_layout=True)
# sns.plotting_context()
sns.heatmap(df,vmin=-2, vmax=2, cmap=sns.color_palette("RdBu_r", 11), ax=ax,square=True,cbar_kws={"shrink": 0.63,},linewidths=2,linecolor="grey")
ax.set_xticklabels(models_keys, fontsize=14,weight='bold')

ax.tick_params(axis='both', which='major', labelsize=15)
plt.setp(ax.get_yticklabels(), fontweight="bold")



# plt.savefig('/g/data/w35/ma9839/Africa_Project/Mustongo_project/Results/Heatmap_result_v1')

plt.show()









