#!/usr/bin/env python
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dateutil import rrule
import datetime
from datetime import timedelta
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
import numpy as np
import numpy.ma as ma
import os
# from mpl_toolkits import basemap
# import ESMF
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
# from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# import shapefile
# import shapely.geometry
import math
from scipy.stats import gaussian_kde
from math import radians, cos, sin, asin, sqrt
# from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import csv
import os.path
import matplotlib.gridspec as gridspec
import scipy
import matplotlib.path as mplPath
import calendar 
from calendar import monthrange
from calendar import isleap
    
from numpy import linspace, meshgrid
from scipy.interpolate import griddata

import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from cartopy import config
import cartopy.crs as ccrs
import geopandas as gpd
import cartopy.feature as cfeature

from matplotlib.cm import ScalarMappable

import random
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.signal import detrend
from scipy.signal import correlate
from scipy.stats.mstats import theilslopes
from scipy.stats import kendalltau

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xarray as xr 
import seaborn as sns

dStartDay=datetime.datetime(1990, 1, 1,0)
dStopDay=datetime.datetime(2022, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

        
# Save Latitude, Longitude and Height in arrays
rgsERAdata='/glade/campaign/collections/rda/data/d633000/e5.oper.invariant/197901/'
sERAconstantFields='/glade/campaign/collections/rda/data/d633000/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat=np.squeeze(ncid.variables['latitude'][:])
rgrLon=np.squeeze(ncid.variables['longitude'][:])
rgrHeight=(np.squeeze(ncid.variables['Z'][:]))/9.81
ncid.close()

rgiSize=rgrHeight.shape # all lon grids have to be -180 to 180 degree
rgrLonERA=np.asarray(([rgrLon,]*rgrLat.shape[0]))
rgrLatERA=np.asarray(([rgrLat,]*rgrLon.shape[0])).transpose()
rgi180=np.where(rgrLonERA > 180)
rgrLonERA[rgi180]=rgrLonERA[rgi180]-360.

# check if we only want to read a subregion
rgiDomain=None
if rgiDomain != None:
    iloW=rgiDomain[3]
    iloE=rgiDomain[1]
    ilaN=rgiDomain[0]
    ilaS=rgiDomain[2]
else:
    iloW=rgrLonERA.shape[1]
    iloE=0
    ilaN=0
    ilaS=rgrLonERA.shape[0]
    
rgsModelVars=['CAPEmax',
              'SRH03',
              'VS03',
              'FLH',
              'CINmax',
              'SRH06',
              'VS06',
              'Dew_T',
              'TT',
              'RH_850',
              'RH_500']

# All variables 
rgsVarsAct=(np.array([rgsModelVars[ll].split('-') for ll in range(len(rgsModelVars))]))
rgsERAVariables = np.array([item for sublist in rgsVarsAct for item in sublist])

#------------------------------------------------
# read NCDF files containing the hail predictors (11 predictors in total) 
#------------------------------------------------

rgrERAVarall=np.zeros((len(rgdTimeDD), len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]), dtype=np.float32) # , dtype=np.float32
iyear=0
#loop over years 
for yy in range(len(iYears)):
    iday=0
    outfile = '/glade/derecho/scratch/bblanc/ERA5_hail_model/ERA5_dailymax/ERA5_new_predictors_' + str(iYears[yy])+ '.npz'
    yearlength=365 + isleap(iYears[yy])
    if os.path.exists(outfile):
        print (f'The file already exists: {iYears[yy]}')
        data_tmp = np.load(outfile)
        rgrERAVarall[iyear:iyear+yearlength,:,:,:] = data_tmp['rgrERAVarsyy']
        rgrLat = data_tmp['rgrLat']
        rgrLon = data_tmp['rgrLon']
        rgrHeight = data_tmp['rgrHeight']
        #rgdTimeDD = pd.to_datetime(data_tmp['rgdTimeDD'])
        iyear=iyear+yearlength
print('done')


# Same with ESSL
dStartDay=datetime.datetime(2010, 1, 1,0)
dStopDay=datetime.datetime(2022, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)
sSaveFolder="/glade/work/bblanc/HailObs/ESSL/"
sDataSet = 'ESSL'
hailsize = ['Hail', 'size']
rgrESSLObs=np.zeros((2, len(rgdTimeDD), 180, 270))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind BoM data')
        sFileName=sSaveFolder+'ESSL-Hail-StormReports_gridded-ERA5_'+str(iYears[yy])+'.nc'
#         print(sFileName)
        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonESSL=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatESSL=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrESSLObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        dd=dd+yearlength
        ncid.close()


# start with NOAA
sSaveFolder="/glade/work/bblanc/HailObs/SPC_data/ncdf_files/"
sDataSet = 'NOAA'


dStartDay=datetime.datetime(2000, 1, 1,0)
dStopDay=datetime.datetime(2021, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

hailsize = ['Hail', 'HailSize']
rgrNOAAObs=np.zeros((2, len(rgdTimeDD), 130, 300))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind NOAA data for year '+str(iYears[yy]))
        sFileName=sSaveFolder+'SPC-Hail-StormReports_gridded-75km_'+str(iYears[yy])+'.nc'

        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonNOAA=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatNOAA=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrNOAAObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        
        dd=dd+yearlength
        ncid.close()


dStartDay=datetime.datetime(1990, 1, 1,0)
dStopDay=datetime.datetime(2015, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)


# Same with BoM
sSaveFolder="/glade/work/bblanc/HailObs/BoM/HailReports-ERA5/ERA5-gridded/"
sDataSet = 'BoM'
hailsize = ['Hail', 'HailSize']
rgrBoMObs=np.zeros((2, len(rgdTimeDD), 160, 200))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind BoM data')
        sFileName=sSaveFolder+'BoM-Hail-StormReports_gridded-75km_'+str(iYears[yy])+'.nc'

        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonBoM=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatBoM=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrBoMObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        dd=dd+yearlength
        ncid.close()    



# define the input and output variables.
print(rgrERAVarall.shape)
print(rgrNOAAObs.shape)
print(rgrESSLObs.shape)
print(rgrBoMObs.shape)
indices = np.concatenate((np.arange(1370,1440), np.arange(0,200)))
dataUS = rgrERAVarall[3652:-365,:,150:280,900:1200]
dataUS = np.moveaxis(dataUS,0,1)
dataUS[:,rgrNOAAObs[0,:]==0] = np.nan
dataUS = dataUS.reshape(dataUS.shape[0],dataUS.shape[1]*dataUS.shape[2]*dataUS.shape[3])

dataEU = rgrERAVarall[7305:,:,60:240,indices]
dataEU = np.moveaxis(dataEU, 0,1)
dataEU[:,rgrESSLObs[0,:]==0]=np.nan
dataEU = dataEU.reshape(dataEU.shape[0],dataEU.shape[1]*dataEU.shape[2]*dataEU.shape[3])

dataAU = rgrERAVarall[0:-2557,:,390:550,430:630]
dataAU = np.moveaxis(dataAU,0,1)
dataAU[:,rgrBoMObs[0,:]==0] = np.nan
dataAU = dataAU.reshape(dataAU.shape[0],dataAU.shape[1]*dataAU.shape[2]*dataAU.shape[3])


rgsModelVars=['CAPE [J/kg]',
              'VS [0-6km] [m/s]',
              'TT [C]',
              'CIN [J/kg]',
              'Td [K]',
              'VS [0-3km] [m/s]',
              'FLH [m]',
              'RH [850 hPa] [%]',
              'SRH [0-3km] [m$^2$/s$^2$]',
              'RH [500 hPa] [%]',
              'SRH [0-6km] [m$^2$/s$^2$]']
              

xxlim = [[0,6000],
         [0,40],
         [35,65],
         [0,200],
         [270,300],
         [0,40],
        [0,6000],
        [0,100],
        [0,1000],
        [0,100],
        [0,1000]]

plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(24, 24))
gs1 = gridspec.GridSpec(4,3)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.3, hspace = 0.3)

aa=0
bb=0
alphabet = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)']
index = [0,6,8,4,7,2,3,9,1,10,5]
bin_min = [0,0,35,0,0,0,0,0,0,0,0]
bin_max = [6000,40,65,200,300,40,6000,100,1000,100,1000]

for ii in range(11):
    bin_size = (bin_max[ii]-bin_min[ii])/20
    bins = np.arange(bin_min[ii], bin_max[ii] + bin_size, bin_size)
    ax1 = plt.subplot(gs1[aa,bb])
    dataUStoplot = dataUS[index[ii],:]
    hailindices = ~np.isnan(dataUStoplot)
    dataUStoplot = dataUStoplot[hailindices]
    plt.hist(dataUStoplot, bins=bins,density=True, alpha=0.5,color='#003f5c', label='CONUS')
    
    dataEUtoplot = dataEU[index[ii],:]
    hailindices = ~np.isnan(dataEUtoplot)
    dataEUtoplot = dataEUtoplot[hailindices]
    plt.hist(dataEUtoplot, bins=bins,density=True, alpha=0.5,color='#bc5090', label='Europe')
    
    dataAUtoplot = np.abs(dataAU[index[ii],:])
    hailindices = ~np.isnan(dataAUtoplot)
    dataAUtoplot = dataAUtoplot[hailindices]
    plt.hist(dataAUtoplot, bins=bins,density=True, alpha=0.5,color='#ffa600', label='Australia')
    
    plt.title(f'{alphabet[ii]}    {rgsModelVars[ii]}', fontsize=20)
    # plt.ylabel('Probability Density')
    print(f'done {rgsModelVars[ii]}')
    plt.xlim(xxlim[ii])
    if ii==0:
        plt.legend()
    if bb==0: 
        plt.ylabel('Probability')

    bb = bb+1
    if bb%3==0:
        bb=0
        aa = aa+1
    
fig.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/Paper/Fig2/PDFs_revised.pdf')
