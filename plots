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

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xarray as xr 
def grid(x, y, z, resX=20, resY=20):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z
########################################
#                            Settings

dStartDay=datetime.datetime(2000, 1, 1,0)
dStopDay=datetime.datetime(2021, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')
rgdFullTime_day=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='d')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

rgsLableABC=string.ascii_lowercase+string.ascii_uppercase
PlotDir='/glade/u/home/bblanc/HailModel/plots/'
# sSaveDataDir='/glade/scratch/prein/Papers/HailModel/data/V6_3-25_CAPE-FLH_CAPE_SRH03_VS03/'
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/'

sERAconstantFields='/glade/scratch/bblanc/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'

rgsModelVars_4i=['CAPEmax',
              'SRH03',
              'VS03',
              'FLH']

rgsModelVars_7n=['CINmax',
              'SRH06',
              'VS06',
              'Dew_T',
              'TT',
              'RH_850',
              'RH_500']

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

# Initial 4 variables
rgsVarsAct_4i=(np.array([rgsModelVars_4i[ll].split('-') for ll in range(len(rgsModelVars_4i))]))
rgsERAVariables_4i = np.array([item for sublist in rgsVarsAct_4i for item in sublist])

# Additional 7 variables
rgsVarsAct_7n =(np.array([rgsModelVars_7n[ll].split('-') for ll in range(len(rgsModelVars_7n))]))
rgsERAVariables_7n = np.array([item for sublist in rgsVarsAct_7n for item in sublist])

# All variables 
rgsVarsAct=(np.array([rgsModelVars[ll].split('-') for ll in range(len(rgsModelVars))]))
rgsERAVariables = np.array([item for sublist in rgsVarsAct for item in sublist])


rgsVarMaxLim=['FLH']
rgsVarMinLim=['CAPEmax','SRH03','VS03','FLH']

iRadius=6 # grid cells that are blended out around hail observations
rMinHailSize= 2.5 # this is the maximum diameter of a hail stone in cm


iCorrelPred=20 # 20 # how many 2D predictors should be used
iPosComb=scipy.math.factorial(len(rgsERAVariables))/(scipy.math.factorial(2) * scipy.math.factorial(len(rgsERAVariables)-2)) # max combinations of variables

qT=2 #3
qD=20 #25
iPercentile=np.array([qT,qD+qT])  # percentiles that are excluded from the distribution
Qtail=iPercentile[0]
QDelta=iPercentile[1]-Qtail
rgrPercSteps=np.linspace(Qtail,QDelta+Qtail,5)
rgrProbSteps=np.append(rgrPercSteps[0]-(rgrPercSteps[1]-rgrPercSteps[0]),rgrPercSteps)
rgrProb=0.5+0.5*np.tanh((rgrProbSteps-np.mean([rgrProbSteps[0],rgrProbSteps[-1]]))/((rgrProbSteps[-1]-rgrProbSteps[0])*0.3))[1:]
rgrProb[-1]=1


#-----------------------------------------------------------------------
# Read the observations from NOAA and BoM 
#-----------------------------------------------------------------------

# start with NOAA
sSaveFolder="/glade/work/bblanc/HailObs/SPC_data/ncdf_files/"
sDataSet = 'NOAA'


#RawData = None

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

#Get monthly and yearly hail observations
daily_NOAAobs = np.zeros((len(iYears), 12, rgrLonNOAA.shape[0], rgrLonNOAA.shape[1]))
for yy in range(len(iYears)):
    for mm in range(12):
        daily_NOAAobs[yy,mm,:,:] = np.sum(rgrNOAAObs[0,(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        
monthly_NOAAobs=np.mean(daily_NOAAobs, axis=0)
yearly_NOAAobs=np.sum(daily_NOAAobs, axis=1)
        
        
# Save Latitude, Longitude and Height in arrays
rgsERAdata='/glade/scratch/bblanc/ERA5_hail_model/'
sERAconstantFields='/glade/scratch/bblanc/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat=np.squeeze(ncid.variables['latitude'][:])
rgrLon=np.squeeze(ncid.variables['longitude'][:])
rgrHeight=(np.squeeze(ncid.variables['Z'][:]))/9.81
ncid.close()

rgiSize=rgrHeight.shape

# all lon grids have to be -180 to 180 degree
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
        
#Get monthly and yearly hail observations
daily_BoMobs = np.zeros((len(iYears), 12, rgrLonBoM.shape[0], rgrLonBoM.shape[1]))
for yy in range(len(iYears)):
    for mm in range(12):
        daily_BoMobs[yy,mm,:,:] = np.sum(rgrBoMObs[0,(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        
monthly_BoMobs=np.mean(daily_BoMobs, axis=0)
yearly_BoMobs=np.sum(daily_BoMobs, axis=1)


# Same with ESSL
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
        
#Get monthly and yearly hail observations
daily_ESSLobs = np.zeros((len(iYears), 12, rgrLonESSL.shape[0], rgrLonESSL.shape[1]))
for yy in range(len(iYears)):
    for mm in range(12):
        daily_ESSLobs[yy,mm,:,:] = np.sum(rgrESSLObs[0,(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        
monthly_ESSLobs=np.mean(daily_ESSLobs, axis=0)
yearly_ESSLobs=np.sum(daily_ESSLobs, axis=1)

#------------------------------------------------
# read NCDF files containing the hail predictors (11 predictors in total) 
#------------------------------------------------

rgrERAVarall=np.zeros((len(rgdTimeDD), len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]))
iyear=0
#loop over years 
for yy in range(len(iYears)):
    iday=0
    outfile = '/glade/scratch/bblanc/ERA5_hail_model/ERA5_dailymax/ERA5_new_predictors_' + str(iYears[yy])+ '.npz'
    yearlength = 365 + isleap(iYears[yy])
    
    if os.path.isfile(outfile) == False:
        rgrERAVarsyy=np.zeros((yearlength, len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]))

        #Loop over months
        for mm in range(len(iMonths)):
            # __________________________________________________________________
            # start reading the ERA data
            dDate=rgdTimeDD[0]
            monthlength=int(monthrange(iYears[yy],mm+1)[1])
            
            # Add the initial 4 variables in the array
            rgrERAdataAll=np.zeros((monthlength*24, len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1])); rgrERAdataAll[:]=np.nan
            print (str(iYears[yy])+str("%02d" % (mm+1))+'    Loading CAPE, VS03, SRH03, FLH')

            # loop over variables
            sFileName=rgsERAdata + 'ERA5-hailpredictors/' + str(iYears[yy])+str("%02d" % (mm+1))+'_ERA-5_HailPredictors_newSRH03.nc'
            ncid=Dataset(sFileName, mode='r')

            for va in range(len(rgsERAVariables_4i)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_4i[va]][:])
                rgrERAdataAll[:,va,:,:]=rgrDataTMP
            ncid.close()  
            
            # Now add the 7 new predictors in the same array for simplicity
            # loop over variables
            sFileName = rgsERAdata + 'ERA5-hailpredictors/' + str(iYears[yy])+str("%02d" % (mm+1))+'_New_RH_TT.nc'
            ncid=Dataset(sFileName, mode='r')
            print (str(iYears[yy])+str("%02d" % (mm+1))+'    CINmax, SRH06, VS06, Dew_T, TT, RH_850, RH_500')
            for va in range(len(rgsERAVariables_7n)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_7n[va]][:])
                rgrERAdataAll[:,va+4,:,:]=rgrDataTMP
            ncid.close()  

            #reshaping the monthly data into daily data to be able to extract the daily max CAPE
            dayrgrERAdataAll=rgrERAdataAll.reshape(int(monthlength),24,rgrERAdataAll.shape[1],rgrERAdataAll.shape[2],rgrERAdataAll.shape[3])
            timemax=np.argmax(dayrgrERAdataAll[:,:,0,:,:],axis=1) # take the time of maximum CAPE
            timemax2=np.zeros((dayrgrERAdataAll.shape[0],dayrgrERAdataAll.shape[1],dayrgrERAdataAll.shape[2],dayrgrERAdataAll.shape[3],dayrgrERAdataAll.shape[4]))
            timemax2[:,:,:,:,:]=timemax[:,None,None,:]
            timemax2=timemax2.astype(int)

            #Take the predictors at the time of max CAPE for each day
            rgrERAVars=np.take_along_axis(dayrgrERAdataAll,timemax2,axis=1)
            rgrERAVars=rgrERAVars[:,1,:,:,:]
            
            #save monthly data in a yearly array
            rgrERAVarsyy[iday:iday+monthlength,:,:,:]=rgrERAVars

            iday=monthlength+iday
        
        np.savez(outfile,
                rgrERAVarsyy = rgrERAVarsyy,
                rgrLat = rgrLat,
                rgrLon = rgrLon,
                rgrHeight = rgrHeight,
                rgdTimeDD = rgdTimeDD)
        #save the yearly data in an array contaning the whole time period
        rgrERAVarall[iyear:iyear+yearlength,:,:,:]=rgrERAVarsyy
        iyear=iyear+yearlength
    
    else:
        print(f'The file already exists: {iYears[yy]}')
        data_tmp = np.load(outfile)
        rgrERAVarall[iyear:iyear+yearlength,:,:,:] = data_tmp['rgrERAVarsyy']
        rgrLat = data_tmp['rgrLat']
        rgrLon = data_tmp['rgrLon']
        rgrHeight = data_tmp['rgrHeight']
        #rgdTimeDD = pd.to_datetime(data_tmp['rgdTimeDD'])
        iyear=iyear+yearlength

## Predictions back to previous years
# World
# Load the model that has been calculated 
model_xgb2 = xgb.Booster()
model_xgb2.load_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/US_XGBmodel_iteration0.jason')

iyear = 0
# Make predictions on the test dataset
for yy in range(len(iYears)):
    yearlength = 365 + isleap(iYears[yy])
    # Create regression matrices
    X_test2 = rgrERAVarall[iyear:iyear+yearlength,:,:,:]
    X_test2 = np.moveaxis(X_test2,1,0)
    input1 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3])
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test =  np.moveaxis(input1, 1, 0)
    iyear=iyear+yearlength
    xg_test = xgb.DMatrix(X_test)
    preds_test = model_xgb2.predict(xg_test)
    preds_test2 = np.copy(preds_test)
    np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_world_{iYears[yy]}.npy', preds_test2)
    print(f"Predictions made on year {iYears[yy]}")

## Previous model
## ERA 5
rgrHailSaveDirnSRH='/glade/scratch/bblanc/ERA5_hail_model/Hail_Probabilities_final/HailProbabilities-ERA-5_4_qT-4_qD-26_CONUS_test_absSRH/ff'
iyear=0
rgrTotalHailProbabilitynSRH = np.zeros((rgdTimeDD.shape[0], 721, 1440))
for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy])
    sFileName = rgrHailSaveDirnSRH+'/'+str(iYears[yy])+'_HailProbabilities_ERA-5_hourly.nc'
    ncid=Dataset(sFileName, mode='r')
    rgrTotalHailProbabilitynSRH[iyear:iyear+yearlength,:,:] = np.squeeze(ncid.variables['HailProb'])
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()


## US
# US 
# Load predicted data
XGBlHailProbabilityUS = np.zeros((len(rgdTimeDD),130,300))
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/US/'

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy]) 
    sFileFin=sSaveDataDir + 'xgboost_predictions_CONUS_'+str(iYears[yy])+ '_v2.nc'
    ncid=Dataset(sFileFin, mode='r')
    XGBlHailProbabilityUS[iyear:iyear+yearlength,:,:] = np.squeeze(ncid.variables['Hail_Probabilities'])
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()

# #-------------------------------------------------
# #plot the global hail map 
# #-------------------------------------------------

# # model trained on the CONUS 

# Create colormap for the global hail map
hail_prob_colors = [(1, 1, 1), (0.8, 0.92, 1),  (0.7, 0.85, 0.95),                  
                    (0.55, 0.75, 0.92),  (0.33, 0.58, 0.9), (0.13, 0.38, 0.8),                     
                    (0.15, 0.3, 0.65), (0.12, 0.22, 0.55), (0.0, 0.4, 0), (0.0, 0.7, 0.2),                     
                     (1.0, 1.0, 0.0),                     
                    (1.0, 0.7, 0.0), (1.0, 0.4, 0.0), (1.0, 0, 0.0),                     
                    (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)
                   ]

hail_prob_cmap = ListedColormap(hail_prob_colors)

intervals = [0.00, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.7, 1.00, 1.50, 2.0, 3.00, 4.0, 5.00, 7, 9.00, 10]
norm = mcolors.BoundaryNorm(intervals, len(intervals) - 1)

%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})

ax = plt.subplot(projection=ccrs.PlateCarree())

rgrLonNOAA_toplot = rgrLonNOAA[14:110,40:-20]
rgrLatNOAA_toplot = rgrLatNOAA[14:110,40:-20]

preds_toplot = np.sum(XGBlHailProbabilityUS[:,14:110,40:-20] >= 0.5, axis=0)/(len(iYears))
pcolormesh(rgrLonNOAA_toplot,rgrLatNOAA_toplot,preds_toplot, cmap=hail_prob_cmap, norm=norm)
ax.coastlines()
#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
# Set the ylocator to have gridlines every 5 degrees in latitude
gl.ylocator = MultipleLocator(base=10.0)

# axis style

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}


#add city names
#Denver
denlat, denlon = 39.75, -105
plt.text(denlon, denlat, 'Denver',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

#Kansa city
kanlat, kanlon = 39, -94.5
plt.text(kanlon, kanlat, 'Kansas City',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

#Colorado Springs
Colat, Colon = 38.8, -104.8
plt.text(Colon, Colat, 'Colorado springs',
         horizontalalignment='right',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

#Rapid city
Raplat, Raplon = 44.1, -103.2
plt.text(Raplon, Raplat, 'Rapid City',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())

#Dallas
Dallat, Dallon = 32.75, -96.8
plt.text(Dallon, Dallat, 'Dallas',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())


#Saint Louis
Louislat, Louislon = 38.63, -90.2
plt.text(Louislon, Louislat, 'Saint Louis',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#Oklahoma City
Oklat, Oklon = 35.5, -97.5
plt.text(Oklon, Oklat, 'Oklahoma City',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())
# Salt lake City 
SLClat, SLClon = 40.74, -111.9
plt.text(SLClon, SLClat, 'Salt lake City',
         horizontalalignment='right',
         verticalalignment='top',
         transform=ccrs.PlateCarree())
#Atlanta
atlat, atlon = 33.75, -84.4
plt.text(atlon, atlat, 'Atlanta',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#plot the dots at each city location
plt.plot([denlon,kanlon,Colon,Raplon,Dallon,Louislon,Oklon,atlon,SLClon], [denlat,kanlat,Colat,Raplat,Dallat,Louislat,Oklat,atlat,SLClat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
         )


plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/US/US_hail_map_1959-2022_v2.png')

%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})

ax = plt.subplot(projection=ccrs.PlateCarree())

rgrLonNOAA_toplot = rgrLonNOAA[14:110,40:-20]
rgrLatNOAA_toplot = rgrLatNOAA[14:110,40:-20]

pcolormesh(rgrLonNOAA_toplot,rgrLatNOAA_toplot,np.sum(rgrTotalHailProbabilitynSRH[:,164:260,940:1180] >= 0.5, axis=0)/(len(iYears)), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

#add city names
#Denver
denlat, denlon = 39.75, -105
plt.text(denlon, denlat, 'Denver',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

#Kansa city
kanlat, kanlon = 39, -94.5
plt.text(kanlon, kanlat, 'Kansas City',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

#Colorado Springs
Colat, Colon = 38.8, -104.8
plt.text(Colon, Colat, 'Colorado springs',
         horizontalalignment='right',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

#Rapid city
Raplat, Raplon = 44.1, -103.2
plt.text(Raplon, Raplat, 'Rapid City',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())

#Dallas
Dallat, Dallon = 32.75, -96.8
plt.text(Dallon, Dallat, 'Dallas',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())


#Saint Louis
Louislat, Louislon = 38.63, -90.2
plt.text(Louislon, Louislat, 'Saint Louis',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Salt lake City 
SLClat, SLClon = 40.74, -111.9
plt.text(SLClon, SLClat, 'Salt lake City',
         horizontalalignment='right',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

#Oklahoma City
Oklat, Oklon = 35.5, -97.5
plt.text(Oklon, Oklat, 'Oklahoma City',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#Atlanta
atlat, atlon = 33.75, -84.4
plt.text(atlon, atlat, 'Atlanta',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#plot the dots at each city location
plt.plot([denlon,kanlon,Colon,Raplon,Dallon,Louislon,SLClon,Oklon,atlon], [denlat,kanlat,Colat,Raplat,Dallat,Louislat,SLClat,Oklat,atlat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
         )
gl.ylocator = MultipleLocator(base=10.0)
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/US/US_predictions_original_model_1959-2022.png')

# #-------------------------------------------------
# #plot the global hail map 
# #-------------------------------------------------

# # model trained on the CONUS 

# Create colormap for the global hail map
hail_diff_colors = [(0.12, 0.22, 0.55),(0.15, 0.3, 0.65),(0.13, 0.38, 0.8), (0.33, 0.58, 0.9),(0.55, 0.75, 0.92),(0.7, 0.85, 0.95), (0.8, 0.92, 1),
                    (1, 1, 1),                                     
                    (1.0, 0.8, 0.8), (0.95, 0.7, 0.7), (0.9, 0.6, 0.6), (0.9, 0.5, 0.5), (0.8, 0.2, 0.2), (0.7, 0.1, 0.1), (0.5, 0.0, 0.0)
                   ]

hail_diff_cmap = ListedColormap(hail_diff_colors)

intervals = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
norm_diff = mcolors.BoundaryNorm(intervals, len(intervals) - 1)


fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})

ax = plt.subplot(projection=ccrs.PlateCarree())
diff = preds_toplot - np.sum(rgrTotalHailProbabilitynSRH[:,164:260,940:1180] >= 0.5, axis=0)/(len(iYears))
pcolormesh(rgrLonNOAA_toplot,rgrLatNOAA_toplot, diff, cmap='RdBu_r', vmin=-2, vmax=2)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False
gl.ylocator = MultipleLocator(base=10.0)

ax.coastlines()
plt.colorbar(ticks=np.arange(-2, 2.5, 0.5), shrink=0.5, orientation='horizontal', extend='both')
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/US/US_predictions_difference_1959-2022_v2.png')

rgrLonAct=rgrLonNOAA
rgrLatAct=rgrLatNOAA

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

#get the grid cells covered with land only
invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()
# 25°N corresponds to the border with Mexico and index 260
# 49°N corresponds to the border with Canada and index 164
# The original iLaN, iLaS are 150, 280
LSM_US = LSM[ilaN:ilaS,iloE:iloW]

# ------------------------------------------------------------------
# Create  test dataset containing cities obs
# And only for a few years so that we test on unseen data 
# Look at the observation in the main cities 
# ------------------------------------------------------------------

preds_US2 = np.copy(XGBlHailProbabilityUS)
preds_US2[preds_US2>=0.5] = 1
preds_US2[preds_US2<0.5]=0

preds_US_o = rgrTotalHailProbabilitynSRH[:,ilaN:ilaS,iloE:iloW]
preds_US_o[preds_US_o>=0.5] = 1
preds_US_o[preds_US_o<0.5] = 0

# remove grid cells with water
preds_US2[:,LSM_US<0.5] = 0
preds_US_o[:,LSM_US<0.5] = 0

# colorado springs 
rgrNOAA_predsCoSp =  np.amax(preds_US2[:,54-1:54+2,121-1:121+2], axis=(1,2))
# Denver
rgrNOAA_predsDen = np.amax(preds_US2[:,51-1:51+2,120-1:120+2], axis=(1,2))
# Kansas City 
rgrNOAA_predsKan = np.amax(preds_US2[:,54-1:54+2,162-1:162+2], axis=(1,2))
# Rapid City 
rgrNOAA_predsRap = np.amax(preds_US2[:,34-1:34+2,127-1:127+2], axis=(1,2))
# Dallas 
rgrNOAA_predsDal = np.amax(preds_US2[:,79-1:79+2,153-1:153+2], axis=(1,2))
# Atlanta
rgrNOAA_predsAtl = np.amax(preds_US2[:,75-1:75+2,202-1:202+2], axis=(1,2))
# San Francisco
rgrNOAA_predsSaF = np.amax(preds_US2[:,60-1:60+2,50-1:50+2], axis=(1,2))
# Los Angeles
rgrNOAA_predsLosA = np.amax(preds_US2[:,74-1:74+2,67-1:67+2], axis=(1,2))
# Washington
rgrNOAA_predsWas = np.amax(preds_US2[:,54-1:54+2,232-1:232+2], axis=(1,2))
# New York
rgrNOAA_predsNY = np.amax(preds_US2[:,47-1:47+2,244-1:244+2], axis=(1,2))
# Alburquerke 
rgrNOAA_predsAlb = np.amax(preds_US2[:,69-1:69+2,114-1:114+2], axis=(1,2))
# Santa Fe 
rgrNOAA_predsSant = np.amax(preds_US2[:,67-1:67+2,116-1:116+2], axis=(1,2))
# Wichita 
rgrNOAA_predsWic = np.amax(preds_US2[:,59-1:59+2,151-1:151+2], axis=(1,2))
# Fort collins 
rgrNOAA_predsForC = np.amax(preds_US2[:,48-1:48+2,120-1:120+2], axis=(1,2))
# Cheyenne 
rgrNOAA_predsChe = np.amax(preds_US2[:,45-1:45+2,121-1:121+2], axis=(1,2))
# Mineapolis 
rgrNOAA_predsMin = np.amax(preds_US2[:,30-1:30+2,167-1:167+2], axis=(1,2))
# Chicago 
rgrNOAA_predsChi = np.amax(preds_US2[:,43-1:43+2,189-1:189+2], axis=(1,2))
# Little rock 
rgrNOAA_predsLit = np.amax(preds_US2[:,71-1:71+2,171-1:171+2], axis=(1,2))
# Memphis 
rgrNOAA_predsMem = np.amax(preds_US2[:,70-1:70+2,180-1:180+2], axis=(1,2))
#Omaha 
rgrNOAA_predsOma = np.amax(preds_US2[:,45-1:45+2,156-1:156+2], axis=(1,2))
# Oklahoma City 
rgrNOAA_predsOkl = np.amax(preds_US2[:,68-1:68+2,150-1:150+2], axis=(1,2))
# Lubbock 
rgrNOAA_predsLub = np.amax(preds_US2[:,76-1:76+2,132-1:132+2], axis=(1,2))
# Saint Louis 
rgrNOAA_predsSL = np.amax(preds_US2[:,55-1:55+2,179-1:179+2], axis=(1,2))
# Salt lake City 
rgrNOAA_predsSLC = np.amax(preds_US2[:,47-1:47+2,92-1:92+2], axis=(1,2))

# Group all these observations in the same array 
rgrpredsCities_test = [rgrNOAA_predsCoSp,rgrNOAA_predsDen,rgrNOAA_predsKan,rgrNOAA_predsRap,rgrNOAA_predsDal,rgrNOAA_predsAtl,
                  rgrNOAA_predsSaF, rgrNOAA_predsLosA,rgrNOAA_predsWas,rgrNOAA_predsNY,rgrNOAA_predsAlb,rgrNOAA_predsSant,
                  rgrNOAA_predsWic,rgrNOAA_predsForC,rgrNOAA_predsChe,rgrNOAA_predsMin,rgrNOAA_predsChi,rgrNOAA_predsLit,
                  rgrNOAA_predsMem,rgrNOAA_predsOma,rgrNOAA_predsOkl,rgrNOAA_predsLub,rgrNOAA_predsSL,rgrNOAA_predsSLC]

hailpreds_cities_test = np.column_stack(rgrpredsCities_test)
# hailpreds_cities_test = hailpreds_cities_test[0:1095,:]
print(hailpreds_cities_test.shape)


# colorado springs 
rgrNOAA_predsCoSp_o =  np.amax(preds_US_o[:,54-1:54+2,121-1:121+2], axis=(1,2))
# Denver
rgrNOAA_predsDen_o = np.amax(preds_US_o[:,51-1:51+2,120-1:120+2], axis=(1,2))
# Kansas City 
rgrNOAA_predsKan_o = np.amax(preds_US_o[:,54-1:54+2,162-1:162+2], axis=(1,2))
# Rapid City 
rgrNOAA_predsRap_o = np.amax(preds_US_o[:,34-1:34+2,127-1:127+2], axis=(1,2))
# Dallas 
rgrNOAA_predsDal_o = np.amax(preds_US_o[:,79-1:79+2,153-1:153+2], axis=(1,2))
# Atlanta
rgrNOAA_predsAtl_o = np.amax(preds_US_o[:,75-1:75+2,202-1:202+2], axis=(1,2))
# San Francisco
rgrNOAA_predsSaF_o = np.amax(preds_US_o[:,59-1:59+2,50-1:50+2], axis=(1,2))
# Los Angeles
rgrNOAA_predsLosA_o = np.amax(preds_US_o[:,74-1:74+2,67-1:67+2], axis=(1,2))
# Washington
rgrNOAA_predsWas_o = np.amax(preds_US_o[:,54-1:54+2,232-1:232+2], axis=(1,2))
# New York
rgrNOAA_predsNY_o = np.amax(preds_US_o[:,47-1:47+2,244-1:244+2], axis=(1,2))
# Alburquerke 
rgrNOAA_predsAlb_o = np.amax(preds_US_o[:,69-1:69+2,114-1:114+2], axis=(1,2))
# Santa Fe 
rgrNOAA_predsSant_o = np.amax(preds_US_o[:,67-1:67+2,116-1:116+2], axis=(1,2))
# Wichita 
rgrNOAA_predsWic_o = np.amax(preds_US_o[:,59-1:59+2,151-1:151+2], axis=(1,2))
# Fort collins 
rgrNOAA_predsForC_o = np.amax(preds_US_o[:,48-1:48+2,120-1:120+2], axis=(1,2))
# Cheyenne 
rgrNOAA_predsChe_o = np.amax(preds_US_o[:,45-1:45+2,121-1:121+2], axis=(1,2))
# Mineapolis 
rgrNOAA_predsMin_o = np.amax(preds_US_o[:,30-1:30+2,167-1:167+2], axis=(1,2))
# Chicago 
rgrNOAA_predsChi_o = np.amax(preds_US_o[:,33-1:33+2,189-1:189+2], axis=(1,2))
# Little rock 
rgrNOAA_predsLit_o = np.amax(preds_US_o[:,71-1:71+2,171-1:171+2], axis=(1,2))
# Memphis 
rgrNOAA_predsMem_o = np.amax(preds_US_o[:,70-1:70+2,180-1:180+2], axis=(1,2))
#Omaha 
rgrNOAA_predsOma_o = np.amax(preds_US_o[:,45-1:45+2,156-1:156+2], axis=(1,2))
# Oklahoma City 
rgrNOAA_predsOkl_o = np.amax(preds_US_o[:,68-1:68+2,150-1:150+2], axis=(1,2))
# Lubbock 
rgrNOAA_predsLub_o = np.amax(preds_US_o[:,76-1:76+2,132-1:132+2], axis=(1,2))
# Saint Louis 
rgrNOAA_predsSL_o = np.amax(preds_US_o[:,55-1:55+2,179-1:179+2], axis=(1,2))
# Salt lake City 
rgrNOAA_predsSLC_o = np.amax(preds_US_o[:,47-1:47+2,92-1:92+2], axis=(1,2))
# Group all these observations in the same array 
rgrpredsCities_test2 = [rgrNOAA_predsCoSp_o,rgrNOAA_predsDen_o,rgrNOAA_predsKan_o,rgrNOAA_predsRap_o,rgrNOAA_predsDal_o,rgrNOAA_predsAtl_o,
                  rgrNOAA_predsSaF_o, rgrNOAA_predsLosA_o,rgrNOAA_predsWas_o,rgrNOAA_predsNY_o,rgrNOAA_predsAlb_o,rgrNOAA_predsSant_o,
                  rgrNOAA_predsWic_o,rgrNOAA_predsForC_o,rgrNOAA_predsChe_o,rgrNOAA_predsMin_o,rgrNOAA_predsChi_o,rgrNOAA_predsLit_o,
                  rgrNOAA_predsMem_o,rgrNOAA_predsOma_o,rgrNOAA_predsOkl_o,rgrNOAA_predsLub_o,rgrNOAA_predsSL_o,rgrNOAA_predsSLC_o]

hailpreds_cities_test_o = np.column_stack(rgrpredsCities_test2)
# hailpreds_cities_test = hailpreds_cities_test[0:1095,:]
print(hailpreds_cities_test_o.shape)

# colorado springs 
rgrNOAAobsCoSp = np.amax(rgrNOAAObs[0,:,54-1:54+2,121-1:121+2], axis=(1,2))
# Denver 
rgrNOAAobsDen = np.amax(rgrNOAAObs[0,:,51-1:51+2,120-1:120+2], axis=(1,2))
# Kansas City 
rgrNOAAobsKan = np.amax(rgrNOAAObs[0,:,54-1:54+2,162-1:162+2], axis=(1,2))
# Rapid City 
rgrNOAAobsRap = np.amax(rgrNOAAObs[0,:,34-1:34+2,127-1:127+2], axis=(1,2))
# Dallas 
rgrNOAAobsDal = np.amax(rgrNOAAObs[0,:,79-1:79+2,153-1:153+2], axis=(1,2))
# Atlanta
rgrNOAAobsAtl = np.amax(rgrNOAAObs[0,:,75-1:75+2,202-1:202+2], axis=(1,2))
# San Francisco
rgrNOAAobsSaF = np.amax(rgrNOAAObs[0,:,59-1:59+2,50-1:50+2], axis=(1,2))
# Los Angeles
rgrNOAAobsLosA = np.amax(rgrNOAAObs[0,:,74-1:74+2,67-1:67+2], axis=(1,2))
# Washington
rgrNOAAobsWas = np.amax(rgrNOAAObs[0,:,54-1:54+2,232-1:232+2], axis=(1,2))
# New York
rgrNOAAobsNY = np.amax(rgrNOAAObs[0,:,47-1:47+2,244-1:244+2], axis=(1,2))
# Alburquerke 
rgrNOAAobsAlb = np.amax(rgrNOAAObs[0,:,69-1:69+2,114-1:114+2], axis=(1,2))
# Santa Fe 
rgrNOAAobsSant = np.amax(rgrNOAAObs[0,:,67-1:67+2,116-1:116+2], axis=(1,2))
# Wichita 
rgrNOAAobsWic = np.amax(rgrNOAAObs[0,:,59-1:59+2,151-1:151+2], axis=(1,2))
# Fort collins 
rgrNOAAobsForC = np.amax(rgrNOAAObs[0,:,48-1:48+2,120-1:120+2], axis=(1,2))
# Cheyenne 
rgrNOAAobsChe = np.amax(rgrNOAAObs[0,:,45-1:45+2,121-1:121+2], axis=(1,2))
# Mineapolis 
rgrNOAAobsMin = np.amax(rgrNOAAObs[0,:,30-1:30+2,167-1:167+2], axis=(1,2))
# Chicago 
rgrNOAAobsChi = np.amax(rgrNOAAObs[0,:,43-1:43+2,189-1:189+2], axis=(1,2))
# Little rock 
rgrNOAAobsLit = np.amax(rgrNOAAObs[0,:,71-1:71+2,171-1:171+2], axis=(1,2))
# Memphis 
rgrNOAAobsMem = np.amax(rgrNOAAObs[0,:,70-1:70+2,180-1:180+2], axis=(1,2))
#Omaha 
rgrNOAAobsOma = np.amax(rgrNOAAObs[0,:,45-1:45+2,156-1:156+2], axis=(1,2))
# Oklahoma City 
rgrNOAAobsOkl = np.amax(rgrNOAAObs[0,:,68-1:68+2,150-1:150+2], axis=(1,2))
# Lubbock 
rgrNOAAobsLub = np.amax(rgrNOAAObs[0,:,76-1:76+2,132-1:132+2], axis=(1,2))
# Saint Louis 
rgrNOAAobsSL = np.amax(rgrNOAAObs[0,:,55-1:55+2,179-1:179+2], axis=(1,2))
# Salt lake City 
rgrNOAAobsSLC = np.amax(rgrNOAAObs[0,:,47-1:47+2,92-1:92+2], axis=(1,2))
# Group all these observations in the same array 
rgrObsCities_test = [rgrNOAAobsCoSp, rgrNOAAobsDen,
                 rgrNOAAobsKan, rgrNOAAobsRap, 
                 rgrNOAAobsDal, rgrNOAAobsAtl,
                 rgrNOAAobsSaF, rgrNOAAobsLosA, 
                 rgrNOAAobsWas, rgrNOAAobsNY, 
                 rgrNOAAobsAlb, rgrNOAAobsSant, 
                 rgrNOAAobsWic, rgrNOAAobsForC, 
                 rgrNOAAobsChe, rgrNOAAobsMin, 
                 rgrNOAAobsChi, rgrNOAAobsLit, 
                 rgrNOAAobsMem, rgrNOAAobsOma, 
                 rgrNOAAobsOkl, rgrNOAAobsLub, 
                 rgrNOAAobsSL, rgrNOAAobsSLC]

hailObs_cities_test = np.column_stack(rgrObsCities_test)
# hailObs_cities = hailObs_cities[0:1095,:]
print(hailObs_cities_test.shape)

# Create arrays 
# For CITIES IN THE US

yearmonth_cities_obs_test = np.zeros((len(iYears), 12, 24))
yearmonth_cities_pred_test = np.zeros((len(iYears), 12, 24))
yearmonth_cities_pred_test_o = np.zeros((len(iYears), 12, 24))
for yy in range(len(iYears)):
    for mm in range(12):
        # US
        yearmonth_cities_obs_test[yy,mm,:] = np.sum(hailObs_cities_test[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred_test[yy,mm,:] = np.sum(hailpreds_cities_test[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred_test_o[yy,mm,:] = np.sum(hailpreds_cities_test_o[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)

    
monthly_cities_obs_test=np.mean(yearmonth_cities_obs_test, axis=0)
yearly_cities_obs_test=np.sum(yearmonth_cities_obs_test, axis=1)

monthly_cities_pred_test=np.mean(yearmonth_cities_pred_test, axis=0)
yearly_cities_pred_test=np.sum(yearmonth_cities_pred_test, axis=1)

monthly_cities_pred_test_o=np.mean(yearmonth_cities_pred_test_o, axis=0)
yearly_cities_pred_test_o=np.sum(yearmonth_cities_pred_test_o, axis=1)
print('done')

%matplotlib inline
fig = plt.figure(figsize=(10,6))
labels = ['Colorado Springs', 'Denver', 'Kansas City', 'Rapid City', 'Dallas', 'Atlanta', 'San Franciso', 'Los Angeles', 
          'Washington', 'New York', 'Alburquerke', 'Santa Fe', 'Wichita', 'Fort Collins', ' Cheyenne', 'Mineapolis', 'Chicago', 
          'Little Rock', 'Memphis', 'Omaha', 'Oklahoma City ', 'Lubbock', 'Saint Louis', 'Salt Lake City']
labels_title = ['Colorado_Springs', 'Denver', 'Kansas_City', 'Rapid_City', 'Dallas', 'Atlanta', 'San_Franciso', 'Los_Angeles', 
                'Washington', 'New York', 'Alburquerke', 'Santa_Fe', 'Wichita', 'Fort_Collins', ' Cheyenne', 'Mineapolis', 'Chicago', 
                'Little_Rock', 'Memphis', 'Omaha', 'Oklahoma_City ', 'Lubbock', 'Saint_Louis', 'Salt_Lake_City']
months_to_label = [2, 4, 6, 8, 10, 12]  # February, April, June, August, October, December
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

for ii in range(len(labels)):
    gs= gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # plot predictions
    preds1 = monthly_cities_pred_test[:,ii]
    plt.plot(iMonths, preds1, label='Predictions xgboost', color='green')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds1)
    hailmod5 = np.percentile(yearmonth_cities_pred_test[:,:,ii], 5, axis=0)
    hailmod95 = np.percentile(yearmonth_cities_pred_test[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y1 = hailmod5
    y2 = hailmod95

    plt.grid()
    ax1.fill_between(x,y1, y2, color='green', alpha=0.3)
    plt.ylabel('Average number of hail events')

     # plot predictions
    preds2 = monthly_cities_pred_test_o[:,ii]
    plt.plot(iMonths, preds2, label='Predictions ERA 5', color='steelblue')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds2)
    hailmod5 = np.percentile(yearmonth_cities_pred_test_o[:,:,ii], 5, axis=0)
    hailmod95 = np.percentile(yearmonth_cities_pred_test_o[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y5 = hailmod5
    y6 = hailmod95

    plt.grid()
    ax1.fill_between(x,y5, y6, color='steelblue', alpha=0.3)
    plt.ylabel('Average number of hail events')


    # plot observations
    obs1 = monthly_cities_obs_test[:,ii]
    plt.plot(iMonths, obs1, label='Observations', color='orange')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_obs=np.mean(obs1)
    hailobs5 = np.percentile(yearmonth_cities_obs_test[:,:,ii], 5, axis=0)
    hailobs95 = np.percentile(yearmonth_cities_obs_test[:,:,ii], 95, axis=0)   

    y3 = hailobs5
    y4 = hailobs95
    # plt.ylabel('time')
    ax1.fill_between(x,y3, y4, color='orange', alpha=0.3)
    plt.xlabel('Month')

    plt.xticks(months_to_label, [month_names[i - 1] for i in months_to_label])
    plt.ylim(0, 3)
    plt.yticks([1,2,3])
    plt.suptitle(f'{labels[ii]}')
    plt.grid()
    plt.legend()
    plt.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/US/{labels_title[ii]}_annualUS_1990_2020_v2.png')

# US
yearmonth_UShail_obs = np.zeros((len(iYears), 12, rgrLonNOAA.shape[0], rgrLonNOAA.shape[1]))
yearmonth_UShail_pred = np.zeros((len(iYears), 12, rgrLonNOAA.shape[0], rgrLonNOAA.shape[1]))
yearmonth_UShail_pred_o = np.zeros((len(iYears), 12, rgrLonNOAA.shape[0], rgrLonNOAA.shape[1]))

predsUS = np.copy(XGBlHailProbabilityUS)
predsUS[predsUS>=0.5]=1
predsUS[predsUS<0.5]=0

# original model 
predsUS_o = rgrTotalHailProbabilitynSRH[:,ilaN:ilaS,iloE:iloW]
predsUS_o[predsUS_o>=0.5]=1
predsUS_o[predsUS_o<0.5]=0
# remove grid cells with water
predsUS[:,LSM_US<0.5] = 0
predsUS_o[:,LSM_US<0.5] = 0

rgrNOAAObs2 = rgrNOAAObs[0,:,:,:]
rgrNOAAObs2[:,LSM_US<0.5] = 0

for yy in range(len(iYears)):
    for mm in range(12):          
        # Australia
        yearmonth_UShail_obs[yy,mm,:,:] = np.sum(rgrNOAAObs2[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        yearmonth_UShail_pred[yy,mm,:,:] = np.sum(predsUS[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        yearmonth_UShail_pred_o[yy,mm,:,:] = np.sum(predsUS_o[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
    
monthly_UShail_obs=np.mean(yearmonth_UShail_obs, axis=0)
yearly_UShail_obs=np.sum(yearmonth_UShail_obs, axis=1)

monthly_UShail_pred=np.mean(yearmonth_UShail_pred, axis=0)
yearly_UShail_pred=np.sum(yearmonth_UShail_pred, axis=1)   

monthly_UShail_pred_o=np.mean(yearmonth_UShail_pred_o, axis=0)
yearly_UShail_pred_o=np.sum(yearmonth_UShail_pred_o, axis=1)   

%matplotlib inline
fig = plt.figure(figsize=(12,6))
gs= gridspec.GridSpec(1,1)
ax1 = plt.subplot(gs[0,0])
iMonths=np.unique(rgdTimeDD.month)

mean_USobs = np.mean(np.sum(monthly_UShail_obs[:,:,:], axis=(1,2)), axis=0)
mean_USpred = np.mean(np.sum(monthly_UShail_pred[:,:,:], axis=(1,2)), axis=0)
mean_USpred_o = np.mean(np.sum(monthly_UShail_pred_o[:,:,:], axis=(1,2)), axis=0)

# new xgboost model
plt.plot(iMonths, np.sum(monthly_UShail_pred[:,:,:], axis=(1,2))/mean_USpred, label='Predictions xgboost')
hailmod5 = np.percentile(np.sum(yearmonth_UShail_pred[:,:,:,:],axis=(2,3)), 5, axis=0)
hailmod95 = np.percentile(np.sum(yearmonth_UShail_pred[:,:,:,:], axis=(2,3)), 95, axis=0)   
x = np.arange(1,13,1)
y1 = hailmod5/mean_USpred
y2 = hailmod95/mean_USpred
ax1.fill_between(x,y1, y2, color='green', alpha=0.3)

# original model
plt.plot(iMonths, np.sum(monthly_UShail_pred_o[:,:,:], axis=(1,2))/mean_USpred_o, label='Original predictions')
hailmod5_o = np.percentile(np.sum(yearmonth_UShail_pred_o[:,:,:,:],axis=(2,3)), 5, axis=0)
hailmod95_o = np.percentile(np.sum(yearmonth_UShail_pred_o[:,:,:,:], axis=(2,3)), 95, axis=0)   
x = np.arange(1,13,1)
y3 = hailmod5_o/mean_USpred_o
y4 = hailmod95_o/mean_USpred_o
ax1.fill_between(x,y3, y4, color='steelblue', alpha=0.3)


plt.plot(iMonths, np.sum(monthly_UShail_obs[:,:,:], axis=(1,2))/mean_USobs, label='Observations')
plt.xlabel('time')
plt.ylabel('Normalized frequency')
# plt.title('Observations')
hailobs5 = np.percentile(np.sum(yearmonth_UShail_obs[:,:,:,:], axis=(2,3)), 5, axis=0)
hailobs95 = np.percentile(np.sum(yearmonth_UShail_obs[:,:,:,:], axis=(2,3)), 95, axis=0)   
y5 = hailobs5/mean_USobs
y6 = hailobs95/mean_USobs
plt.ylim(0, 5)
plt.yticks([1,2,3,4,5])

plt.grid()
ax1.fill_between(x,y5, y6, color='orange', alpha=0.3)
plt.legend()
plt.title('US annual cycle')
plt.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/US/US_annual_cycle_1990_2020.png')


dStartDay=datetime.datetime(2010, 1, 1,0)
dStopDay=datetime.datetime(2022, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')
rgdFullTime_day=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='d')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

rgrLonAct=rgrLonESSL
rgrLatAct=rgrLatESSL

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

# define the input and output variables.
indices = np.concatenate((np.arange(1370,1440), np.arange(0,200)))

#get the grid cells covered with land only
invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()
# The original iLaN, iLaS are 150, 280
LSM_EU = LSM[ilaN:ilaS,indices]

#EU 
# Load predicted data
XGBlHailProbabilityEU = np.zeros((len(rgdTimeDD),180,270))
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/Europe/'

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy]) 
    sFileFin=sSaveDataDir + 'xgboost_predictions_europe_'+str(iYears[yy])+ '_v2.nc'
    ncid=Dataset(sFileFin, mode='r')
    XGBlHailProbabilityEU[iyear:iyear+yearlength,:,:] = np.squeeze(ncid.variables['Hail_Probabilities'])
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()

%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())


rgrLonESSLtoplot = rgrLonESSL[20::,:]
rgrLatESSLtoplot = rgrLatESSL[20::,:]

pcolormesh(rgrLonESSLtoplot,rgrLatESSLtoplot,np.sum(XGBlHailProbabilityEU[:,20::,:]>=0.5, axis=0)/len(iYears), norm=norm, cmap=hail_prob_cmap)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

#add city names

#Paris
Parlat, Parlon = 48.85, 2.35
plt.text(Parlon, Parlat, 'Paris',
         verticalalignment='top',
         horizontalalignment='center',
         transform=ccrs.PlateCarree())

# Munich
Munlat, Munlon = 48.14, 11.56
plt.text(Munlon, Munlat, 'Munich',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

# Graz
Gralat, Gralon = 47.05, 15.43
plt.text(Gralon, Gralat, 'Graz',
         horizontalalignment='left',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Turin
Turlat, Turlon = 45.08, 7.66
plt.text(Turlon, Turlat, 'Turin',
         horizontalalignment='center',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Warsaw
Warlat, Warlon = 52.24, 21.02
plt.text(Warlon, Warlat, 'Warsaw',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Zurich
Zurlat, Zurlon = 47.38, 8.53
plt.text(Zurlon, Zurlat, 'Zurich',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Rome
Romlat, Romlon = 41.90, 12.50
plt.text(Romlon, Romlat, 'Rome',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Berlin
Berlat, Berlon = 52.51, 13.4
plt.text(Berlon, Berlat, 'Berlin',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#plot the dots at each city location
plt.plot([Parlon,Munlon,Gralon, Turlon,Warlon, Zurlon,Romlon,Berlon], [Parlat,Munlat,Gralat, Turlat,Warlat, Zurlat,Romlat, Berlat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
       )


#add title
fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'
# fig.show()
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/EU/EU_hail_map_xgboost_2016-2022_v2.png')


%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

pcolormesh(rgrLonESSL,rgrLatESSL,np.sum(rgrTotalHailProbabilitynSRH[:,60:240,indices] >= 0.5, axis=0)/(len(iYears)), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

#add city names

#Paris
Parlat, Parlon = 48.85, 2.35
plt.text(Parlon, Parlat, 'Paris',
         verticalalignment='top',
         horizontalalignment='center',
         transform=ccrs.PlateCarree())

# Munich
Munlat, Munlon = 48.14, 11.56
plt.text(Munlon, Munlat, 'Munich',
         verticalalignment='bottom',
         transform=ccrs.PlateCarree())

# Graz
Gralat, Gralon = 47.05, 15.43
plt.text(Gralon, Gralat, 'Graz',
         horizontalalignment='left',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Turin
Turlat, Turlon = 45.08, 7.66
plt.text(Turlon, Turlat, 'Turin',
         horizontalalignment='center',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Warsaw
Warlat, Warlon = 52.24, 21.02
plt.text(Warlon, Warlat, 'Warsaw',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Zurich
Zurlat, Zurlon = 47.38, 8.53
plt.text(Zurlon, Zurlat, 'Zurich',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Rome
Romlat, Romlon = 41.90, 12.50
plt.text(Romlon, Romlat, 'Rome',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Berlin
Berlat, Berlon = 52.51, 13.4
plt.text(Berlon, Berlat, 'Berlin',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#plot the dots at each city location
plt.plot([Parlon,Munlon,Gralon, Turlon,Warlon, Zurlon,Romlon,Berlon], [Parlat,Munlat,Gralat, Turlat,Warlat, Zurlat,Romlat, Berlat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
       )

#add title
fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/EU/EU_predictions_original_model_1959-2022.png')


# Create colormap for the global hail map
hail_diff_colors = [(0.12, 0.22, 0.55),(0.15, 0.3, 0.65),(0.13, 0.38, 0.8), (0.33, 0.58, 0.9),(0.55, 0.75, 0.92),(0.7, 0.85, 0.95), (0.8, 0.92, 1),
                    (1, 1, 1),                                     
                    (1.0, 0.8, 0.8), (0.95, 0.7, 0.7), (0.9, 0.6, 0.6), (0.9, 0.5, 0.5), (0.8, 0.2, 0.2), (0.7, 0.1, 0.1), (0.5, 0.0, 0.0)
                   ]

hail_diff_cmap = ListedColormap(hail_diff_colors)

intervals = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
norm_diff = mcolors.BoundaryNorm(intervals, len(intervals) - 1)

rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

diff = np.sum(XGBlHailProbabilityEU >= 0.5, axis=0)/(len(iYears)) - np.sum(rgrTotalHailProbabilitynSRH[:,60:240,indices] >= 0.5, axis=0)/(len(iYears))
pcolormesh(rgrLonESSL,rgrLatESSL, diff, cmap='RdBu_r', vmin=-2, vmax=2)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='both') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/EU/EU_predictions_difference_1959-2022.png')

#-------------------------------------------------
#plot the hail days per months in the major cities (Europe)
#-------------------------------------------------

rgrLonAct=rgrLonESSL
rgrLatAct=rgrLatESSL

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

# define the input and output variables.
indices = np.concatenate((np.arange(1370,1440), np.arange(0,200)))

invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()
# The original iLaN, iLaS are 150, 280
LSM_EU = LSM[ilaN:ilaS,indices]


preds_EU =np.copy(XGBlHailProbabilityEU)
preds_EU[preds_EU>=0.5] = 1
preds_EU[preds_EU<0.5] = 0

preds_EU_o = rgrTotalHailProbabilitynSRH[:,ilaN:ilaS,indices]
preds_EU_o[preds_EU_o>=0.5] = 1
preds_EU_o[preds_EU_o<0.5] = 0

# remove grid cells with water
preds_EU[:,LSM_EU<0.5] = 0
preds_EU_o[:,LSM_EU<0.5] = 0


# Munich
rgrESSL_predsMun =  np.amax(preds_EU[:,107-1:107+2,116-1:116+2], axis=(1,2))
# Warsaw
rgrESSL_predsWar = np.amax(preds_EU[:,91-1:91+2,154-1:154+2], axis=(1,2))
# Graz
rgrESSL_predsGra = np.amax(preds_EU[:,112-1:112+2,132-1:132+2], axis=(1,2))
# Turin
rgrESSL_predsTur = np.amax(preds_EU[:,120-1:120+2,101-1:101+2], axis=(1,2))
# Barcelone
rgrESSL_predsBar = np.amax(preds_EU[:,134-1:134+2,79-1:79+2], axis=(1,2))
# Malta
rgrESSL_predsMal = np.amax(preds_EU[:,157-1:157+2,128-1:128+2], axis=(1,2))
# Paris
rgrESSL_predsPar = np.amax(preds_EU[:,105-1:105+2,80-1:80+2], axis=(1,2))
# Zurich
rgrESSL_predsZur = np.amax(preds_EU[:,110-1:110+2,104-1:104+2], axis=(1,2))
# Ljubljana
rgrESSL_predsLju = np.amax(preds_EU[:,116-1:116+2,128-1:128+2], axis=(1,2))
# Roma
rgrESSL_predsRom = np.amax(preds_EU[:,132-1:132+2,120-1:120+2], axis=(1,2))
# Berlin
rgrESSL_predsBer = np.amax(preds_EU[:,90-1:90+2,124-1:124+2], axis=(1,2))
# Milan
rgrESSL_predsMil = np.amax(preds_EU[:,118-1:118+2,107-1:107+2], axis=(1,2))
# Venice
rgrESSL_predsVen = np.amax(preds_EU[:,118-1:118+2,119-1:119+2], axis=(1,2))

# Group all these observations in the same array 
rgrpredsCities = [rgrESSL_predsMun, rgrESSL_predsWar, rgrESSL_predsGra, rgrESSL_predsTur,
                  rgrESSL_predsBar, rgrESSL_predsMal, rgrESSL_predsPar, rgrESSL_predsZur, 
                  rgrESSL_predsLju, rgrESSL_predsRom, rgrESSL_predsBer, 
                  rgrESSL_predsMil, rgrESSL_predsVen]

hailpreds_cities = np.column_stack(rgrpredsCities)
# hailpreds_cities = hailpreds_cities[0:2191,:]
print(hailpreds_cities.shape)

# Original ERA 5 model
# Munich
rgrESSL_predsMun_o =  np.amax(preds_EU_o[:,107-1:107+2,116-1:116+2], axis=(1,2))
# Warsaw
rgrESSL_predsWar_o = np.amax(preds_EU_o[:,91-1:91+2,154-1:154+2], axis=(1,2))
# Graz
rgrESSL_predsGra_o = np.amax(preds_EU_o[:,112-1:112+2,132-1:132+2], axis=(1,2))
# Turin
rgrESSL_predsTur_o = np.amax(preds_EU_o[:,120-1:120+2,101-1:101+2], axis=(1,2))
# Barcelone
rgrESSL_predsBar_o = np.amax(preds_EU_o[:,134-1:134+2,79-1:79+2], axis=(1,2))
# Malta
rgrESSL_predsMal_o = np.amax(preds_EU_o[:,157-1:157+2,128-1:128+2], axis=(1,2))
# Paris
rgrESSL_predsPar_o = np.amax(preds_EU_o[:,105-1:105+2,80-1:80+2], axis=(1,2))
# Zurich
rgrESSL_predsZur_o = np.amax(preds_EU_o[:,110-1:110+2,104-1:104+2], axis=(1,2))
# Ljubljana
rgrESSL_predsLju_o = np.amax(preds_EU_o[:,116-1:116+2,128-1:128+2], axis=(1,2))
# Roma
rgrESSL_predsRom_o = np.amax(preds_EU_o[:,132-1:132+2,120-1:120+2], axis=(1,2))
# Berlin
rgrESSL_predsBer_o = np.amax(preds_EU_o[:,90-1:90+2,124-1:124+2], axis=(1,2))
# Milan
rgrESSL_predsMil_o = np.amax(preds_EU_o[:,118-1:118+2,107-1:107+2], axis=(1,2))
# Venice
rgrESSL_predsVen_o = np.amax(preds_EU_o[:,118-1:118+2,119-1:119+2], axis=(1,2))

# Group all these observations in the same array 
rgrpredsCities2 = [rgrESSL_predsMun_o, rgrESSL_predsWar_o, rgrESSL_predsGra_o, rgrESSL_predsTur_o,
                  rgrESSL_predsBar_o, rgrESSL_predsMal_o, rgrESSL_predsPar_o, rgrESSL_predsZur_o, 
                  rgrESSL_predsLju_o, rgrESSL_predsRom_o, rgrESSL_predsBer_o, 
                  rgrESSL_predsMil_o, rgrESSL_predsVen_o]

hailpreds_cities_o = np.column_stack(rgrpredsCities2)
# hailpreds_cities = hailpreds_cities[0:1826,:]
print(hailpreds_cities_o.shape)

# Munich
rgrESSLobsMun =  np.amax(rgrESSLObs[0,:,107-1:107+2,116-1:116+2], axis=(1,2))
# Warsaw
rgrESSLobsWar = np.amax(rgrESSLObs[0,:,91-1:91+2,154-1:154+2], axis=(1,2))
# Graz
rgrESSLobsGra = np.amax(rgrESSLObs[0,:,112-1:112+2,132-1:132+2], axis=(1,2))
# Turin
rgrESSLobsTur = np.amax(rgrESSLObs[0,:,120-1:120+2,101-1:101+2], axis=(1,2))
# Barcelone
rgrESSLobsBar = np.amax(rgrESSLObs[0,:,134-1:134+2,79-1:79+2], axis=(1,2))
# Malta
rgrESSLobsMal = np.amax(rgrESSLObs[0,:,157-1:157+2,128-1:128+2], axis=(1,2))
# Paris
rgrESSLobsPar = np.amax(rgrESSLObs[0,:,105-1:105+2,80-1:80+2], axis=(1,2))
# Zurich
rgrESSLobsZur = np.amax(rgrESSLObs[0,:,110-1:110+2,104-1:104+2], axis=(1,2))
# Ljubljana
rgrESSLobsLju = np.amax(rgrESSLObs[0,:,116-1:116+2,128-1:128+2], axis=(1,2))
# Roma
rgrESSLobsRom = np.amax(rgrESSLObs[0,:,132-1:132+2,120-1:120+2], axis=(1,2))
# Berlin
rgrESSLobsBer = np.amax(rgrESSLObs[0,:,90-1:90+2,124-1:124+2], axis=(1,2))
# Milan
rgrESSLobsMil = np.amax(rgrESSLObs[0,:,118-1:118+2,107-1:107+2], axis=(1,2))
# Venice
rgrESSLobsVen = np.amax(rgrESSLObs[0,:,118-1:118+2,119-1:119+2], axis=(1,2))


# Group all these observations in the same array 
rgrobsCities2 = [rgrESSLobsMun, rgrESSLobsWar, rgrESSLobsGra, rgrESSLobsTur,
                  rgrESSLobsBar, rgrESSLobsMal, rgrESSLobsPar, rgrESSLobsZur,
                rgrESSLobsLju, rgrESSLobsRom, rgrESSLobsBer, rgrESSLobsMil, 
                rgrESSLobsVen]

hailobs_cities = np.column_stack(rgrobsCities2)
print(hailobs_cities.shape)

# Create arrays 
# For CITIES IN EUROPE

yearmonth_cities_obs = np.zeros((len(iYears), 12, 13))
yearmonth_cities_pred = np.zeros((len(iYears), 12, 13))
yearmonth_cities_pred_o = np.zeros((len(iYears), 12, 13))
yearmonth_cities_pred_oINT = np.zeros((len(iYears), 12, 13))
for yy in range(len(iYears)):
    for mm in range(12):
        # US
        yearmonth_cities_obs[yy,mm,:] = np.sum(hailobs_cities[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred[yy,mm,:] = np.sum(hailpreds_cities[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred_o[yy,mm,:] = np.sum(hailpreds_cities_o[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        
monthly_cities_obs=np.mean(yearmonth_cities_obs, axis=0)
yearly_cities_obs=np.sum(yearmonth_cities_obs, axis=1)

monthly_cities_pred=np.mean(yearmonth_cities_pred, axis=0)
yearly_cities_pred=np.sum(yearmonth_cities_pred, axis=1)
                         
monthly_cities_pred_o=np.mean(yearmonth_cities_pred_o, axis=0)
yearly_cities_pred_o=np.sum(yearmonth_cities_pred_o, axis=1)
                         
print('done')

%matplotlib inline
fig = plt.figure(figsize=(10,6))
labels = ['Munich', 'Warsaw', 'Graz', 'Turin', 'Barcelona', 'Malta', 'Paris', 'Zurich', 'Ljubljana', 'Rome', 'Berlin', 'Milan', 'Venice']
months_to_label = [2, 4, 6, 8, 10, 12]  # February, April, June, August, October, December
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

for ii in range(len(labels)):
    gs= gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # plot predictions xgboost
    preds1 = monthly_cities_pred[:,ii]
    # monthly_cities_pred6 = yearmonth_cities_pred[6,:,ii]
    plt.plot(iMonths, preds1, label='Predictions xgboost', color='green')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds1)
    hailmod5 = np.percentile(yearmonth_cities_pred[:,:,ii], 5, axis=0)
    hailmod95 = np.percentile(yearmonth_cities_pred[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y1 = hailmod5
    y2 = hailmod95

    plt.grid()
    ax1.fill_between(x,y1, y2, color='green', alpha=0.3)
    plt.ylabel('Average number of hail events')
    
    # plot predictions ERA 5
    preds2 = monthly_cities_pred_o[:,ii]
    # monthly_cities_pred6 = yearmonth_cities_pred[6,:,ii]
    plt.plot(iMonths, preds2, label='Predictions ERA 5', color='steelblue')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds2)
    hailmod5 = np.percentile(yearmonth_cities_pred_o[:,:,ii], 5, axis=0)
    hailmod95 = np.percentile(yearmonth_cities_pred_o[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y3 = hailmod5
    y4 = hailmod95

    plt.grid()
    ax1.fill_between(x,y3, y4, color='steelblue', alpha=0.3)
    plt.ylabel('Average number of hail events')

    # plot observations
    obs1 = monthly_cities_obs[:,ii]
    # monthly_cities_obs6 = yearmonth_cities_obs[6,:,ii]
    plt.plot(iMonths, obs1, label='Observations',  color='orange')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_obs=np.mean(obs1)
    hailobs5 = np.percentile(yearmonth_cities_obs[:,:,ii], 5, axis=0)
    hailobs95 = np.percentile(yearmonth_cities_obs[:,:,ii], 95, axis=0)   

    y7 = hailobs5
    y8 = hailobs95
    # plt.ylabel('time')
    ax1.fill_between(x,y7, y8, color='orange', alpha=0.3)
    plt.xlabel('Month')

    plt.xticks(months_to_label, [month_names[i - 1] for i in months_to_label])
    plt.ylim([0,3])
    plt.yticks([0,1,2,3])
    plt.suptitle(f'{labels[ii]}')
    plt.grid()
    plt.legend()

    plt.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/EU/v2/{labels[ii]}2016_2022_v2.png')

#AU
# Load predicted data
XGBlHailProbabilityAU = np.zeros((len(rgdTimeDD),160,200))
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/AU/'

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy]) 
    sFileFin=sSaveDataDir + 'xgboost_predictions_australia_'+str(iYears[yy])+ '.nc'
    ncid=Dataset(sFileFin, mode='r')
    XGBlHailProbabilityAU[iyear:iyear+yearlength,:,:] = np.squeeze(ncid.variables['Hail_Probabilities'])
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()


%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

pcolormesh(rgrLonBoM,rgrLatBoM,np.sum(XGBlHailProbabilityAU>=0.5, axis=0)/len(iYears), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
#add city names
# Sydney
sydlat, sydlon = -33.9, 151.2
plt.text(sydlon, sydlat, 'Sydney',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#Brisbane
brilat, brilon = -27.5, 153
plt.text(brilon, brilat, 'Brisbane',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Melbourne
Melat, Melon = -37.8, 145.0
plt.text(Melon, Melat, 'Melbourne',
         horizontalalignment='left',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Perth
Perlat, Perlon = -31.96, 115.87
plt.text(Perlon, Perlat, 'Perth',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())



#plot the dots at each city location
plt.plot([sydlon,brilon,Melon,Perlon], [sydlat,brilat,Melat,Perlat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
         )
gl.ylocator = MultipleLocator(base=10.0)
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'
# fig.show()
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/AU/AU_xgboost_predictions_1959-2022.png')

%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

pcolormesh(rgrLonBoM,rgrLatBoM,np.sum(rgrTotalHailProbabilitynSRH[:,390:550,430:630] >= 0.5, axis=0)/(len(iYears)), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
#add city names
# Sydney
sydlat, sydlon = -33.9, 151.2
plt.text(sydlon, sydlat, 'Sydney',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

#Brisbane
brilat, brilon = -27.5, 153
plt.text(brilon, brilat, 'Brisbane',
         horizontalalignment='left',
         transform=ccrs.PlateCarree())

# Melbourne
Melat, Melon = -37.8, 145.0
plt.text(Melon, Melat, 'Melbourne',
         horizontalalignment='left',
         verticalalignment='top',
         transform=ccrs.PlateCarree())

# Perth
Perlat, Perlon = -31.96, 115.87
plt.text(Perlon, Perlat, 'Perth',
         horizontalalignment='right',
         transform=ccrs.PlateCarree())


gl.ylocator = MultipleLocator(base=10.0)
#plot the dots at each city location
plt.plot([sydlon,brilon,Melon,Perlon], [sydlat,brilat,Melat,Perlat],
         color='black', linewidth=0, marker='o', mfc='none',
         transform=ccrs.PlateCarree(),
         )
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'
# fig.show()
fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/AU/AU_predictions_original_model_1959-2022.png')

# #-------------------------------------------------
# #plot the Difference between models
# #-------------------------------------------------
hail_diff_colors = [(0.12, 0.22, 0.55),(0.15, 0.3, 0.65),(0.13, 0.38, 0.8), (0.33, 0.58, 0.9),(0.55, 0.75, 0.92),(0.7, 0.85, 0.95), (0.8, 0.92, 1),
                    (1, 1, 1),                                     
                    (1.0, 0.8, 0.8), (0.95, 0.7, 0.7), (0.9, 0.6, 0.6), (0.9, 0.5, 0.5), (0.8, 0.2, 0.2), (0.7, 0.1, 0.1), (0.5, 0.0, 0.0)
                   ]

hail_diff_cmap = ListedColormap(hail_diff_colors)

intervals = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
norm_diff = mcolors.BoundaryNorm(intervals, len(intervals) - 1)

rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

diff = np.sum(XGBlHailProbabilityAU >= 0.5, axis=0)/(len(iYears)) - np.sum(rgrTotalHailProbabilitynSRH[:,390:550,430:630] >= 0.5, axis=0)/(len(iYears))
pcolormesh(rgrLonBoM,rgrLatBoM, diff, cmap='RdBu_r', vmin=-2, vmax=2)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.ylocator = MultipleLocator(base=10.0)
# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(ticks=np.arange(-2, 2.5, 0.5), orientation='horizontal', shrink = 0.5, extend='both') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/AU/AU_predictions_difference_1959-2022.png')

rgrLonAct=rgrLonBoM
rgrLatAct=rgrLatBoM

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

#get the grid cells covered with land only
invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()
# 25°N corresponds to the border with Mexico and index 260
# 49°N corresponds to the border with Canada and index 164
# The original iLaN, iLaS are 150, 280
LSM_AU = LSM[ilaN:ilaS,iloE:iloW]

# For Australia
yearmonth_AUhail_obs = np.zeros((len(iYears), 12, rgrLonBoM.shape[0], rgrLonBoM.shape[1]))
yearmonth_AUhail_pred = np.zeros((len(iYears), 12, rgrLonBoM.shape[0], rgrLonBoM.shape[1]))
yearmonth_AUhail_pred_o = np.zeros((len(iYears), 12, rgrLonBoM.shape[0], rgrLonBoM.shape[1]))

predsAU = np.copy(XGBlHailProbabilityAU)
predsAU[predsAU>=0.5]=1
predsAU[predsAU<0.5]=0

# original model 
predsAU_o = rgrTotalHailProbabilitynSRH[:,ilaN:ilaS,iloE:iloW]
predsAU[predsAU_o>=0.5]=1
predsAU[predsAU_o<0.5]=0
# remove grid cells with water
predsAU[:,LSM_AU<0.5] = 0
predsAU_o[:,LSM_AU<0.5] = 0

rgrBoMObs2 = rgrBoMObs[0,:,:,:]
rgrBoMObs2[:,LSM_AU<0.5] = 0

for yy in range(len(iYears)):
    for mm in range(12):          
        # Australia
        yearmonth_AUhail_obs[yy,mm,:,:] = np.sum(rgrBoMObs2[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        yearmonth_AUhail_pred[yy,mm,:,:] = np.sum(predsAU[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        yearmonth_AUhail_pred_o[yy,mm,:,:] = np.sum(predsAU_o[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
    
monthly_AUhail_obs=np.mean(yearmonth_AUhail_obs, axis=0)
yearly_AUhail_obs=np.sum(yearmonth_AUhail_obs, axis=1)

monthly_AUhail_pred=np.mean(yearmonth_AUhail_pred, axis=0)
yearly_AUhail_pred=np.sum(yearmonth_AUhail_pred, axis=1)   

monthly_AUhail_pred_o=np.mean(yearmonth_AUhail_pred_o, axis=0)
yearly_AUhail_pred_o=np.sum(yearmonth_AUhail_pred_o, axis=1)   

# XGboost model
# Brisbane
rgrBoM_predsBri = np.amax(predsAU[:,80-1:80+2,182-1:182+2], axis=(1,2))
# Sydney
rgrBoM_predsSyd = np.amax(predsAU[:,105-1:105+2,175-1:175+2], axis=(1,2))
# Melbourne
rgrBoM_predsMel = np.amax(predsAU[:,121-1:121+2,150-1:150+2], axis=(1,2))
# Perth
rgrBoM_predsPer = np.amax(predsAU[:,98-1:98+2,33-1:33+2], axis=(1,2))
# Gold Coast
rgrBoM_predsGoCo = np.amax(predsAU[:,82-1:82+2,184-1:184+2], axis=(1,2))

# Group all these observations in the same array 
rgrpredsCities = [rgrBoM_predsBri, rgrBoM_predsSyd, rgrBoM_predsMel, rgrBoM_predsPer,
                  rgrBoM_predsGoCo]

hailpreds_cities = np.column_stack(rgrpredsCities)
# hailpreds_cities = hailpreds_cities[0:5844,:]
print(hailpreds_cities.shape)

# Original model
preds_AU_o = rgrTotalHailProbabilitynSRH[:,390:550,430:630] 
# Brisbane
rgrBoM_predsBri_o = np.amax(predsAU_o[:,80-1:80+2,182-1:182+2], axis=(1,2))
# Sydney
rgrBoM_predsSyd_o = np.amax(predsAU_o[:,105-1:105+2,175-1:175+2], axis=(1,2))
# Melbourne
rgrBoM_predsMel_o = np.amax(predsAU_o[:,121-1:121+2,150-1:150+2], axis=(1,2))
# Perth
rgrBoM_predsPer_o = np.amax(predsAU_o[:,98-1:98+2,33-1:33+2], axis=(1,2))
# Gold Coast
rgrBoM_predsGoCo_o = np.amax(predsAU_o[:,82-1:82+2,184-1:184+2], axis=(1,2))

# Group all these observations in the same array 
rgrpredsCities2 = [rgrBoM_predsBri_o, rgrBoM_predsSyd_o, rgrBoM_predsMel_o, rgrBoM_predsPer_o,
                  rgrBoM_predsGoCo_o]

hailpreds_cities_o = np.column_stack(rgrpredsCities2)
print(hailpreds_cities_o.shape)

# Observations
# Brisbane
rgrBoMobsBri = np.amax(rgrBoMObs[0,:,80-1:80+2,182-1:182+2], axis=(1,2))
# Sydney
rgrBoMobsSyn = np.amax(rgrBoMObs[0,:,105-1:105+2,175-1:175+2], axis=(1,2))
# Melbourne
rgrBoMobsMel = np.amax(rgrBoMObs[0,:,121-1:121+2,150-1:150+2], axis=(1,2))
# Perth
rgrBoMobsPer = np.amax(rgrBoMObs[0,:,98-1:98+2,33-1:33+2], axis=(1,2))
# Gold Coast
rgrBoMobsGoCo = np.amax(rgrBoMObs[0,:,82-1:82+2,184-1:184+2], axis=(1,2))


# Group all these observations in the same array 
rgrobsCities2 = [rgrBoMobsBri, rgrBoMobsSyn, rgrBoMobsMel, rgrBoMobsPer,
                rgrBoMobsGoCo]

hailobs_cities = np.column_stack(rgrobsCities2)
print(hailobs_cities.shape)

# Create arrays 
# For CITIES IN australia

yearmonth_cities_obs = np.zeros((len(iYears), 12, 5))
yearmonth_cities_pred = np.zeros((len(iYears), 12, 5))
yearmonth_cities_pred_o = np.zeros((len(iYears), 12, 5))
for yy in range(len(iYears)):
    for mm in range(12):
        # US
        yearmonth_cities_obs[yy,mm,:] = np.sum(hailobs_cities[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred[yy,mm,:] = np.sum(hailpreds_cities[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
        yearmonth_cities_pred_o[yy,mm,:] = np.sum(hailpreds_cities_o[(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:], axis=0)
       
    
monthly_cities_obs=np.mean(yearmonth_cities_obs, axis=0)
yearly_cities_obs=np.sum(yearmonth_cities_obs, axis=1)

monthly_cities_pred=np.mean(yearmonth_cities_pred, axis=0)
yearly_cities_pred=np.sum(yearmonth_cities_pred, axis=1)

monthly_cities_pred_o=np.mean(yearmonth_cities_pred_o, axis=0)
yearly_cities_pred_o=np.sum(yearmonth_cities_pred_o, axis=1)
print('done')

%matplotlib inline
fig = plt.figure(figsize=(10,6))
labels = ['Brisbane', 'Sydney', 'Melbourne', 'Perth', 'Gold Coast']
months_to_label = [2, 4, 6, 8, 10, 12]  # February, April, June, August, October, December
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

for ii in range(len(labels)):
    gs= gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # plot predictions xgboost model
    preds1 = monthly_cities_pred[:,ii]
    plt.plot(iMonths, preds1, label='Predictions xgboost', color='green')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds1)
    hailmod5 = np.percentile(yearmonth_cities_pred[:,:,ii], 5, axis=0)
    hailmod95 = np.percentile(yearmonth_cities_pred[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y1 = hailmod5
    y2 = hailmod95

    ax1.fill_between(x,y1, y2, color='green', alpha=0.3)
    plt.ylabel('Average number of hail events')
    
    # Plot predictions of original model
    preds_o = monthly_cities_pred_o[:,ii]
    plt.plot(iMonths, preds_o, label='Original Predictions', color='steelblue')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_pred=np.mean(preds_o)
    hailmod5_o = np.percentile(yearmonth_cities_pred_o[:,:,ii], 5, axis=0)
    hailmod95_o = np.percentile(yearmonth_cities_pred_o[:,:,ii], 95, axis=0)   
    x = np.arange(1,13,1)
    y3 = hailmod5_o
    y4 = hailmod95_o

    ax1.fill_between(x,y3, y4, color='steelblue', alpha=0.3)


    # plot observations
    obs1 = monthly_cities_obs[:,ii]
    plt.plot(iMonths, obs1, label='Observations', color='orange')
    # plot the shaded area
    # 95% interval using -+ 2 std
    mean_hail_obs=np.mean(obs1)
    hailobs5 = np.percentile(yearmonth_cities_obs[:,:,ii], 5, axis=0)
    hailobs95 = np.percentile(yearmonth_cities_obs[:,:,ii], 95, axis=0)   

    y5 = hailobs5
    y6 = hailobs95
    # plt.ylabel('time')
    ax1.fill_between(x,y5, y6, color='orange', alpha=0.3)
    plt.ylim(0, 3)
    plt.yticks([1,2,3])
    plt.xlabel('Month')
    plt.xticks(months_to_label, [month_names[i - 1] for i in months_to_label])
    plt.suptitle(f'{labels[ii]}')
    plt.grid()
    plt.legend()
    plt.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/AU/{labels[ii]}_annualAU_cycle_1990_2020.png')

%matplotlib inline
fig = plt.figure(figsize=(12,6))
gs= gridspec.GridSpec(1,1)
ax1 = plt.subplot(gs[0,0])
iMonths=np.unique(rgdTimeDD.month)
mean_AUobs = np.mean(np.sum(monthly_AUhail_obs[:,:,:], axis=(1,2)), axis=0)
mean_AUpred = np.mean(np.sum(monthly_AUhail_pred[:,:,:], axis=(1,2)), axis=0)
mean_AUpred_o = np.mean(np.sum(monthly_AUhail_pred_o[:,:,:], axis=(1,2)), axis=0)

plt.plot(iMonths, np.sum(monthly_AUhail_pred[:,:,:], axis=(1,2))/mean_AUpred, label='Predictions xgboost')
hailmod5 = np.percentile(np.sum(yearmonth_AUhail_pred[:,:,:,:],axis=(2,3)), 5, axis=0)
hailmod95 = np.percentile(np.sum(yearmonth_AUhail_pred[:,:,:,:], axis=(2,3)), 95, axis=0)   
x = np.arange(1,13,1)
y1 = hailmod5/mean_AUpred
y2 = hailmod95/mean_AUpred
ax1.fill_between(x,y1, y2, color='green', alpha=0.3)

plt.plot(iMonths, np.sum(monthly_AUhail_pred_o[:,:,:], axis=(1,2))/mean_AUpred_o, label='Original predictions')
hailmod5_o = np.percentile(np.sum(yearmonth_AUhail_pred_o[:,:,:,:],axis=(2,3)), 5, axis=0)
hailmod95_o = np.percentile(np.sum(yearmonth_AUhail_pred_o[:,:,:,:], axis=(2,3)), 95, axis=0)   
x = np.arange(1,13,1)
y3 = hailmod5_o/mean_AUpred_o
y4 = hailmod95_o/mean_AUpred_o
ax1.fill_between(x,y3, y4, color='steelblue', alpha=0.3)


plt.plot(iMonths, np.sum(monthly_AUhail_obs[:,:,:], axis=(1,2))/mean_AUobs, label='Observations')
plt.xlabel('time')
plt.ylabel('Normalized frequency')
# plt.title('Observations')
hailobs5 = np.percentile(np.sum(yearmonth_AUhail_obs[:,:,:,:], axis=(2,3)), 5, axis=0)
hailobs95 = np.percentile(np.sum(yearmonth_AUhail_obs[:,:,:,:], axis=(2,3)), 95, axis=0)   
y5 = hailobs5/mean_AUobs
y6 = hailobs95/mean_AUobs

plt.ylim(0, 5)
plt.yticks([1,2,3,4,5])

plt.grid()
ax1.fill_between(x,y5, y6, color='orange', alpha=0.3)
plt.legend()
plt.title('AU annual cycle')
plt.savefig(f'/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/AU/AU_annual_cycle_1990_2020.png')

# X_test

X_land_test = np.copy(rgrERAVarall)
print(X_land_test.shape)
# Load the model that has been calculated 
model_xgb2 = xgb.Booster()
model_xgb2.load_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/US_XGBmodel_iteration0.jason')
yearlength = 365 + isleap(iYears[yy])

# Create regression matrices
X_test2 = X_land_test[iyear:iyear+yearlength,:,:,:]
X_test2 = np.moveaxis(X_test2,1,0)
input1 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3])
input1[1,:] = np.abs(input1[1,:])
input1[5,:] = np.abs(input1[5,:])
X_test =  np.moveaxis(input1, 1, 0)
print(X_test.shape)
xg_test = xgb.DMatrix(X_test)
preds_test = model_xgb2.predict(xg_test)
preds_test2 = np.copy(preds_test)
np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_world_{iYears[yy]}.npy', preds_test2)



index=0
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/world/'

# Make predictions on the test dataset
for yy in range(11):
    dStartDayy=datetime.datetime(iYears[index], 1, 1,0)
    dStopDayy=datetime.datetime(iYears[index], 12, 31,23)
    rgdTimeDDy = pd.date_range(dStartDayy, end=dStopDayy, freq='d')
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDDy) == 1)[0]
    
    yearlength = 365 +isleap(iYears[index])
    filename = '/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_World'+str(yy)+'.npy'
    preds_world = np.load(filename)  
    preds_world2 = preds_world.reshape(int(preds_world.shape[0]/(721*1440)), 721, 1440)
    preds_world3 = preds_world2[0:yearlength,:,:]
    preds_world4 = preds_world2[yearlength::,:,:]
    print(preds_world2.shape)
    
    sFileFin=sSaveDataDir + 'xgboost_predictions_world_'+str(iYears[index])+ '.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatERA.shape[1])
    root_grp.createDimension('latitude', rgrLonERA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatERA[:,0]
    lon[:]=rgrLonERA[0,:]
    Hail_Probabilities[:]=preds_world3
    time[:]=iTime
    root_grp.close()
    print(f"Predictions made on year {iYears[index]}")


    dStartDayy=datetime.datetime(iYears[index+1], 1, 1,0)
    dStopDayy=datetime.datetime(iYears[index+1], 12, 31,23)
    rgdTimeDDy = pd.date_range(dStartDayy, end=dStopDayy, freq='d')
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDDy) == 1)[0]
    sFileFin=sSaveDataDir + 'xgboost_predictions_world_'+str(iYears[index+1])+ '.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatERA.shape[1])
    root_grp.createDimension('latitude', rgrLonERA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatERA[:,0]
    lon[:]=rgrLonERA[0,:]
    Hail_Probabilities[:]=preds_world4
    time[:]=iTime
    root_grp.close()
    
    print(f"Predictions made on year {iYears[index+1]}")
    index = index +2

# X_test

X_land_test = np.copy(rgrERAVarall)
print(X_land_test.shape)
# Load the model that has been calculated 
model_xgb2 = xgb.Booster()
model_xgb2.load_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/US_XGBmodel_iteration0_v2.jason')
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/world/'
# Make predictions on the test dataset
for yy in range(len(iYears)):
    yearlength = 365 + isleap(iYears[yy])
    
    # Create regression matrices
    X_test2 = X_land_test[iyear:iyear+yearlength,:,:,:]
    X_test2 = np.moveaxis(X_test2,1,0)
    input1 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3])
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test =  np.moveaxis(input1, 1, 0)
    print(X_test.shape)
    xg_test = xgb.DMatrix(X_test)
    preds_test = model_xgb2.predict(xg_test)
    preds_test2 = np.copy(preds_test)
    np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_world_{iYears[yy]}_v2.npy', preds_test2)
    
    dStartDayy=datetime.datetime(iYears[yy], 1, 1,0)
    dStopDayy=datetime.datetime(iYears[yy], 12, 31,23)
    rgdTimeDDy = pd.date_range(dStartDayy, end=dStopDayy, freq='d')
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDDy) == 1)[0]
    
    preds_world3 = preds_test2.reshape(yearlength,721,1440)
    sFileFin=sSaveDataDir + 'xgboost_predictions_world_'+str(iYears[yy])+ '.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatERA.shape[1])
    root_grp.createDimension('latitude', rgrLonERA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatERA[:,0]
    lon[:]=rgrLonERA[0,:]
    Hail_Probabilities[:]=preds_world3
    time[:]=iTime
    root_grp.close()

    print(f"Predictions made on year {iYears[yy]}")

# World
# Load predicted data
preds_world = np.zeros((len(rgdTimeDD),721,1440))
index = 0
for ii in range(11):
    preds_test = np.load(f"/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_World{ii}.npy")
    preds_test2 = preds_test.reshape(int(preds_test.shape[0]/(721*1440)), 721, 1440)

    preds_world[index:index+preds_test2.shape[0],:,:] = preds_test2
    index = index + preds_test2.shape[0]
preds_world = preds_world[0:len(rgdTimeDD),:,:]  

%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

pcolormesh(rgrLonERA,rgrLatERA,np.sum(preds_world>=0.5, axis=0)/len(iYears), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/World_predictions_1959-2022.png')


%matplotlib inline
rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

pcolormesh(rgrLonERA,rgrLatERA,np.sum(rgrTotalHailProbabilitynSRH>=0.5, axis=0)/len(iYears), cmap=hail_prob_cmap, norm=norm)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(orientation='horizontal', shrink = 0.5, extend='max') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/World_predictions_original_model_1959-2022.png')


# #-------------------------------------------------
# #plot the Difference between models
# #-------------------------------------------------
hail_diff_colors = [(0.12, 0.22, 0.55),(0.15, 0.3, 0.65),(0.13, 0.38, 0.8), (0.33, 0.58, 0.9),(0.55, 0.75, 0.92),(0.7, 0.85, 0.95), (0.8, 0.92, 1),
                    (1, 1, 1),                                     
                    (1.0, 0.8, 0.8), (0.95, 0.7, 0.7), (0.9, 0.6, 0.6), (0.9, 0.5, 0.5), (0.8, 0.2, 0.2), (0.7, 0.1, 0.1), (0.5, 0.0, 0.0)
                   ]

hail_diff_cmap = ListedColormap(hail_diff_colors)

intervals = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
norm_diff = mcolors.BoundaryNorm(intervals, len(intervals) - 1)

rgrLonPlots = np.copy(rgrLonERA)
rgrLonPlots[rgrLonERA < 0] = rgrLonPlots[rgrLonERA < 0] + 360
#map hail probabily
#create new figure
fig = plt.figure(figsize=(18, 10))
plt.rcParams.update({'font.size': 15})
gs1 = gridspec.GridSpec(1,1)
gs1.update(left = 0.05, right = 0.95,
          bottom = 0.05, top = 0.95,
          wspace = 0.05, hspace = 0.05)
ax = plt.subplot(gs1[0,0],projection=ccrs.PlateCarree())

diff = np.sum(preds_world >= 0.5, axis=0)/(len(iYears)) - np.sum(rgrTotalHailProbabilitynSRH >= 0.5, axis=0)/(len(iYears))
pcolormesh(rgrLonPlots,rgrLatERA, diff, cmap='RdBu_r', vmin=-2, vmax=2)

#add latitude and longitude labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

# axis style
gl.xlabel_style = {'size': 11, 'color': 'black'}
gl.ylabel_style = {'size': 11, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

ax.coastlines()
plt.colorbar(ticks=np.arange(-2, 2.5, 0.5), orientation='horizontal', shrink = 0.5, extend='both') #, label='Modeled hail days per year'

fig.savefig('/glade/u/home/bblanc/Hail_Project_Extension/Hail_model/images/World_predictions_difference_1959-2022.png')

dStartDay=datetime.datetime(2022, 1, 1,0)
dStopDay=datetime.datetime(2022, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')
rgdFullTime_day=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='d')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)


# US V2
# define the input and output variables.
rgrLonAct=rgrLonNOAA
rgrLatAct=rgrLatNOAA

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]
rgrERAVarall_US = rgrERAVarall[:,:,ilaN:ilaS,iloE:iloW]

# Load the model that has been calculated 
model_xgb2 = xgb.Booster()
model_xgb2.load_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/US_XGBmodel_iteration0_v2.jason')

iyear = 0
# Make predictions on the test dataset
for yy in range(len(iYears)):
    yearlength = 365 + isleap(iYears[yy])
    # Create regression matrices
    X_test2 = rgrERAVarall_US[iyear:iyear+yearlength,:,:,:]
    X_test2 = np.moveaxis(X_test2,1,0)
    input1 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3])
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test =  np.moveaxis(input1, 1, 0)
    iyear=iyear+yearlength
    xg_test = xgb.DMatrix(X_test)
    preds_test = model_xgb2.predict(xg_test)
    preds_test2 = np.copy(preds_test)
    np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_US_{iYears[yy]}_v2.npy', preds_test2)
    print(f"Predictions made on year {iYears[yy]}")

# X_test
# define the input and output variables.
rgrLonAct=rgrLonNOAA
rgrLatAct=rgrLatNOAA

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]

ilaS=rgiDomain[2]
rgrERAVarall_US = rgrERAVarall[:,:,ilaN:ilaS,iloE:iloW]
X_land_test = np.copy(rgrERAVarall_US)
print(X_land_test.shape)
# Load the model that has been calculated 
model_xgb2 = xgb.Booster()
model_xgb2.load_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/US_XGBmodel_iteration0_v2.jason')
iyear = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/US/'
# Make predictions on the test dataset
for yy in range(len(iYears)):
    yearlength = 365 + isleap(iYears[yy])
    
    # Create regression matrices
    X_test2 = X_land_test[iyear:iyear+yearlength,:,:,:]
    X_test2 = np.moveaxis(X_test2,1,0)
    input1 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3])
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test =  np.moveaxis(input1, 1, 0)
    print(X_test.shape)
    xg_test = xgb.DMatrix(X_test)
    preds_test = model_xgb2.predict(xg_test)
    preds_test2 = np.copy(preds_test)
    np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_US_{iYears[yy]}_v2.npy', preds_test2)
    
    dStartDayy=datetime.datetime(iYears[yy], 1, 1,0)
    dStopDayy=datetime.datetime(iYears[yy], 12, 31,23)
    rgdTimeDDy = pd.date_range(dStartDayy, end=dStopDayy, freq='d')
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDDy) == 1)[0]
    
    preds_US3 = preds_test2.reshape(yearlength,130,300)
    sFileFin=sSaveDataDir + 'xgboost_predictions_CONUS_'+str(iYears[yy])+ '_v2.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatNOAA.shape[1])
    root_grp.createDimension('latitude', rgrLonNOAA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatNOAA[:,0]
    lon[:]=rgrLonNOAA[0,:]
    Hail_Probabilities[:]=preds_US3
    time[:]=iTime
    root_grp.close()

    print(f"Predictions made on year {iYears[yy]}")


# US
index = 0
sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/US/'

for yy in range(11):
    dStartDay=datetime.datetime(iYears[index], 1, 1,0)
    dStopDay=datetime.datetime(iYears[index], 12, 31,23)
    rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')

    yearlength = 365 + isleap(iYears[index])
    preds_US = np.load(f"/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_US_{iYears[index]}_{iYears[index+1]}_v2.npy")  
    preds_US2 = preds_US.reshape(int(preds_US.shape[0]/(130*300)), 130, 300)
    if index%4==0:
        preds_US3 = preds_US2[0:366,:,:]
        preds_US4 = preds_US2[365::,:,:]
    else:
        preds_US3 = preds_US2[0:365,:,:]
        preds_US4 = preds_US2[366::,:,:]
    print(preds_US2.shape)
    print(preds_US3.shape)
    print(preds_US4.shape)
        
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDD) == 1)[0]
    sFileFin=sSaveDataDir + 'xgboost_predictions_CONUS_'+str(iYears[index])+ '_v2.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatNOAA.shape[1])
    root_grp.createDimension('latitude', rgrLonNOAA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatNOAA[:,0]
    lon[:]=rgrLonNOAA[0,:]
    Hail_Probabilities[:]=preds_US3
    time[:]=iTime
    root_grp.close()
    print (f'FINISHED {iYears[index]}')
    

    StartDay=datetime.datetime(iYears[index+1], 1, 1,0)
    dStopDay=datetime.datetime(iYears[index+1], 12, 31,23)
    rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')

    yearlength = 365 + isleap(iYears[index+1])

        
    iTime=np.where(np.isin(rgdFullTime, rgdTimeDD) == 1)[0]
    sFileFin=sSaveDataDir + 'xgboost_predictions_CONUS_'+str(iYears[index+1])+ '_v2.nc' 
    root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', rgrLatNOAA.shape[1])
    root_grp.createDimension('latitude', rgrLonNOAA.shape[0])
    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    time = root_grp.createVariable('time', 'f8', ('time',))
    Hail_Probabilities = root_grp.createVariable('Hail_Probabilities', 'f4', ('time','latitude','longitude',),fill_value=-99999)
    time.calendar = "gregorian"
    time.units = "hours since 1959-1-1 00:00:00"
    time.standard_name = "time"
    time.long_name = "time"
    time.axis = "T"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    # write data to netcdf
    lat[:]=rgrLatNOAA[:,0]
    lon[:]=rgrLonNOAA[0,:]
    Hail_Probabilities[:]=preds_US4
    time[:]=iTime
    root_grp.close()
    print (f'FINISHED {iYears[index+1]}')
    
    index = index+2

## Annual cycle
## ERA 5
rgrHailSaveDirnSRH='/glade/scratch/bblanc/ERA5_hail_model/Hail_Probabilities_final/HailProbabilities-ERA-5_4_qT-4_qD-26_CONUS_test_absSRH/ff'
iyear=0
iyears=0
rgrTotalHailProbabilitynSRH = np.zeros((rgdTimeDD.shape[0], 721, 1440))
rgrLonAct=rgrLonNOAA
rgrLatAct=rgrLatNOAA

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]
iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

#get the grid cells covered with land only
invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()

LSM_US = LSM[164:260,iloE:iloW]

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy])
    sFileName = rgrHailSaveDirnSRH+'/'+str(iYears[yy])+'_HailProbabilities_ERA-5_hourly.nc'
    ncid=Dataset(sFileName, mode='r')
    rgrTotalHailProbabilitynSRH[iyear:iyear+yearlength,:,:] = np.squeeze(ncid.variables['HailProb'])
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()

preds_US2 = np.zeros((rgdTimeDD.shape[0],130,300))
iyear=0

sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/US/'

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy])
    sFileName = sSaveDataDir+'/' +'xgboost_predictions_CONUS_'+str(iYears[yy])+ '_v2.nc' 
    ncid=Dataset(sFileName, mode='r')
    preds_US = np.squeeze(ncid.variables['Hail_Probabilities'])
    preds_US = preds_US[0:yearlength,:,:]
    preds_US2[iyear:iyear+yearlength,:,:] = preds_US
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()
    

# ------------------------------------------------------------------
# Create  test dataset containing cities obs
# And only for a few years so that we test on unseen data 
# Look at the observation in the main cities 
# ------------------------------------------------------------------
preds_US[preds_US>=0.5] = 1
preds_US[preds_US<0.5]=0

preds_US2 = preds_US[:,14:110,:]


preds_US_o = rgrTotalHailProbabilitynSRH[:,164:260,iloE:iloW]
preds_US_o[preds_US_o>=0.5] = 1
preds_US_o[preds_US_o<0.5] = 0

# remove grid cells with water
preds_US2[:,LSM_US<0.5] = 0
preds_US_o[:,LSM_US<0.5] = 0

preds_US2 = np.zeros((rgdTimeDD.shape[0],130,300))
iyear=0

sSaveDataDir='/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/ncdf/US/'

for yy in range(len(iYears)):
    yearlength=365 + isleap(iYears[yy])
    sFileName = sSaveDataDir+'/' +'xgboost_predictions_CONUS_'+str(iYears[yy])+ '_v2.nc' 
    ncid=Dataset(sFileName, mode='r')
    preds_US = np.squeeze(ncid.variables['Hail_Probabilities'])
    preds_US = preds_US[0:yearlength,:,:]
    preds_US2[iyear:iyear+yearlength,:,:] = preds_US
    print(str(iYears[yy]))
    iyear=iyear+yearlength
    ncid.close()
    

